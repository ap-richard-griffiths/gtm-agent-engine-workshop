import importlib
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langsmith import tracing_context

with patch("dotenv.load_dotenv"), patch.dict(
    "os.environ", {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}
):
    agent = importlib.import_module("gtm_agent.gtm_agent")

from gtm_agent import data_service


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.cache = patch.dict(data_service._PROFILES, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.tracing = tracing_context(enabled=False)
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)
        self.pid = next(iter(data_service.PROSPECTS))
        self.record = deepcopy(data_service.PROSPECTS[self.pid])
        self.offering = next(iter(data_service.OFFERINGS.values()))

    def assertPublic(self, value):
        serialized = json.dumps(value)
        self.assertNotIn("billing_qualification", serialized)
        for sensitive in self.record["billing_qualification"].values():
            self.assertNotIn(sensitive, serialized)

    def test_contact_allow_list(self):
        with patch.dict(data_service.PROSPECTS[self.pid], {"future_private_field": "private"}):
            result = agent.get_prospect.invoke({"prospect_id": self.pid})
        self.assertEqual(result, {
            "prospect": {
                "prospect_id": self.pid,
                **{key: self.record[key] for key in ("name", "email", "disqualified")},
            },
            "found": True,
        })

    def test_new_profiles_and_scoring_arguments_are_public(self):
        for pid, record in data_service.PROSPECTS.items():
            with self.subTest(prospect_id=pid):
                result = agent.build_prospect_profile.invoke({"prospect_id": pid})
                profile = result["prospect_profile"]
                self.assertTrue(result["found"])
                self.assertPublic(result)
                self.assertPublic(data_service._PROFILES[pid])
                self.assertPublic({"prospect_profile": profile, "offering": self.offering})
                self.assertEqual(profile, {
                    "prospect_id": pid,
                    **{key: value for key, value in record.items() if key != "billing_qualification"},
                })
                self.assertIn("billing_qualification", record)

    def test_legacy_cache_is_sanitized_before_traced_return(self):
        legacy = {"prospect_id": self.pid, **self.record}
        data_service._PROFILES[self.pid] = legacy
        runs = []
        with tracing_context(enabled=True, client=Mock()):
            result = data_service.get_profile_from_db(
                self.pid, langsmith_extra={"on_end": runs.append}
            )
        self.assertEqual(len(runs), 1)
        self.assertPublic(runs[0].outputs)
        self.assertPublic(result)
        self.assertPublic(data_service._PROFILES[self.pid])
        self.assertIn("billing_qualification", legacy)
        cached = agent.build_prospect_profile.invoke({"prospect_id": self.pid})
        self.assertEqual(cached, {**result, "found": True})
        self.assertEqual(cached["prospect_profile"]["annual_revenue"], self.record["annual_revenue"])

    def test_scoring_sanitizes_legacy_input_and_preserves_fit_data(self):
        profile = {"prospect_id": self.pid, **self.record, "segment": "Enterprise"}
        score = {"score": 80, "justification": "Good fit"}
        llm = Mock()
        llm.invoke.return_value.model_dump.return_value = score
        with patch.object(agent, "_scoring_llm", llm):
            result = agent.score_prospect.invoke({
                "prospect_profile": {**profile, "tech_stack": ["unverified"]},
                "offering": self.offering,
            })
        self.assertEqual(result, score)
        messages = llm.invoke.call_args.args[0]
        self.assertPublic(messages)
        user = messages[1]["content"]
        offering, serialized = user.removeprefix("Offering:\n").split("\n\nProspect profile:\n")
        self.assertEqual(json.loads(offering), self.offering)
        self.assertEqual(json.loads(serialized), {
            key: value for key, value in profile.items() if key != "billing_qualification"
        })
        self.assertIn("billing_qualification", profile)

    def test_scoring_without_prospect_id_is_sanitized(self):
        llm = Mock()
        with patch.object(agent, "_scoring_llm", llm):
            agent.score_prospect.invoke({"prospect_profile": self.record, "offering": self.offering})
        self.assertPublic(llm.invoke.call_args.args[0])

    def test_missing_records_and_invalid_offering(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}), {
            "prospect": None, "found": False,
        })
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}), {
            "prospect_profile": None, "found": False,
        })
        self.assertEqual(agent.score_prospect.invoke({"prospect_profile": {}}), {
            "score": None, "error": "Cannot score without a valid offering.",
        })

    def test_contact_still_supports_email(self):
        prospect = agent.get_prospect.invoke({"prospect_id": self.pid})["prospect"]
        runtime = SimpleNamespace(config={"metadata": {"user_id": data_service.REP_IDS[0]["rep_id"]}})
        result = agent.send_prospect_email.func(prospect, "Hello", "Let's talk", runtime)
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], self.record["email"])
        self.assertEqual(result["to_name"], self.record["name"])
        self.assertEqual(result["from"], data_service.REP_IDS[0]["email"])
        self.assertEqual(result["body"], "Let's talk")
        self.assertPublic(result)


if __name__ == "__main__":
    unittest.main()
