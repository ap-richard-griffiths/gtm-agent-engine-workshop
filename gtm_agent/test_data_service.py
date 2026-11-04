import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from . import data_service
from . import gtm_agent


class ProspectUpdateTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-71001"
        records = deepcopy(data_service.PROSPECTS)
        self.records_patch = patch.dict(data_service.PROSPECTS, records, clear=True)
        self.records_patch.start()
        self.addCleanup(self.records_patch.stop)
        self.profiles_patch = patch.dict(data_service._PROFILES, {}, clear=True)
        self.profiles_patch.start()
        self.addCleanup(self.profiles_patch.stop)

    def test_update_persists_technology_and_preserves_response(self):
        original_stack = list(data_service.fetch_tech_stack(self.prospect_id))
        self.assertNotIn("Kafka", original_stack)

        result = gtm_agent.update_prospect_info.invoke({
            "prospect_id": self.prospect_id, "technology": "Kafka",
        })

        self.assertEqual(result, {
            "prospect_id": self.prospect_id,
            "technology": "Kafka",
            "updated": True,
            "found": True,
            "tech_stack": original_stack + ["Kafka"],
        })
        self.assertEqual(
            data_service.fetch_tech_stack(self.prospect_id),
            original_stack + ["Kafka"],
        )
        self.assertEqual(
            data_service.update_prospect_info(self.prospect_id, "Kafka"),
            {"updated": True, "found": True, "tech_stack": original_stack + ["Kafka"]},
        )

    def test_missing_prospect_preserves_response(self):
        self.assertEqual(
            data_service.update_prospect_info("missing", "Kafka"),
            {"updated": False, "found": False},
        )

    def test_profile_built_before_update_is_refreshed(self):
        before = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertNotIn("Kafka", before["prospect_profile"]["tech_stack"])

        data_service.update_prospect_info(self.prospect_id, "Kafka")
        after = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        self.assertTrue(after["found"])
        self.assertEqual(
            after["prospect_profile"]["tech_stack"],
            before["prospect_profile"]["tech_stack"] + ["Kafka"],
        )

    def test_scoring_after_update_uses_persisted_stack(self):
        profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        offering = gtm_agent.lookup_offering.invoke({"offering_id": "OFFER-10001"})

        def score_saved_stack(messages):
            supplied_profile = json.loads(messages[1]["content"].split("Prospect profile:\n")[1])
            missing = sorted(set(offering["offering"]["required_tech_stack"]) -
                             set(supplied_profile["tech_stack"]))
            return gtm_agent.ProspectScore(
                score=100 - 10 * len(missing),
                justification="Missing: " + ", ".join(missing),
                rubric_breakdown=gtm_agent.RubricBreakdown(
                    revenue_fit=100, tech_stack_match=100 - 10 * len(missing),
                    segment_fit=100,
                ),
            )

        arguments = {"prospect_profile": profile["prospect_profile"], "offering": offering["offering"]}
        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.side_effect = score_saved_stack
            before = gtm_agent.score_prospect.invoke(arguments)
            update = gtm_agent.update_prospect_info.invoke({
                "prospect_id": self.prospect_id, "technology": "Kafka",
            })
            after = gtm_agent.score_prospect.invoke(arguments)

        self.assertTrue(update["updated"])
        self.assertIn("Kafka", before["justification"])
        self.assertNotIn("Kafka", after["justification"])
        self.assertGreater(after["score"], before["score"])
