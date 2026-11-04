import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch


with patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class SendProspectEmailTests(unittest.TestCase):
    def send(self, prospect):
        return agent.send_prospect_email.func(
            prospect=prospect,
            subject="Demo invitation",
            body="Would you like a demo?",
            runtime=SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}}),
        )

    def test_disqualified_prospect_is_blocked(self):
        result = self.send({
            "prospect_id": "LEAD-50001",
            "email": "other@example.com",
            "disqualified": False,
        })
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect is disqualified; email not sent.",
        })

    def test_unknown_or_missing_prospect_id_fails(self):
        for prospect in ({"prospect_id": "unknown", "email": "other@example.com"},
                         {"email": "other@example.com"}):
            with self.subTest(prospect=prospect):
                self.assertEqual(self.send(prospect), {
                    "status": "failed", "error": "Unknown prospect.",
                })

    def test_eligible_prospect_uses_source_contact(self):
        source = agent.data_service.get_prospect_record("LEAD-12853")
        result = self.send({
            "prospect_id": "LEAD-12853",
            "email": "other@example.com",
            "name": "Wrong name",
        })
        self.assertEqual(result["status"], "sent")
        self.assertTrue(result["message_id"])
        self.assertEqual(result["to"], source["email"])
        self.assertEqual(result["to_name"], source["name"])
        self.assertEqual(result["from"], "ava.mills@northpoint.com")
        self.assertEqual(result["subject"], "Demo invitation")
        self.assertEqual(result["body"], "Would you like a demo?")

    def test_source_without_email_fails(self):
        with patch.object(agent.data_service, "get_prospect_record", return_value={
            "name": "Eligible prospect", "disqualified": False,
        }):
            self.assertEqual(self.send({
                "prospect_id": "LEAD-12853", "email": "other@example.com",
            }), {
                "status": "failed", "error": "Prospect record has no email address.",
            })
