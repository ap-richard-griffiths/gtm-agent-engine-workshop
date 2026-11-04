"""Offline checks for the GTM agent's model and tool boundary."""

import importlib
import json
import os
import unittest
import uuid
from unittest.mock import patch

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI


class AgentTests(unittest.TestCase):
    def test_application_tools_and_runtime(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
            app = importlib.import_module("gtm_agent.gtm_agent")
            rep = app.REP_IDS[0]
            requests = []
            responses = iter([
                AIMessage(content="", tool_calls=[
                    {"name": "get_current_rep", "args": {}, "id": "rep"},
                    {"name": "send_prospect_email", "args": {
                        "prospect": {"name": "Prospect", "email": "prospect@example.com"},
                        "subject": "Hello", "body": "Following up",
                    }, "id": "email"},
                ]),
                AIMessage(content="Sent the email."),
            ])

            def generate(model, messages, **kwargs):
                requests.append((messages, kwargs))
                return ChatResult(generations=[ChatGeneration(message=next(responses))])

            run_id = uuid.uuid4()
            message_ids = [run_id] + [uuid.uuid4() for _ in range(30)]
            with patch.object(ChatOpenAI, "_generate", generate), \
                    patch.object(app.uuid, "uuid4", side_effect=message_ids), \
                    patch.object(app.gtm_agent, "invoke", wraps=app.gtm_agent.invoke) as invoke:
                result = app.run_agent(
                    "Email the prospect", user_id=rep["rep_id"],
                    environment="test", thread_id="thread-test",
                )

        messages, options = requests[0]
        self.assertEqual([m.content for m in messages if isinstance(m, SystemMessage)], [app.SYSTEM_PROMPT])
        schemas = {tool["function"]["name"]: tool["function"] for tool in options["tools"]}
        self.assertEqual(set(schemas), {
            "lookup_offering", "build_prospect_profile", "get_prospect",
            "send_prospect_email", "score_prospect", "update_prospect_info", "get_current_rep",
        })
        for name in ("get_current_rep", "send_prospect_email"):
            self.assertNotIn("runtime", schemas[name]["parameters"]["properties"])
        outputs = {m.name: json.loads(m.content) for m in result["messages"] if isinstance(m, ToolMessage)}
        self.assertEqual(outputs["get_current_rep"], {"rep": rep, "found": True})
        email = outputs["send_prospect_email"]
        self.assertEqual(email["status"], "sent")
        self.assertEqual(email["from"], rep["email"])
        self.assertEqual(email["from_name"], rep["name"])
        self.assertEqual(email["to"], "prospect@example.com")
        self.assertEqual(result["reply"], "Sent the email.")
        self.assertEqual(result["run_id"], str(run_id))
        self.assertEqual(invoke.call_args.kwargs["config"], {
            "run_name": "GTM Assistant", "run_id": run_id,
            "metadata": {"thread_id": "thread-test", "user_id": rep["rep_id"],
                         "environment": "test", "request_intent": "send_email"},
        })


if __name__ == "__main__":
    unittest.main()
