import importlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

with patch('dotenv.load_dotenv'), patch.dict('os.environ', {
    'OPENAI_API_KEY': 'test-key', 'LANGSMITH_TRACING': 'false',
}):
    agent = importlib.import_module('gtm_agent.gtm_agent')


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class GTMToolsTest(unittest.TestCase):
    def setUp(self):
        self.prospect_id = 'LEAD-test'
        self.offering_id = 'OFFER-test'
        self.prospect = {
            'name': 'Test Prospect', 'email': 'prospect@example.com',
            'engagement_history': [], 'account_details': [],
            'tech_stack': ['Current'], 'annual_revenue': 1000000,
        }
        self.offering = {
            'description': 'Test offering', 'required_tech_stack': ['Current'],
            'min_annual_revenue': 100000,
        }
        self.rep = {'rep_id': 'rep-test', 'name': 'Test Rep', 'email': 'rep@example.com'}
        for name, value in (
            ('PROSPECTS', {self.prospect_id: self.prospect}),
            ('OFFERINGS', {self.offering_id: self.offering}),
            ('REP_IDS', [self.rep]), ('_PROFILES', {}),
        ):
            patcher = patch.object(agent.data_service, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.score = agent.ProspectScore(
            score=80, justification='Good fit',
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=80, tech_stack_match=80, segment_fit=80,
            ),
        )
        self.scorer = Mock()
        self.scorer.invoke.return_value = self.score
        patcher = patch.object(agent, '_scoring_llm', self.scorer)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_score_builds_and_loads_profile_with_current_tech_stack(self):
        args = {'prospect_id': self.prospect_id, 'offering_id': self.offering_id}
        self.assertEqual(agent.score_prospect.invoke(args), self.score.model_dump())
        saved = agent.build_prospect_profile.invoke({'prospect_id': self.prospect_id})
        self.assertTrue(saved['found'])
        self.assertEqual(saved['prospect_profile']['tech_stack'], ['Current'])
        self.prospect['tech_stack'] = ['Updated']
        self.assertEqual(agent.score_prospect.invoke(args), self.score.model_dump())
        content = self.scorer.invoke.call_args.args[0][1]['content']
        offering, profile = content.removeprefix('Offering:\n').split('\n\nProspect profile:\n')
        self.assertEqual(json.loads(offering), self.offering)
        self.assertEqual(json.loads(profile), {
            **saved['prospect_profile'], 'tech_stack': ['Updated'],
        })
        self.assertEqual(saved['prospect_profile']['tech_stack'], ['Current'])

    def test_invalid_offering_does_not_score(self):
        for offering in (None, {}, {'description': 'Incomplete'}):
            with self.subTest(offering=offering):
                agent.data_service.OFFERINGS[self.offering_id] = offering
                self.assertEqual(agent.score_prospect.invoke({
                    'prospect_id': self.prospect_id, 'offering_id': self.offering_id,
                }), {'score': None, 'error': 'Cannot score without a valid offering.'})
        self.scorer.invoke.assert_not_called()

    def test_missing_prospect_does_not_score(self):
        self.assertEqual(agent.score_prospect.invoke({
            'prospect_id': 'missing', 'offering_id': self.offering_id,
        }), {'score': None, 'error': 'Cannot score without a valid prospect.'})
        self.scorer.invoke.assert_not_called()

    def test_email_requires_stored_recipient_email(self):
        runtime = SimpleNamespace(config={'metadata': {'user_id': 'rep-test'}})
        for record in (None, {'name': 'No email'}, {'email': ''}):
            with self.subTest(record=record), patch.object(agent.uuid, 'uuid4') as message_id:
                agent.data_service.PROSPECTS[self.prospect_id] = record
                self.assertEqual(agent.send_prospect_email.func(
                    self.prospect_id, 'Subject', 'Body', runtime,
                ), {'status': 'failed', 'error': 'Prospect record has no email address.'})
                message_id.assert_not_called()

    def test_email_uses_stored_recipient_and_runtime_sender(self):
        runtime = SimpleNamespace(config={'metadata': {'user_id': 'rep-test'}})
        result = agent.send_prospect_email.func(self.prospect_id, 'Subject', 'Body', runtime)
        self.assertRegex(result.pop('message_id'), r'^msg-[0-9a-f]{12}$')
        self.assertEqual(result, {
            'status': 'sent', 'to': 'prospect@example.com', 'to_name': 'Test Prospect',
            'from': 'rep@example.com', 'from_name': 'Test Rep',
            'subject': 'Subject', 'body': 'Body',
        })
        self.assertEqual(agent.get_current_rep.func(runtime), {'rep': self.rep, 'found': True})

    def test_graph_completes_actions_with_ids_in_one_tool_turn(self):
        for name, args in (
            ('score_prospect', {'prospect_id': self.prospect_id, 'offering_id': self.offering_id}),
            ('send_prospect_email', {
                'prospect_id': self.prospect_id, 'subject': 'Subject', 'body': 'Body',
            }),
            ('get_current_rep', {}),
        ):
            with self.subTest(tool=name):
                model = ToolCallingModel(responses=[
                    AIMessage(content='', tool_calls=[{
                        'name': name, 'args': args, 'id': 'action',
                    }]), AIMessage(content='Done'),
                ])
                graph = agent.create_deep_agent(
                    model=model, system_prompt=agent.SYSTEM_PROMPT,
                    tools=[agent.score_prospect, agent.send_prospect_email, agent.get_current_rep],
                )
                result = graph.invoke(
                    {'messages': [{'role': 'user', 'content': 'Complete the action'}]},
                    config={'metadata': {'user_id': 'rep-test'}},
                )
                outputs = [m for m in result['messages'] if isinstance(m, ToolMessage)]
                self.assertEqual(len(outputs), 1)
                payload = json.loads(outputs[0].content)
                if name == 'score_prospect':
                    self.assertEqual(payload, self.score.model_dump())
                elif name == 'send_prospect_email':
                    self.assertEqual(payload['status'], 'sent')
                    self.assertEqual(payload['from'], self.rep['email'])
                else:
                    self.assertEqual(payload, {'rep': self.rep, 'found': True})
                self.assertEqual(result['messages'][-1].content, 'Done')


if __name__ == '__main__':
    unittest.main()
