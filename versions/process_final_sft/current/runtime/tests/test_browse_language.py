import copy,json,tempfile,unittest
from pathlib import Path
from trainable_agent.contracts import source_script_hint,validate_action
from trainable_agent.context import ContextBuilder,OfflineCodec
from trainable_agent.engine import Engine,initial_state
from trainable_agent.storage import Store
from test_agent import ScriptModel,ScriptTools,rank

def fixture(preview=None):
    s=initial_state('language','儿童两药治疗差异？','p','f','t')
    s['requirements']={'R1':'临床效果和不良反应'};s['assessments']={'R1':dict(support='missing',evidence_ids=[],gap='缺少比较',conflict='')}
    s['candidates']=[dict(source_id='PMID:1',channel='paper',title='Clinical trial',preview=preview or 'Adults received treatment versus placebo. Outcomes included clinical response and adverse effects. '*3,url='')]
    return s

def browse(focus):return dict(action='browse',requirement_ids=['R1'],source_id='PMID:1',reading_focus=focus)

class LanguageTests(unittest.TestCase):
    def test_latin_source_chinese_query_rejected_before_execution(self):
        s=fixture();saved=copy.deepcopy(s)
        with self.assertRaisesRegex(ValueError,'browse_query_script_mismatch'):validate_action(browse('查找治疗组与对照组的临床效果和不良反应结果'),s)
        self.assertEqual(s,saved)
    def test_source_language_query_does_not_translate_question_or_state(self):
        s=fixture();saved=copy.deepcopy(s);self.assertEqual(validate_action(browse('treatment placebo clinical response adverse events'),s)['source_id'],'PMID:1')
        self.assertEqual(s,saved)
    def test_chinese_source_chinese_query_allowed_english_rejected(self):
        s=fixture('本临床研究比较治疗组与对照组的不良反应和临床疗效。'*12)
        self.assertEqual(source_script_hint(s['candidates'][0]),'cjk')
        validate_action(browse('治疗组 对照组 临床疗效 不良反应'),s)
        with self.assertRaises(ValueError):validate_action(browse('treatment versus control clinical efficacy and adverse effects'),s)
    def test_unknown_and_mixed_are_not_hard_translated(self):
        for text in ['short preview','English中文'*30]:
            s=fixture(text);validate_action(browse('查找临床结果'),s)
    def test_hint_not_exact_language_detection(self):
        s=fixture('Les patients ont reçu un traitement ou un placebo dans une étude clinique. '*6)
        self.assertEqual(source_script_hint(s['candidates'][0]),'latin')
        validate_action(browse('traitement placebo réponse clinique effets indésirables'),s)
    def test_hint_visible_order_and_ids_stable(self):
        s=fixture();saved=copy.deepcopy(s);v=json.loads(ContextBuilder(OfflineCodec(),100000).build('decision',s)['messages'][1]['content'])
        self.assertEqual(v['candidates'][0]['source_script_hint'],'latin');self.assertEqual(v['candidates'][0]['source_id'],'A');self.assertEqual(s,saved)

class LanguageRepair(unittest.IsolatedAsyncioTestCase):
    async def test_repair_keeps_raw_rejection_and_consumes_no_tool_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/'r.sqlite')
            try:
                s=fixture();model=ScriptModel([browse('查找治疗组与对照组的临床结果'),browse('treatment placebo clinical response adverse events')]);tools=ScriptTools([])
                engine=Engine(store,model,None,tools,ContextBuilder(OfflineCodec(),100000),rank)
                result=(await engine.decision({'session':s}))['session']
                rows=store.events('language','generation')
                self.assertEqual([r['data']['accepted'] for r in rows],[False,True])
                self.assertIn('browse_query_script_mismatch',rows[0]['data']['error'])
                self.assertEqual(result['next_node'],'execute');self.assertEqual(result['budget']['actual'],0);self.assertEqual(tools.calls,[])
            finally:store.db.close()
