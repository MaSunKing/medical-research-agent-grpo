import copy,importlib,sys,types,unittest
from pathlib import Path
from trainable_agent.context_processing.browser import repeated_window,opened_window_scope,current_opened_windows
from trainable_agent.tools import create_server,MCPTools
APIS=Path(__file__).resolve().parents[1]/'retrieval_snapshot_v2/backend/agent/dr_agent/mcp_backend/apis'
PKG='offline_novelty_apis'
pkg=types.ModuleType(PKG);pkg.__path__=[str(APIS)];sys.modules[PKG]=pkg
handoff=importlib.import_module(PKG+'.evidence_handoff_v3')

class Chars:
    def encode(self,text,add_special_tokens=False):return list(text)

def fixture(texts,sid='WEB:one'):
    d={'sections':[{'heading':'Results','text':t} for t in texts]}
    rows=[dict(chunk_id=f'{sid}#s{i}-c0',heading='Results',section_index=i,start_char=0,end_char=len(t),text=t,structure_kind='prose') for i,t in enumerate(texts)]
    return d,rows

class NoveltyTests(unittest.TestCase):
    def pack(self,texts,opened=(),budget=1000,top_k=3):
        doc,rows=fixture(texts);saved=copy.deepcopy((doc,rows,opened))
        out,audit=handoff.pack_browse(rows,doc,tokenizer=Chars(),max_tokens=budget,max_chars=budget,top_k=top_k,opened_windows=opened)
        self.assertEqual((doc,rows,opened),saved)
        self.assertLessEqual(sum(len(c['text']) for c in out),budget)
        for c in out:self.assertEqual(doc['sections'][c['section_index']]['text'][c['start_char']:c['end_char']],c['text'])
        return out,audit
    def test_cross_round_refills_lower_rank_and_keeps_id_ledger(self):
        texts=['Patients received treatment.','Patients had monitoring.','Symptoms did not improve.','Adverse events were recorded.']
        opened=[dict(source_id='WEB:one',evidence_id=f'E{i+1}',text=t) for i,t in enumerate(texts[:3])]
        out,audit=self.pack(texts,opened)
        self.assertEqual([c['section_index'] for c in out],[3])
        self.assertEqual(sum(a['reason']=='already_opened_window' for a in audit),3)
    def test_other_source_same_text_remains(self):
        out,_=self.pack(['Patients received treatment.'],[dict(source_id='PMID:1',text='Patients received treatment.')])
        self.assertEqual(len(out),1)
    def test_partial_overlap_with_new_content_remains(self):
        out,_=self.pack(['Patients received treatment. Symptoms did not improve.'],[dict(source_id='WEB:one',text='Patients received treatment.')])
        self.assertIn('did not improve',out[0]['text'])
    def test_final_clipped_duplicate_releases_slot(self):
        common='Patients received treatment.'
        texts=[common+' '+('Followup '+word+' ')*9+'.' for word in ['alpha','beta','gamma']]+['Adverse events were recorded.']
        out,audit=self.pack(texts,budget=120)
        self.assertEqual(len({c['text'] for c in out}),len(out))
        self.assertTrue(any(c['section_index']==3 for c in out))
        self.assertTrue(any(a['reason']=='repeated_final_window' for a in audit))
    def test_clipped_window_matches_old_prefix(self):
        common='Patients received treatment.'
        texts=[common+' '+('Followup response ')*12+'.','Symptoms did not improve.','Adverse events were recorded.']
        out,audit=self.pack(texts,[dict(source_id='WEB:one',evidence_id='E1',text=common)],budget=100)
        self.assertTrue(any(a['reason']=='repeated_final_window' for a in audit))
        self.assertFalse(any(c['text']==common for c in out))
    def test_exhausted_pool_is_empty_with_reasons(self):
        text='Patients received treatment.'
        out,audit=self.pack([text],[dict(source_id='WEB:one',text=text)])
        self.assertFalse(out);self.assertTrue(audit)
    def test_scope_restores_on_exception(self):
        with self.assertRaises(RuntimeError):
            with opened_window_scope([{'source_id':'WEB:one','text':'original'}]):
                self.assertEqual(len(current_opened_windows()),1)
                raise RuntimeError()
        self.assertEqual(current_opened_windows(),())
    def test_contained_long_window_skipped_but_short_substring_kept(self):
        long='Patients received treatment with monitoring and did not improve despite continued followup.'
        opened=[dict(source_id='WEB:one',text=long+' Adverse events occurred.')]
        self.assertTrue(repeated_window(long,'WEB:one',opened))
        self.assertIsNone(repeated_window('did not improve','WEB:one',opened))

class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_program_ledger_crosses_mcp_boundary(self):
        from fastmcp import Client
        seen=[]
        class Bridge:
            async def request(self,operation,args):seen.append((operation,args));return {'data':[]}
        ledger=[dict(source_id='WEB:one',evidence_id='E1',text='Patients received treatment.')]
        async with Client(create_server(Bridge())) as client:
            result=await MCPTools(client).execute(dict(question='original',channel='web',opened_windows=ledger,
                action=dict(action='browse',source_id='WEB:one',reading_focus='safety',requirement_ids=['R1'])))
        self.assertEqual(seen[0][1]['opened_windows'],ledger)
        self.assertEqual(seen[0][1]['query'],'safety');self.assertEqual(result['status'],'empty')
