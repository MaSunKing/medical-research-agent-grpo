import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from trainable_agent.context import ContextBuilder, ContextOverflow, OfflineCodec
from trainable_agent.context_processing.aliases import register, project, resolve_action
from trainable_agent.context_processing.fallback import ScopedFallback
from trainable_agent.context_processing.browser import preprocess_document
from trainable_agent.engine import initial_state, Engine
from trainable_agent.storage import Store
from trainable_agent.contracts import validate_action


class StabilityTests(unittest.TestCase):
    def test_alias_survives_sort_eviction_checkpoint_and_reappearance(self):
        s=initial_state("r","Q","p","f","t")
        s["candidates"]=[{"source_id":"PMID:1"},{"source_id":"WEB:2"}]
        self.assertEqual(register(s),{"PMID:1":"A","WEB:2":"B"})
        s=json.loads(json.dumps(s))
        s["candidates"]=[{"source_id":"WEB:2"},{"source_id":"PMID:3"}]
        register(s)
        s["candidates"]=[{"source_id":"PMID:1"}]
        self.assertEqual(register(s),{"PMID:1":"A","WEB:2":"B","PMID:3":"C"})
        s["requirements"]={"R1":"Q"}
        action=dict(action="browse",source_id="A",requirement_ids=["R1"],reading_focus="Results")
        self.assertEqual(validate_action(resolve_action(action,s),s)["source_id"],"PMID:1")
        with self.assertRaisesRegex(ValueError,"visible_candidates"):
            validate_action(resolve_action({**action,"source_id":"B"},s),s)
        with self.assertRaisesRegex(ValueError,"unknown_source_alias"):
            resolve_action({**action,"source_id":"Z"},s)

    def test_all_history_preserved_and_never_mutated(self):
        s=initial_state("r","Q","p","f","t")
        s["history"]=[dict(turn=i,action=dict(action="search",query=f"query {i}"),feedback=dict(status="empty")) for i in range(16)]
        before=copy.deepcopy(s)
        req=ContextBuilder(OfflineCodec(),50000,evidence_cards=False).build("state",s)
        self.assertEqual(json.loads(req["messages"][1]["content"])["history"],s["history"])
        self.assertEqual(req["context_audit"]["history_events_omitted"],0)
        with self.assertRaises(ContextOverflow):
            ContextBuilder(OfflineCodec(),2000,evidence_cards=False).build("state",s)
        self.assertEqual(s,before)

    def test_browser_adapter_uses_shared_rules_without_behavior_change(self):
        path=Path(__file__).resolve().parents[1]/"retrieval_snapshot_v2/backend/agent/dr_agent/mcp_backend/apis/browse_preprocess.py"
        spec=importlib.util.spec_from_file_location("browser_adapter_fixture",path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertIs(module.preprocess_document,preprocess_document)
        doc={"sections":[{"heading":"Results","text":"Patients did not improve; 12% reported adverse events."},
                         {"heading":"Site navigation","text":"Home | About | Contact"}]}
        before=copy.deepcopy(doc)
        output=module.preprocess_document(doc)
        self.assertEqual(doc,before)
        self.assertTrue(output["sections"][0]["retrieval_spans"])
        self.assertFalse(output["sections"][1]["retrieval_spans"])

    def test_reference_witness_compression_restores_dependents(self):
        from trainable_agent.compression import card
        s=initial_state("r","Q","p","f","t")
        text="No reduction in mortality was established in adults with prior disease. "*3
        s["evidence"]={"E1":dict(source_id="PMID:1",chunk_id="c1",text=text),
                       "E2":dict(source_id="PMID:1",chunk_id="c2",text=text+"Additional context.")}
        override=card(s["evidence"]["E2"],["S0"])
        req=ContextBuilder(OfflineCodec(),50000).build("state",s,overrides={"E2":override},allow_cards=False)
        e=json.loads(req["messages"][1]["content"])["opened_evidence"]
        self.assertNotIn("context_reference",e["E1"])
        self.assertEqual(e["E1"]["text"],text)


class Provider:
    identity="offline-scoped-fixture"
    def __init__(self,output):self.output=output;self.requests=[]
    async def generate(self,request):
        self.requests.append(request)
        return {"text":json.dumps(self.output)}


class ScopedTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=Store(Path(self.tmp.name)/"records.sqlite")
        self.addCleanup(self.tmp.cleanup);self.addCleanup(self.store.db.close)
        self.chunk=dict(source_id="PMID:1",chunk_id="c1",text="Advertisement\nPatients did not improve; adverse events occurred in 12%.\n",metadata={"year":2024})
        self.focus=dict(question="Benefits and harms?",checklist={"R1":"Benefits and harms"})
    def fallback(self,provider):return ScopedFallback(provider,OfflineCodec(),self.store,"r",100000)

    async def test_only_designated_text_and_readonly_focus_sent(self):
        p=Provider({"remove_line_ids":["L0"]})
        before=copy.deepcopy(self.chunk)
        result=await self.fallback(p).transform(self.chunk,self.focus,5000,"clean")
        self.assertEqual(result["text"],self.chunk["text"].splitlines(keepends=True)[1])
        self.assertEqual(self.chunk,before)
        payload=json.loads(p.requests[0]["messages"][1]["content"])
        self.assertEqual(set(payload),{"text_parts","target_text_tokens","mode","question","checklist"})
        self.assertNotIn("source_id",payload)
        self.assertEqual(result["metadata"],self.chunk["metadata"])

    async def test_no_model_call_when_no_partial_line_or_complete_passage_can_fit(self):
        p=Provider({"remove_line_ids":[]})
        chunk={**self.chunk,"text":"Patients did not improve; adverse events occurred in 12%."}
        self.assertIsNone(await self.fallback(p).transform(chunk,self.focus,1000,"clean"))
        self.assertIsNone(await self.fallback(p).transform(chunk,self.focus,1,"compress"))
        self.assertEqual(p.requests,[])

    async def test_deleting_clinical_line_or_unknown_id_is_rejected(self):
        for ids in (["L1"],["L999"],["L0","L0"]):
            self.assertIsNone(await self.fallback(Provider({"remove_line_ids":ids})).transform(self.chunk,self.focus,5000,"clean"))

    async def test_extracts_original_and_enforces_target_not_model_claim(self):
        chunk={**self.chunk,"text":"No mortality benefit was established. Background context. Additional observation."}
        result=await self.fallback(Provider({"passage_ids":["S0"]})).transform(chunk,self.focus,1000,"compress")
        self.assertTrue(result["text"].startswith("No mortality benefit"))
        self.assertFalse(result["evidence_card"]["semantic_verified"])
        self.assertIsNone(await self.fallback(Provider({"passage_ids":["S0"]})).transform(chunk,self.focus,1,"compress"))
        self.assertEqual(chunk["source_id"],"PMID:1")

    async def test_bounded_retry_from_original_and_fixed_floor_no_calls(self):
        s=initial_state("r","Q","p","f","t")
        s["evidence"]={"E1":{**self.chunk,"text":"No benefit. "*1000}}
        class NoReduction:
            def __init__(self):self.calls=[]
            async def transform(inner,chunk,focus,target,mode):
                inner.calls.append((copy.deepcopy(chunk),target,mode));return None
        f=NoReduction();b=ContextBuilder(OfflineCodec(),5000,evidence_cards=False,fallback_rounds=3)
        with self.assertRaises(ContextOverflow):await b.prepare("state",s,fallback=f)
        # Once different rounds hit the same indivisible target, do not pay
        # for identical no-reduction proposals repeatedly.
        self.assertGreaterEqual(len(f.calls),1)
        self.assertLessEqual(len(f.calls),3)
        self.assertTrue(all(c[2]=='compress' for c in f.calls))
        self.assertEqual(len({c[1] for c in f.calls}),len(f.calls))
        self.assertTrue(all(c[0]==s["evidence"]["E1"] for c in f.calls))
        self.assertEqual([c[1] for c in f.calls],sorted((c[1] for c in f.calls),reverse=True))
        f.calls=[]
        with self.assertRaises(ContextOverflow):await ContextBuilder(OfflineCodec(),100,evidence_cards=False).prepare("state",s,fallback=f)
        self.assertEqual(f.calls,[])

    async def test_latest_evidence_can_be_projected_without_raw_loss(self):
        from trainable_agent.compression import card
        s=initial_state("r","Q","p","f","t")
        s["evidence"]={"E1":{**self.chunk,"text":"No benefit was established. "+"Background observation. "*300}}
        s["new_evidence_ids"]=["E1"]
        before=copy.deepcopy(s)
        class Extract:
            async def transform(inner,chunk,focus,target,mode):
                return card(chunk,["S0"]) if mode=="compress" else None
        b=ContextBuilder(OfflineCodec(),5000,evidence_cards=False,safety_margin=128)
        req=await b.prepare("state",s,fallback=Extract())
        self.assertEqual(s,before)
        self.assertLessEqual(len(req["prompt_token_ids"])+req["max_new_tokens"]+128,5000)
        self.assertEqual(req["raw_context_payload"]["opened_evidence"],s["evidence"])
        self.assertEqual(json.loads(req["messages"][1]["content"])["new_evidence_ids"],["E1"])

    async def test_prefinal_contains_all_history_and_raw_evidence(self):
        s=initial_state("r","Q","p","f","t",process_only=True)
        s["history"]=[{"turn":i} for i in range(16)]
        s["evidence"]={"E1":self.chunk};s["termination"]="runtime_budget_exhausted"
        e=Engine(self.store,None,None,None,ContextBuilder(OfflineCodec()),None)
        out=(await e.prefinal({"session":s}))["session"]
        self.assertEqual(out["prefinal"]["compact_history"],s["history"])
        self.assertEqual(out["prefinal"]["opened_evidence"],s["evidence"])
        self.assertEqual(out["prefinal"]["source_aliases"],{"PMID:1":"A"})


class DeficitAllocationTests(unittest.IsolatedAsyncioTestCase):
    async def test_large_chunk_budget_not_divided_by_number_of_small_chunks(self):
        s=initial_state("deficit","Q","p","f","t")
        s["evidence"]={"E1":dict(source_id="PMID:1",chunk_id="c1",text="No benefit was established. "+"Background observation. "*100)}
        for i in range(2,9):
            s["evidence"][f"E{i}"]=dict(source_id=f"PMID:{i}",chunk_id=f"c{i}",text="Short original observation.")
        b=ContextBuilder(OfflineCodec(),50000,evidence_cards=False,safety_margin=128)
        raw=b.build("state",s,allow_cards=False)
        b.limit=len(raw["prompt_token_ids"])+400+128-100
        from trainable_agent.compression import card
        class Scoped:
            def __init__(self):self.targets=[]
            async def transform(inner,chunk,focus,target,mode):
                if mode=="compress":
                    inner.targets.append(target)
                    return card(chunk,["S0"])
                return None
        f=Scoped()
        result=await b.prepare("state",s,fallback=f)
        self.assertGreater(f.targets[0],500)
        self.assertLessEqual(len(result["prompt_token_ids"])+400+128,b.limit)
        self.assertEqual(result["context_audit"]["history_events_omitted"],0)


class DistributedDeficitTests(unittest.IsolatedAsyncioTestCase):
    async def test_deficit_larger_than_one_chunk_allows_multiple_reductions(self):
        from trainable_agent.compression import card
        s=initial_state("distributed","Q","p","f","t")
        s["evidence"]={f"E{i}":dict(source_id=f"PMID:{i}",chunk_id=f"c{i}",text="No benefit was established. "+"Background observation. "*15) for i in range(1,9)}
        b=ContextBuilder(OfflineCodec(),50000,evidence_cards=False,safety_margin=128)
        raw=b.build("state",s,allow_cards=False)
        b.limit=len(raw["prompt_token_ids"])+400+128-700
        class F:
            def __init__(inner):inner.targets=[]
            async def transform(inner,chunk,focus,target,mode):
                if mode=="clean":return None
                inner.targets.append(target)
                if target < 60:return None
                return card(chunk,["S0"])
        f=F();req=await b.prepare("state",s,fallback=f)
        self.assertGreater(f.targets[0],60)
        self.assertGreater(len(req["context_audit"]["compressed_evidence_ids"]),1)
        self.assertLessEqual(len(req["prompt_token_ids"])+400+128,b.limit)
