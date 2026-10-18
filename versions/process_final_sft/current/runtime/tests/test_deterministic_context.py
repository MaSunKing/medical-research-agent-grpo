import copy
import json
from pathlib import Path
import tempfile
import unittest
from trainable_agent.cleaning import clean_chunk
from trainable_agent.context import ContextBuilder,ContextOverflow,OfflineCodec
from trainable_agent.deterministic_context import project_evidence,validate_projection
from trainable_agent.engine import Engine,initial_state
from trainable_agent.storage import Store

class DeterministicProjectionTests(unittest.TestCase):
    def test_provenance_archived_but_dates_negation_and_metadata_retained(self):
        original={"E1":{"source_id":"PMID:1","chunk_id":"c1","text":"No benefit was established.",
                        "metadata":{"year":2024,"population":"pregnant adults"},"cleaning":{"kept_spans":[[0,26]],"original_text_sha256":"hash"}}}
        saved=copy.deepcopy(original)
        visible,audit=project_evidence(original)
        self.assertNotIn("cleaning",visible["E1"])
        self.assertEqual(visible["E1"]["metadata"],original["E1"]["metadata"])
        self.assertEqual(visible["E1"]["text"],original["E1"]["text"])
        self.assertEqual(original,saved)
        self.assertEqual(audit["provenance_omitted_ids"],["E1"])

    def test_same_source_containment_is_reconstructable_without_id_loss(self):
        text="No reduction in mortality was established in pregnant adults; comparison with placebo remained uncertain. "
        original={"E1":{"source_id":"PMID:1","chunk_id":"c1","text":text},
                  "E2":{"source_id":"PMID:1","chunk_id":"c2","text":"Earlier context. "+text+" Later qualification."}}
        visible,audit=project_evidence(original)
        self.assertEqual(set(visible),set(original))
        ref=visible["E1"]["context_reference"]
        self.assertEqual(visible["E2"]["text"][ref["start"]:ref["end"]],text)
        self.assertEqual(visible["E2"]["text"],original["E2"]["text"])
        validate_projection(visible,original)
        visible["E1"]["context_reference"]["end"]-=1
        with self.assertRaisesRegex(ValueError,"text_mismatch"):validate_projection(visible,original)

    def test_different_sources_and_partial_overlap_preserved(self):
        text="No benefit in adults. "*10
        original={"E1":{"source_id":"PMID:1","text":text},
                  "E2":{"source_id":"PMID:2","text":text},
                  "E3":{"source_id":"PMID:1","text":text+"Conflicting finding."},
                  "E4":{"source_id":"PMID:1","text":"Different context. "+text[:80]}}
        visible,_=project_evidence(original)
        self.assertEqual(visible["E2"]["text"],text)
        self.assertEqual(visible["E4"]["text"],original["E4"]["text"])

    def test_toc_prefix_and_editorial_only_removed_with_exact_span_audit(self):
        body="Treatment did not reduce mortality; severe adverse events were more frequent."
        text="Background.... 1 Methods.... 2 Results.... 3 "+body
        out=clean_chunk({"text":text})
        self.assertEqual(out["text"],body)
        self.assertEqual("".join(text[a:b] for a,b in out["cleaning"]["kept_spans"]),body)
        editorial="Background.... 1 Methods.... 2 Results.... 3 Suggested citation: [Title]. Editor in Chief A; Managing Editor B; Editorial Board C."
        self.assertEqual(clean_chunk({"text":editorial})["text"],"")
        mixed=editorial+" Participants experienced adverse events."
        self.assertIn("adverse events",clean_chunk({"text":mixed})["text"])

    def test_unknown_prose_and_clinical_heading_not_removed(self):
        for text in ("Adverse effects","Editor recommendations: monitor adverse effects.",
                     "Methods and Results reported no benefit in adults."):
            self.assertEqual(clean_chunk({"text":text})["text"],text)

    def test_context_has_raw_archive_and_no_model_card_needed(self):
        state=initial_state("test","question","p","f","t")
        state["evidence"]={"E1":{"source_id":"PMID:1","chunk_id":"c1","text":"No benefit.",
                               "cleaning":{"original_text_sha256":"x"*10000}}}
        req=ContextBuilder(OfflineCodec(),5000,evidence_cards=False).build("state",state)
        self.assertIn("cleaning",req["raw_context_payload"]["opened_evidence"]["E1"])
        self.assertNotIn("cleaning",json.loads(req["messages"][1]["content"])["opened_evidence"]["E1"])
        self.assertFalse(req["context_audit"]["evidence_compressed"])

class OverflowStopTests(unittest.IsolatedAsyncioTestCase):
    async def test_overflow_stops_at_prefinal_without_model_call_or_evidence_loss(self):
        class NoModel:
            async def generate(self,request):raise AssertionError("No model call permitted")
        with tempfile.TemporaryDirectory() as folder:
            store=Store(Path(folder)/"trajectory.sqlite")
            self.addCleanup(store.db.close)
            s=initial_state("run","question","p","f","t",process_only=False)
            s["next_node"]="state"
            s["requirements"]={"R1":"Safety"}
            s["assessments"]={"R1":{"support":"partial","evidence_ids":["E1"],"gap":"Comparator missing","conflict":""}}
            s["evidence"]={"E1":{"source_id":"PMID:1","chunk_id":"c1","text":"No benefit. "*1000}}
            result=await Engine(store,NoModel(),NoModel(),None,ContextBuilder(OfflineCodec(),2000,evidence_cards=False),None).run(s)
            self.assertEqual(result["termination"],"context_budget_exhausted")
            self.assertIsNone(result["final"])
            self.assertEqual(result["prefinal"]["opened_evidence"],s["evidence"])
            self.assertIn("R1",result["prefinal"]["unresolved"])
            self.assertEqual(len(store.events("run","context_failure")),1)
            store.db.close()
