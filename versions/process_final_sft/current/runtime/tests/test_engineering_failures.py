import asyncio,copy,json,sys,tempfile,unittest
from pathlib import Path
from test_agent import ScriptModel,ScriptTools,CHECK,SEARCH,BROWSE,STOP,FOUND,rank
from trainable_agent.engine import Engine,initial_state
from trainable_agent.context import ContextBuilder,OfflineCodec
from trainable_agent.contracts import validate_action
from trainable_agent.storage import Store,DefiniteCallFailure
from trainable_agent.tools import WorkerBridge,normalize,attach_failure_details
from trainable_agent.recovery import prepare_resume

class Broken:
    def __init__(self,known=False):self.known=known
    async def generate(self,request):
        if self.known:raise DefiniteCallFailure("model_http_status:429")
        raise TimeoutError("sensitive detail must not enter feedback")

class EngineeringTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.store=Store(self.root/"records.sqlite")
        self.state=initial_state("run","question","p","f","t",process_only=True)
    async def asyncTearDown(self):self.store.db.close();self.tmp.cleanup()
    def engine(self,p,tools=None,f=None):
        return Engine(self.store,p,f,tools or ScriptTools([]),ContextBuilder(OfflineCodec(),100000),rank)
    async def test_unknown_model_timeout_saves_failed_handoff_and_blocks_resume(self):
        result=await self.engine(Broken()).run(self.state)
        self.assertEqual(result["run_status"],"failed")
        self.assertEqual(result["prefinal"]["stop_reason"],"engineering_failure")
        self.assertNotIn("sensitive detail",json.dumps(self.store.events("run")))
        with self.assertRaisesRegex(ValueError,"uncertain_calls"):prepare_resume(self.store,"run")
    async def test_definite_model_failure_explicit_resume_preserves_original(self):
        await self.engine(Broken(True)).run(self.state)
        keys=[k for k,s in self.store.db.execute("SELECT key,status FROM receipts") if s=="failed"]
        with self.assertRaisesRegex(ValueError,"explicit_retry"):prepare_resume(self.store,"run")
        prepare_resume(self.store,"run",keys)
        result=await self.engine(ScriptModel([CHECK,STOP])).run(self.state)
        self.assertEqual(result["run_status"],"process_complete")
        self.assertEqual(len(self.store.events("run","engineering_failure")),1)
        self.assertEqual(len(self.store.events("run","retry_authorized")),1)
    async def test_final_failure_keeps_process_reason_and_explicit_final_status(self):
        self.state["process_only"]=False
        result=await self.engine(ScriptModel([CHECK,STOP]),f=ScriptModel(["bad","bad"])).run(self.state)
        self.assertEqual(result["process_termination"],"policy_stop:sufficient")
        self.assertEqual(result["final_status"],"failed")
        self.assertEqual(result["run_status"],"failed")
    async def test_content_gate_charged_and_feedback_in_next_decision(self):
        obs=normalize({"failed":True,"failure_type":"environment_failure","error":"no_evidence_bearing_chunks","retryable":False},"paper")
        p=ScriptModel([CHECK,SEARCH,STOP])
        result=await self.engine(p,ScriptTools([obs])).run(self.state)
        self.assertEqual(result["budget"]["charged"],1)
        self.assertEqual(result["budget"]["exemptions"],0)
        self.assertIn("message",result["history"][0]["feedback"])
        self.assertIn("no_evidence_bearing_chunks",p.requests[-1]["messages"][-1]["content"])
    async def test_blocked_pdf_cannot_be_browsed_again(self):
        obs=normalize({"failed":True,"failure_type":"environment_failure","error_code":"archive_http_403","retryable":True},"paper","PMID:1")
        p=ScriptModel([CHECK,SEARCH,BROWSE,BROWSE,STOP])
        t=ScriptTools([FOUND,obs]);r=await self.engine(p,t).run(self.state)
        self.assertEqual(len(t.calls),2)
        self.assertIn("PMID:1",r["blocked_sources"])
        self.assertFalse(r["history"][-1]["feedback"]["retryable"])
    async def test_redirect_no_environment_exemption(self):
        obs=normalize({"failed":True,"error_code":"archive_redirect_limit","failure_type":"source_unreadable","retryable":True},"paper","PMID:1")
        r=await self.engine(ScriptModel([CHECK,SEARCH,BROWSE,STOP]),ScriptTools([FOUND,obs])).run(self.state)
        self.assertEqual(r["budget"]["charged"],2)
        self.assertIn("PMID:1",r["blocked_sources"])
    def test_retry_limit_cannot_be_bypassed_by_changed_reading_focus(self):
        self.state["requirements"]={"R1":"question"};self.state["candidates"]=FOUND["candidates"]
        self.state["history"]=[{"action":BROWSE,"feedback":{"status":"environment_failure"}}]*2
        with self.assertRaisesRegex(ValueError,"retry_limit"):
            validate_action({**BROWSE,"reading_focus":"changed"},self.state)
    async def test_failed_lease_not_stolen(self):
        await self.engine(Broken(True)).run(self.state)
        owner=self.store.acquire("run")
        try:
            with self.assertRaisesRegex(RuntimeError,"manual_lease"):prepare_resume(self.store,"run")
        finally:self.store.release("run",owner)
    async def test_worker_timeout_discards_old_stream(self):
        script=self.root/"worker.py"
        script.write_text('import json,sys,time\nprint(json.dumps({"status":"ready"}),flush=True)\nfor line in sys.stdin:\n p=json.loads(line)\n if p["operation"]=="slow":time.sleep(.5)\n print(json.dumps({"status":"ok","request_id":p["request_id"],"value":{"operation":p["operation"]}}),flush=True)\n')
        b=WorkerBridge([sys.executable,"-u",str(script)],self.root/"stderr.log",timeout=.15)
        try:
            with self.assertRaises(asyncio.TimeoutError):await b.request("slow",{})
            self.assertIsNone(b.process)
            b.timeout=2
            self.assertEqual((await b.request("next",{}))["operation"],"next")
        finally:await b.close()
    async def test_worker_mismatched_id_is_rejected(self):
        script=self.root/"bad_worker.py"
        script.write_text('import json,sys\nprint(json.dumps({"status":"ready"}),flush=True)\nfor line in sys.stdin:print(json.dumps({"status":"ok","request_id":"wrong","value":{}}),flush=True)\n')
        b=WorkerBridge([sys.executable,"-u",str(script)],self.root/"stderr.log")
        try:
            with self.assertRaisesRegex(RuntimeError,"request_id_mismatch"):await b.request("read",{})
            self.assertIsNone(b.process)
        finally:await b.close()

    async def test_qwen_http_failure_captured_without_credentials(self):
        from unittest.mock import patch
        import httpx
        from trainable_agent.model import EndpointModel
        class Client:
            def __init__(self,**kwargs):pass
            async def __aenter__(self):return self
            async def __aexit__(self,*args):pass
            async def post(self,*args,**kwargs):return httpx.Response(429,text='{"error":"rate limited"}')
        request={"stage":"decision","messages":[{"role":"user","content":"question"}],"max_new_tokens":400}
        model=EndpointModel("http://offline.invalid","Qwen3-8B",key_env="TEST_ONLY_KEY",capture_dir=self.root/"captures")
        with patch.dict("os.environ",{"TEST_ONLY_KEY":"fake-secret-not-a-real-key"}),patch("trainable_agent.model.httpx.AsyncClient",Client):
            with self.assertRaises(DefiniteCallFailure):await model.generate(request)
        files=list((self.root/"captures").rglob("*.*"))
        self.assertEqual({p.name for p in files},{"REQUEST.json","RESPONSE.txt","CAPTURE.json"})
        self.assertNotIn("fake-secret-not-a-real-key","".join(p.read_text() for p in files))
        self.assertEqual(json.loads(next(p for p in files if p.name=="CAPTURE.json").read_text())["http_status"],429)
    async def test_tool_crash_preserves_request_and_last_opened_evidence(self):
        from test_agent import OPENED,UPDATE
        class Tools(ScriptTools):
            async def execute(self,args):
                if not self.outputs:raise TimeoutError("private")
                return await super().execute(args)
        p=ScriptModel([CHECK,SEARCH,BROWSE,UPDATE,{**SEARCH,"query":"next question"}])
        r=await self.engine(p,Tools([FOUND,OPENED])).run(self.state)
        self.assertEqual(r["run_status"],"failed")
        self.assertEqual(r["prefinal"]["opened_evidence"],r["evidence"])
        self.assertEqual(len(r["evidence"]),1)
        self.assertEqual(len(self.store.events("run","tool_request")),3)

    def test_empty_document_and_parse_failure_are_not_environment_exemptions(self):
        for code in ("no_readable_biomedical_document","webpage_content_parse_failed"):
            obs=normalize({"failed":True,"failure_type":"environment_failure","error":code,"retryable":False},"paper","PMID:1")
            self.assertEqual(obs["status"],"source_unreadable")
            self.assertFalse(obs["retryable"])

    async def test_changed_focus_does_not_bypass_retry_after(self):
        from unittest.mock import patch,AsyncMock
        p=ScriptModel([CHECK,SEARCH,BROWSE,{**BROWSE,"reading_focus":"different focus"},STOP])
        obs=dict(status="environment_failure",retryable=True,retry_after_seconds=5.0)
        with patch("asyncio.sleep",new_callable=AsyncMock) as wait:
            r=await self.engine(p,ScriptTools([FOUND,obs,dict(status="empty")])).run(self.state)
            wait.assert_awaited_once_with(5.0)
        self.assertEqual(r["budget"]["actual"],3)
        self.assertEqual(len(self.store.events("run","retry_wait")),1)
    def test_nonretryable_environment_failure_is_not_repeated(self):
        self.state["requirements"]={"R1":"question"};self.state["candidates"]=FOUND["candidates"]
        self.state["history"]=[{"action":BROWSE,"feedback":{"status":"environment_failure","retryable":False}}]
        with self.assertRaisesRegex(ValueError,"nonretryable"):
            validate_action(BROWSE,self.state)
