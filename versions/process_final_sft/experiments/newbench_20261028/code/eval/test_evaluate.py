import asyncio
import contextlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch
import eval_common as ec
import evaluate as ev

class Tests(unittest.TestCase):
    def setUp(self):
        tmp=Path.cwd()/'tmp';tmp.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=tmp);self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_seal_rejects_late_write(self):
        ec.save(self.root/'RESULT.json',dict(run_identity='id'))
        ec.seal(self.root,'id');self.assertTrue(ec.verify_done(self.root,'id'))
        (self.root/'stderr.log').write_text('late')
        with self.assertRaisesRegex(ValueError,'unsealed_result'):ec.verify_done(self.root,'id')

    def test_close_before_seal(self):
        ec.save(self.root/'RESULT.json',dict(run_identity='id'))
        (self.root/'trajectory.sqlite-wal').write_bytes(b'pending')
        with self.assertRaisesRegex(ValueError,'SQLite_not_closed'):ec.seal(self.root,'id')

    def test_nested_manifests_and_DONE_are_sealed_too(self):
        ec.save(self.root/'RESULT.json',dict(run_identity='id'))
        ec.save(self.root/'worker/DONE.json',dict(worker='finished'))
        ec.save(self.root/'data/FILES.sha256.json',dict(fixture='hash'))
        ec.seal(self.root,'id')
        files=ec.read(self.root/'DONE.json')['files']
        self.assertIn('worker/DONE.json',files)
        self.assertIn('data/FILES.sha256.json',files)
        ec.save(self.root/'worker/DONE.json',dict(worker='changed'))
        with self.assertRaisesRegex(ValueError,'file_identity_changed'):ec.verify_done(self.root,'id')

    def test_safe_paths(self):
        for p in ('../a','/tmp/a','a\\b','C:a'):
            with self.assertRaises(ValueError):ec.safe(self.root,p)

    def test_final_failure_does_not_zero_process(self):
        self.assertEqual(ev.classify(dict(process_termination='policy_stop:diminishing_returns',
            final_status='failed',failure=dict(stage='final'))),('final_failure','complete'))
        self.assertEqual(ev.classify(dict(process_termination='model_contract_failure',final_status='not_started')),
                         ('process_failure','failed'))

    def test_actual_request_not_prefinal_archive(self):
        req=dict(role='final',input_sha256='actual',messages=[dict(role='user',content='delivered')])
        events=[dict(kind='model_request',data=dict(request=req))]
        r,d=ev.result_from(dict(final='<answer>x</answer>',final_status='complete',prefinal={'bigger':'archive'}),
                          events,dict(index=1,question_id='q',benchmark='healthbench',source_id='source'),'base','id',1.)
        self.assertEqual(r['final_input_sha256'],'actual');self.assertEqual(d,req)

    def test_base_disables_adapter_and_head_restores_on_failure(self):
        # Only a synthetic linear head, no pretrained model or inference.
        here=Path(__file__).resolve().parent
        native_dir=here.parent/'runtime/training/process_grpo'
        if not native_dir.is_dir():native_dir=here.parents[1]/'training/process_grpo'
        sys.path.insert(0,str(native_dir))
        import decoupled
        class Linear:
            weight=SimpleNamespace(requires_grad=False)
            bias=None
            def forward(self,x):return x
        class Model:
            def __init__(self):self.lm_head=Linear();self.disabled=False
            def get_base_model(self):return self
            def eval(self):return self
            def requires_grad_(self,v):return self
            @contextlib.contextmanager
            def disable_adapter(self):
                self.disabled=True
                try:yield
                finally:self.disabled=False
        model=Model();original=model.lm_head.forward
        def fail(*args,**kwargs):
            self.assertTrue(model.disabled);self.assertNotEqual(model.lm_head.forward,original)
            raise RuntimeError('synthetic failure')
        native=SimpleNamespace(sample=fail)
        functional=ModuleType('torch.nn.functional')
        nn=ModuleType('torch.nn');nn.Linear=Linear;nn.functional=functional
        torch=ModuleType('torch');torch.nn=nn
        torch.autocast=lambda **k:contextlib.nullcontext()
        tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**k:[1])
        req=dict(messages=[],prompt_token_ids=[1],max_new_tokens=2,stage='checklist',input_sha256='x')
        endpoint=ev.EvalProcessPolicy(model,tokenizer,'base',1,self.root,'id',lambda stage:None)
        with patch.dict(sys.modules,{'native':native,'decoupled':decoupled,
                                    'torch':torch,'torch.nn':nn,'torch.nn.functional':functional}):
            # generate has no suspension point; drive directly without a socket-
            # backed Windows event loop in the restricted offline test runtime.
            coro=endpoint.generate(req)
            try:
                with self.assertRaisesRegex(RuntimeError,'synthetic'):coro.send(None)
            finally:coro.close()
        self.assertFalse(model.disabled);self.assertEqual(model.lm_head.forward,original)
        self.assertTrue((self.root/'0001_checklist/FAILURE.json').exists())

    def test_config_forbids_inline_secrets(self):
        c=dict(base_model='base',original_adapter='adapter',training_output='train',seed=42,
               retrieval=dict(worker=dict(kind='native',api_key='should-not-be-inline')))
        with self.assertRaisesRegex(ValueError,'inline_credentials'):ev.validate_config(c)

if __name__=='__main__':unittest.main()
