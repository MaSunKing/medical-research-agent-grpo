import copy
from pathlib import Path
import tempfile
import unittest
from eval_common import read,save,sha,manifest
import evaluate as e

class BindingTests(unittest.TestCase):
    def setUp(self):
        tmp=Path.cwd()/'tmp';tmp.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=tmp);self.root=Path(self.temp.name)
        self.package=self.root/'package';self.train=self.root/'train';self.original=self.root/'original'
        self.original.mkdir()
        for n in ('adapter_model.safetensors','adapter_config.json'):(self.original/n).write_bytes(b'original fixture')
        files={p.name:sha(p) for p in self.original.iterdir()}
        save(self.package/'data/EXPECTED_ORIGINAL972.json',dict(adapter_files=files))
        self.identity=dict(dataset_sha256='dataset',config={'lr':1e-6},code={'fixture.py':'hash'},
            initial_adapter=dict(adapter_model_safetensors='unused',**files),
            environment=dict(deterministic_algorithms=True,gpu='synthetic'))
        save(self.package/'data/EXPECTED_TRAINING.json',dict(dataset_sha256='dataset',config={'lr':1e-6},code={'fixture.py':'hash'}))
        save(self.train/'TRAINING_COMPLETE.json',dict(completed_batches=25,checkpoint='checkpoint-0025',
             mid_checkpoint='checkpoint-0013',identity=self.identity))
        for step in (13,25):
            cp=self.train/f'checkpoint-{step:04d}';cp.mkdir()
            for n in files:(cp/n).write_bytes(b'updated fixture')
            (cp/'optimizer.pt').write_bytes(b'synthetic only')
            save(cp/'STATE.json',dict(step=step,identity=self.identity))
            save(cp/'COMPLETE.json',manifest(cp))
        self.config=dict(original_adapter=str(self.original),training_output=str(self.train))
        expected={a:dict(files=files,adapter_disabled=(a=='base')) for a in ('base','sft')}
        for a,step in [('grpo13',13),('grpo25',25)]:
            cp=self.train/f'checkpoint-{step:04d}'
            expected[a]=dict(files={n:sha(cp/n) for n in files},adapter_disabled=False)
        save(self.package/'data/EXPECTED_EVAL_MODELS.json',expected)
    def tearDown(self):self.temp.cleanup()

    def test_binds_exact_middle_and_final(self):
        models,env=e.bind_models(self.package,self.config)
        self.assertTrue(models['base']['adapter_disabled'])
        self.assertFalse(models['sft']['adapter_disabled'])
        self.assertEqual(models['grpo13']['training_step'],13)
        self.assertEqual(models['grpo25']['training_step'],25)

    def test_incomplete_training_rejected(self):
        p=self.train/'TRAINING_COMPLETE.json';d=read(p);d['completed_batches']=24;save(p,d)
        with self.assertRaisesRegex(ValueError,'complete_first'):e.bind_models(self.package,self.config)

    def test_other_config_cannot_pass_same_dataset(self):
        p=self.package/'data/EXPECTED_TRAINING.json';d=read(p);d['config']['lr']=2e-6;save(p,d)
        with self.assertRaisesRegex(ValueError,'wrong_training_configuration'):e.bind_models(self.package,self.config)

    def test_changed_adapter_rejected(self):
        (self.train/'checkpoint-0013/adapter_model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'file_identity_changed'):e.bind_models(self.package,self.config)

    def test_resealed_other_weights_still_rejected(self):
        cp=self.train/'checkpoint-0013'
        (cp/'adapter_model.safetensors').write_bytes(b'different run weights')
        save(cp/'COMPLETE.json',manifest(cp,excludes=('COMPLETE.json',)))
        with self.assertRaisesRegex(ValueError,'prior_verified_checkpoint'):e.bind_models(self.package,self.config)

if __name__=='__main__':unittest.main()
