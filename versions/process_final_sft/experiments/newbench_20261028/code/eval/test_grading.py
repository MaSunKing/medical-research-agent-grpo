"""Synthetic-only HealthBench bridge coverage. No actual HTTP or model loading."""
import http.client
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import grading
from eval_common import read, save, sha, digest, seal, lock


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if (HERE.parent/'grader').is_dir():
    GRADER = HERE.parent/'grader'
    UPSTREAM = GRADER/'healthbench_eval.py'
else:
    GRADER = REPO/'training/final_sft'
    UPSTREAM = REPO/'outgoing/final_sft_gold600_hb50_20261021_v2/server_final_sft/upstream/healthbench_eval.py'


class GradingTests(unittest.TestCase):
    def setUp(self):
        scratch = Path.cwd()/'tmp'
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temp.name)
        self.package = self.root/'package'
        self.work = self.root/'work'
        self.out = self.work/'grades'
        (self.package/'grader').mkdir(parents=True)
        for name in ('healthbench.py','common.py','GRADER_CONFIG.json'):
            shutil.copyfile(GRADER/name,self.package/'grader'/name)
        shutil.copyfile(UPSTREAM,self.package/'grader/healthbench_eval.py')
        self.questions = [dict(index=i,question_id='synthetic-%02d' % i,
            question='SYNTHETIC QUESTION %d' % i,benchmark='healthbench' if i<=50 else 'medbrowsecomp',
            native_prompt=[dict(role='user',content='SYNTHETIC QUESTION %d' % i)],
            rubrics=[dict(criterion='SYNTHETIC POSITIVE',points=2,tags=[]),
                     dict(criterion='SYNTHETIC NEGATIVE',points=-1,tags=[])])
            for i in range(1,101)]
        save(self.package/'data/questions.json',self.questions)
        self.identity = dict(protocol='process_benchmark_eval_v1',
            dataset_sha256=sha(self.package/'data/questions.json'), final_variant='base',
            arms=list(grading.ARMS), generated_arms=list(grading.NEW_ARMS), package_sha256='synthetic-only')
        save(self.work/'IDENTITY.json',self.identity)
        self.wdigest = digest(self.identity)

    def tearDown(self):
        self.temp.cleanup()

    def result(self,arm='base',index=1,status='complete'):
        folder=self.work/'results'/arm/('Q%02d' % index)
        answer='<answer>SYNTHETIC ANSWER %s %d</answer>' % (arm,index) if status=='complete' else None
        save(folder/'RESULT.json',dict(question_id='synthetic-%02d' % index,index=index,arm=arm,
            run_identity=self.wdigest,status=status,
            process_status='failed' if status in ('process_failure','engineering_failure') else 'complete',
            process_termination='synthetic_only',final_status='complete' if answer else 'not_started',
            answer=answer,answer_sha256=digest(answer) if answer else None,
            final_input_sha256='synthetic-input' if answer else None,issues=[]))
        save(folder/'raw.json',dict(synthetic=True))
        seal(folder,self.wdigest)
        return folder

    def run_bridge(self,post_fn,**kwargs):
        kwargs.setdefault('arms',grading.ARMS)
        return grading.run(self.package,self.work,self.out,key='synthetic-not-a-secret',
                           post_fn=post_fn,**kwargs)

    @staticmethod
    def valid_response(payload, *_):
        return dict(content=json.dumps(dict(criteria_met=True,explanation='SYNTHETIC explanation')))

    def test_all_four_arms_failures_missing_and_cache(self):
        for arm,status in zip(grading.ARMS,('complete','process_failure','final_failure','engineering_failure')):
            self.result(arm,status=status)
        calls=[]
        def post(*args):
            calls.append(args[0]);return self.valid_response(*args)
        counts=self.run_bridge(post)
        self.assertEqual(counts,dict(scored=1,deferred=0,unavailable=399,engineering_hold=0))
        self.assertEqual(len(calls),2)
        score=read(self.out/'scores/base/Q01/SCORE.json')
        self.assertEqual(score['score'],.5)
        self.assertFalse(score['official_leaderboard'])
        self.assertEqual(score['replacement_judge'],'qwen3.7-max')
        for arm,status in zip(grading.ARMS[1:],('process_failure','final_failure','engineering_failure')):
            folder=self.out/'scores'/arm/'Q01'
            self.assertEqual(read(folder/'STATUS.json')['status'],status)
            self.assertFalse((folder/'SCORE.json').exists())
        self.run_bridge(lambda *_:self.fail('cached score must not call HTTP'))
        self.assertEqual(read(self.out/'scores/base/Q02/STATUS.json')['status'],'generation_not_finished')

    def test_negative_score_not_clipped_per_question(self):
        self.result()
        def post(payload,*_):
            text=payload['messages'][-1]['content']
            criterion='SYNTHETIC NEGATIVE' in text
            return dict(content=json.dumps(dict(criteria_met=criterion,explanation='SYNTHETIC')))
        self.run_bridge(post,arms=('base',))
        self.assertEqual(read(self.out/'scores/base/Q01/SCORE.json')['score'],-.5)

    def test_corrupt_sealed_generation_is_hold_and_never_calls_http(self):
        folder=self.result()
        (folder/'raw.json').write_text('{}')
        counts=self.run_bridge(lambda *_:self.fail('corrupt source called HTTP'),arms=('base',))
        self.assertEqual(counts['engineering_hold'],1)
        self.assertEqual(read(self.out/'scores/base/Q01/STATUS.json')['status'],'engineering_hold')
        self.assertFalse((self.out/'scores/base/Q01/SCORE.json').exists())

    def test_old_score_preserved_but_invalid_after_source_damage(self):
        folder=self.result()
        self.run_bridge(self.valid_response,arms=('base',))
        old_sha=sha(self.out/'scores/base/Q01/SCORE.json')
        (folder/'raw.json').write_text('{}')
        self.run_bridge(lambda *_:self.fail('damaged source called HTTP'),arms=('base',))
        self.assertEqual(sha(self.out/'scores/base/Q01/SCORE.json'),old_sha)
        self.assertFalse(read(self.out/'scores/base/Q01/STATUS.json')['valid_score'])

    def test_timeout_retries_once_then_next_arm_still_scores(self):
        self.result('base');self.result('sft')
        calls=[]
        def post(payload,*args):
            calls.append(payload)
            if 'ANSWER base' in payload['messages'][-1]['content']:
                raise http.client.RemoteDisconnected()
            return self.valid_response(payload,*args)
        with patch('time.sleep'):
            counts=self.run_bridge(post,arms=('base','sft'))
        self.assertEqual(len(calls),4)
        self.assertEqual(counts['deferred'],1)
        self.assertEqual(counts['scored'],1)
        self.assertEqual(read(self.out/'scores/base/Q01/STATUS.json')['status'],'deferred')
        self.assertFalse((self.out/'scores/base/Q01/SCORE.json').exists())

    def test_unknown_prior_request_not_replayed(self):
        self.result()
        save(self.out/'scores/base/Q01/rubric-001/requests/old/STARTED.json',dict(synthetic=True))
        counts=self.run_bridge(lambda *_:self.fail('uncertain prior request replayed'),arms=('base',))
        self.assertEqual(counts['deferred'],1)
        self.assertEqual(read(self.out/'scores/base/Q01/STATUS.json')['completed_rubrics'],0)

    def test_grader_config_and_native_protocol_are_pinned(self):
        (self.package/'grader/GRADER_CONFIG.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'pinned_grader_changed'):
            self.run_bridge(lambda *_:self.fail('changed config called HTTP'))

    def test_legacy_common_alias_is_restored(self):
        self.result()
        sentinel=types.ModuleType('synthetic_common_sentinel')
        previous=sys.modules.get('common')
        sys.modules['common']=sentinel
        try:
            self.run_bridge(self.valid_response,arms=('base',))
            self.assertIs(sys.modules['common'],sentinel)
        finally:
            if previous is None:sys.modules.pop('common',None)
            else:sys.modules['common']=previous

    def test_real_http_has_explicit_allocated_gate(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ValueError,'explicit_allocated_opt_in'):
                grading.run(self.package,self.work,self.out,key='synthetic',real_api=True)

    def test_dataset_hash_change_blocks_before_http(self):
        self.questions[0]['question']='CHANGED SYNTHETIC QUESTION'
        save(self.package/'data/questions.json',self.questions)
        with self.assertRaisesRegex(ValueError,'evaluation_identity_changed'):
            self.run_bridge(lambda *_:self.fail('changed dataset called HTTP'))

    def test_shared_work_lock_prevents_grade_during_export(self):
        self.result()
        with lock(self.work/'EVAL.lock'):
            with self.assertRaises(OSError):
                self.run_bridge(lambda *_:self.fail('locked work called HTTP'))

    def test_default_grades_all_four_fresh_arms(self):
        for arm in grading.ARMS:self.result(arm)
        calls=[]
        def post(*args):
            calls.append(args[0]);return self.valid_response(*args)
        counts=grading.run(self.package,self.work,self.out,key='synthetic',post_fn=post)
        self.assertEqual(counts['scored'],4)
        self.assertEqual(len(calls),8)
        self.assertTrue((self.out/'scores/base/Q01/SCORE.json').exists())
        self.assertTrue((self.out/'scores/sft/Q01/SCORE.json').exists())
        self.assertEqual(read(self.out/'PASS_STATUS.json')['selected_arms'],list(grading.NEW_ARMS))


if __name__=='__main__':
    unittest.main()
