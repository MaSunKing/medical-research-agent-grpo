"""Adversarial offline fixtures; no pretrained model or real HTTP."""
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import eval_common as ec
import evaluate
import grading
import ops
import prepare_data as pd
import review_export as rexp

HERE=Path(__file__).resolve().parent

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.package=self.root/'package';self.work=self.root/'work'
        self.rows=[dict(index=i,question_id='test-'+str(i),source_id='source-'+str(i),
            benchmark='healthbench' if i<=50 else 'medbrowsecomp',cluster='cluster-'+str(i),
            question='Synthetic question '+str(i),native_prompt=[dict(role='user',content='Synthetic question '+str(i))],
            rubrics=[dict(criterion='SYNTHETIC REFERENCE '+str(i),points=1,tags=[])]) for i in range(1,101)]
        ec.save(self.package/'data/questions.json',self.rows)
        ec.save(self.package/'data/DATASET.json',dict(benchmarks={'healthbench':{},'medbrowsecomp':{}}))
        for name in grading.PINNED_GRADER_FILES:
            src=HERE.parent/'grader'/name
            if not src.exists():src=Path(os.environ['NEWBENCH_TEST_PACKAGE'])/'grader'/name
            target=self.package/'grader'/name;target.parent.mkdir(exist_ok=True);shutil.copyfile(src,target)
        ec.save(self.package/'FILES.sha256.json',ec.manifest(self.package))
        self.identity=dict(protocol='process_benchmark_eval_v1',arms=list(ec.ARMS),generated_arms=list(ec.ARMS),
            dataset_sha256=ec.sha(self.package/'data/questions.json'),package_sha256=ec.sha(self.package/'FILES.sha256.json'),
            final_variant='base')
        ec.save(self.work/'IDENTITY.json',self.identity);self.rid=ec.digest(self.identity)

    def tearDown(self):self.temp.cleanup()

    def result(self,arm,index,status='complete',database=False):
        q=self.rows[index-1];folder=ec.record_folder(self.work,arm,index)
        req=dict(role='final',stage='final',messages=[dict(role='user',content='ACTUAL DELIVERED EVIDENCE')])
        req['input_sha256']=ec.digest(req['messages'])
        answer='SYNTHETIC answer' if status=='complete' else None
        ec.save(folder/'RESULT.json',dict(index=index,question_id=q['question_id'],benchmark=q['benchmark'],
            source_id=q['source_id'],arm=arm,run_identity=self.rid,status=status,process_status='complete',
            final_status='complete' if answer else 'not_started',final_variant='base',
            answer=answer,answer_sha256=ec.digest(answer) if answer else None,
            final_input_sha256=req['input_sha256'] if answer else None,issues=[]))
        ec.save(folder/'EVENTS.json',[dict(kind='model_request',data=dict(request=req))] if answer else [])
        ec.save(folder/'PRE_FINAL_PACKAGE.json',dict(diagnostic_only='DO NOT SUBSTITUTE'))
        if answer:ec.save(folder/'FINAL_INPUT.json',req)
        if database:
            with closing(sqlite3.connect(folder/'trajectory.sqlite')) as db,db:
                db.execute('create table fixture (x integer)');db.execute('insert into fixture values(1)')
        ec.seal(folder,self.rid)
        return folder

    def score(self,arm,index,score):
        g=self.work/'grades';ident=dict(work_identity=self.rid,synthetic=True)
        ec.save(g/'IDENTITY.json',ident);p=g/'scores'/arm/f'Q{index:02d}'
        source=ec.record_folder(self.work,arm,index)/'RESULT.json'
        ec.save(p/'SCORE.json',dict(index=index,question_id=self.rows[index-1]['question_id'],
            process_arm=arm,final_variant='base',score=score,identity=ec.digest(ident),final_sha256=ec.sha(source)))
        ec.save(p/'STATUS.json',dict(status='scored',valid_score=True,score_sha256=ec.sha(p/'SCORE.json')))

    def test_grid_all400_and_no_false_zero(self):
        grid=rexp.grid(self.package,self.work)
        self.assertEqual(len(grid),400)
        self.assertTrue(all(x['benchmark_score'] is None and x['process_score'] is None for x in grid))
        summary=rexp.summarize(grid)
        self.assertEqual(set(summary['benchmarks']),{'healthbench','medbrowsecomp'})
        self.assertNotIn('raw_mean',summary)

    def test_summary_uses_separate_denominators_and_clip_after_mean(self):
        for arm,index,score in [('base',1,-.4),('base',2,.2),('sft',1,.1),('base',51,1),('sft',51,0)]:
            self.result(arm,index);self.score(arm,index,score)
        summary=rexp.summarize(rexp.grid(self.package,self.work))['benchmarks']
        hb=summary['healthbench']['arms']['base'];mb=summary['medbrowsecomp']['arms']['base']
        self.assertAlmostEqual(hb['raw_mean'],-.1);self.assertEqual(hb['clipped_mean'],0)
        self.assertEqual(mb['valid_answer_accuracy'],1);self.assertEqual(mb['correct_deliveries_per_planned_question'],.02)
        pair=summary['healthbench']['paired']['base__sft']['benchmark_score']
        self.assertEqual(pair['common_questions'],1);self.assertEqual(pair['delta_b_minus_a'],.5)

    def test_grading_dispatch_and_cache_both_benchmarks(self):
        self.result('base',1);self.result('sft',51)
        calls=[]
        def post(payload,*args):
            text=payload['messages'][-1]['content'];calls.append(text)
            return dict(content=json.dumps(dict(criteria_met=True,explanation='Synthetic comparison')))
        grading.run(self.package,self.work,self.work/'grades',arms=('base','sft'),key='fixture',post_fn=post)
        self.assertEqual(len(calls),2)
        self.assertTrue(any('Do not solve the' in s for s in calls))
        med=ec.read(self.work/'grades/scores/sft/Q51/SCORE.json')
        self.assertEqual(med['score'],1);self.assertFalse(med['native_HealthBench_rubrics'])
        self.assertEqual(med['benchmark'],'medbrowsecomp')
        grading.run(self.package,self.work,self.work/'grades',arms=('base','sft'),key='fixture',
            post_fn=lambda *_:self.fail('cache repeated request'))

    def test_both_false_med_grade_is_valid_zero(self):
        self.result('base',51)
        grading.run(self.package,self.work,self.work/'grades',arms=('base',),key='fixture',
            post_fn=lambda *_:dict(content='{"criteria_met":false,"explanation":"Different final answer"}'))
        self.assertEqual(ec.read(self.work/'grades/scores/base/Q51/SCORE.json')['score'],0)
        self.assertTrue(ec.read(self.work/'grades/scores/base/Q51/STATUS.json')['valid_score'])

    def test_model_failure_retained_unscored(self):
        folder=self.result('base',51,'process_failure')
        grading.run(self.package,self.work,self.work/'grades',arms=('base',),key='fixture',
            post_fn=lambda *_:self.fail('failed generation graded'))
        row=next(x for x in rexp.grid(self.package,self.work) if x['arm']=='base' and x['index']==51)
        self.assertEqual(row['status'],'process_failure');self.assertIsNone(row['benchmark_score']);self.assertTrue(folder.exists())

    def test_export_whole_grid_sqlite_hashes_templates_and_sources_unchanged(self):
        self.result('base',1,database=True);self.result('sft',51,'process_failure')
        before=ec.manifest(self.work)
        archive=rexp.export_bundle(self.package,self.work,self.root/'export')
        after=ec.manifest(self.work,excludes=('EVAL.lock',))
        self.assertEqual(before,after)
        with tarfile.open(archive) as tar:
            members={m.name:m for m in tar if m.isfile()}
            base='process_grpo_eval_results/'
            self.assertIn(base+'work/results/base/Q01/trajectory.sqlite',members)
            allrows=json.load(tar.extractfile(members[base+'summary/ALL_400_RECORDS.json']))
            self.assertEqual(len(allrows),400)
            template=json.load(tar.extractfile(members[base+'review-template/records/grpo25/Q100/review.json']))
            self.assertIsNone(template['p1']);self.assertEqual(template['status'],'pending')
            cp=json.load(tar.extractfile(members[base+'review-template/questions/Q51.json']))
            self.assertEqual(cp['status'],'pending')

    def test_export_rejects_nonempty_wal(self):
        f=self.result('base',1,database=True);(f/'trajectory.sqlite-wal').write_bytes(b'pending')
        with self.assertRaisesRegex(ValueError,'WAL_or_SHM'):rexp.export_bundle(self.package,self.work,self.root/'export')

    def test_actual_delivery_not_raw_prefinal(self):
        f=self.result('base',1);r=ec.read(f/'RESULT.json');d=rexp.delivery(f,r)
        self.assertEqual(d['request']['messages'][0]['content'],'ACTUAL DELIVERED EVIDENCE')

    def test_reject_fabricated_delivery(self):
        f=self.result('base',1);ec.save(f/'FINAL_INPUT.json',dict(messages=[]))
        with self.assertRaisesRegex(ValueError,'original_first_request'):rexp.delivery(f,ec.read(f/'RESULT.json'))

    def test_no_historical_reuse(self):
        ec.save(self.package/'data/HISTORICAL_HEALTHBENCH.json',{})
        with self.assertRaisesRegex(ValueError,'historical_results_forbidden'):rexp.grid(self.package,self.work)

    def test_pilot_gate_all_eight_and_preserves_failure(self):
        with self.assertRaises((FileNotFoundError,ValueError)):ops.require_pilot(self.work,self.identity)
        for arm in ec.ARMS:
            for index in (1,51):self.result(arm,index,'process_failure' if arm=='base' and index==1 else 'complete')
        ops.validate_pilot(self.package,self.work);ops.require_pilot(self.work,self.identity)
        f=ec.record_folder(self.work,'sft',1);(f/'RESULT.json').write_text('{}')
        with self.assertRaises(ValueError):ops.require_pilot(self.work,self.identity)

    def test_pilot_rejects_engineering_failure(self):
        for arm in ec.ARMS:
            for index in (1,51):self.result(arm,index,'engineering_failure' if arm=='sft' and index==51 else 'complete')
        with self.assertRaisesRegex(ValueError,'engineering_failure'):ops.validate_pilot(self.package,self.work)
        self.assertFalse((self.work/'PILOT_ACCEPTANCE.json').exists())

    def test_selection_excludes_normalized_and_near_duplicates(self):
        source='What is the current recommendation for treatment of this specific medical condition in adults?'
        ex=[dict(question_id='old',normalized_text=pd.normalized(source))]
        self.assertEqual(pd.overlap(source.upper().replace(' ','  '),'new',ex)[0],'normalized_exact')
        self.assertEqual(pd.overlap(source.replace('current','latest'),'new',ex)[0],'near_duplicate')
        self.assertEqual(pd.overlap('Completely different question','old',ex)[0],'id')

    def test_indices_cover_both_benchmarks_without_question_mutation(self):
        selected=evaluate.selected_questions(self.rows,100,[1,51])
        self.assertEqual([q['benchmark'] for q in selected],['healthbench','medbrowsecomp'])
        with self.assertRaises(ValueError):evaluate.selected_questions(self.rows,100,[101])

    def test_balanced_schedule_exact400_and_four_arms_per_block(self):
        units=list(evaluate.work_units(self.rows,list(ec.ARMS)))
        cells=[(q['index'],arm) for qs,arm in units for q in qs]
        self.assertEqual(len(cells),400);self.assertEqual(len(set(cells)),400)
        self.assertEqual([arm for _,arm in units[:4]],list(ec.ARMS))
        self.assertEqual([arm for _,arm in units[4:8]],['sft','grpo13','grpo25','base'])
        self.assertEqual({q['index'] for qs,_ in units[:4] for q in qs},set(range(1,6)))

    def test_coordinate100_and_reject101(self):
        self.assertEqual(ec.record_folder(self.work,'grpo25',100).name,'Q100')
        with self.assertRaises(ValueError):ec.record_folder(self.work,'base',101)

    def test_mismatched_score_hash_is_not_consumed(self):
        f=self.result('base',51);self.score('base',51,1)
        ec.save(self.work/'grades/scores/base/Q51/SCORE.json',dict(score=0))
        row=next(x for x in rexp.grid(self.package,self.work) if x['arm']=='base' and x['index']==51)
        self.assertIsNone(row['benchmark_score']);self.assertIn('Benchmark_score_integrity_hold',row['issues'])

    def test_secret_scanner_and_path_traversal(self):
        p=self.root/'secret';p.write_bytes(b'Bearer '+b'x'*32)
        with self.assertRaises(ValueError):rexp.scan_file(p)
        for name in ('../outside','/tmp/out','C:out','a\\b'):
            with self.assertRaises(ValueError):ec.safe(self.root,name)

    def test_submit_runs_compute_only_after_slurm_heredoc(self):
        script=(HERE/'submit.sh').read_text(encoding='utf8')
        login,compute=script.split("<<'SH'",1)
        self.assertNotIn('unittest',login)
        self.assertNotIn('offline.py" --package',login)
        self.assertIn('-m unittest discover',compute)
        self.assertIn('CPUS=8;MEM=28G',login)
        self.assertIn('watch -n 60',script)

    def test_judge_smoke_only_selected_records_then_full_uses_cache(self):
        for arm in ec.ARMS:
            for i in (1,51):self.result(arm,i)
        calls=[]
        def post(*args):
            calls.append(args[0]);return dict(content='{"criteria_met":true,"explanation":"Synthetic equivalence"}')
        grading.run(self.package,self.work,self.work/'grades',key='fixture',post_fn=post,indices=[1,51])
        self.assertEqual(len(calls),8)
        ops.validate_judge(self.package,self.work)
        grading.run(self.package,self.work,self.work/'grades',key='fixture',
            post_fn=lambda *_:self.fail('pilot graded twice'))

if __name__=='__main__':unittest.main()
