"""Synthetic CPU tests: contracts, exact gradients, reduction and resume. No real models/API."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import torch
from contract import (acceptance_groups,batches,digest,load_data,save,sha,safe,
                      validate_call,validate_config,require)
from objective import terms,backward_call
import runtime as rt

torch.set_num_threads(2)


def temporary():
    root=Path.cwd()/'tmp/process_grpo_update_tests'
    root.mkdir(parents=True,exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


def fixture():
    q=dict(role='process',stage='state',exact_tokenizer=True,messages=[dict(role='user',content='fixture')],
           max_new_tokens=8,prompt_token_ids=[1,2])
    q['input_sha256']=digest(q['messages'])
    r=dict(request_digest=digest(q),token_ids=[3,4],behavior_logprobs=[-1.,-2.],
        old_train_logprobs=[-1.1,-2.1],replay_logprobs=[-1.1,-2.1],cached_replay_logprobs=[-1.,-2.],
        policy_token_mask=[1,1],capture_mode='decoupled_capture_v1',training_eligible=False,
        correction_required=True,grpo_capture_ready=True,process_head='fp32_linear_unmerged',policy_identity='fixture',
        cached_replay_check=dict(passed=True,tolerance=1e-4,maximum_absolute_error=0.,mean_absolute_error=0.),
        replay_check=dict(passed=False,tolerance=.03,maximum_absolute_error=max(abs(-1.+1.1),abs(-2.+2.1)),
                          mode='full_forward_diagnostic_not_collection_gate'),
        sampling=dict(temperature=1.,effective_parameters_verified=True,effective=dict(do_sample=True,
            temperature=1.,top_p=1.,top_k=0,num_beams=1,repetition_penalty=1.,max_new_tokens=8,
            use_cache=True,num_return_sequences=1)),text='{"updates":[]}',usage={})
    return dict(call_id='0000_state',request=q,response=r,accepted=False,json_repaired=True)


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.randn(5,5)*.1,requires_grad=False)
        self.lora_A=torch.nn.Parameter(torch.randn(2,5)*.1);self.lora_B=torch.nn.Parameter(torch.randn(5,2)*.1)
    def forward(self,x):
        return torch.tanh(torch.nn.functional.linear(x,self.weight)+
            torch.nn.functional.linear(torch.nn.functional.linear(x,self.lora_A),self.lora_B))


class Body(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(9,5);self.embedding.weight.requires_grad_(False)
        self.layers=torch.nn.ModuleList([Layer(),Layer()])
    def forward(self,input_ids,attention_mask,use_cache):
        x=self.embedding(input_ids)
        for l in self.layers:x=l(x)
        return types.SimpleNamespace(last_hidden_state=x)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=Body();self.lm_head=torch.nn.Linear(5,9,bias=False).requires_grad_(False)
    def get_base_model(self):return self
    def save_pretrained(self,path,safe_serialization):
        from safetensors.torch import save_file
        save_file({n:p.detach().cpu().contiguous() for n,p in self.named_parameters() if p.requires_grad},
                  str(Path(path)/'adapter_model.safetensors'))
        save(Path(path)/'adapter_config.json',dict(synthetic=True))


def fake_peft():
    def state(m):return {n:p for n,p in m.named_parameters() if p.requires_grad}
    def put(m,s):
        with torch.no_grad():
            for n,p in m.named_parameters():
                if n in s:p.copy_(s[n])
    return types.SimpleNamespace(get_peft_model_state_dict=state,set_peft_model_state_dict=put)


class ContractTests(unittest.TestCase):
    def test_actual_rejection_and_repair_still_admitted(self):
        validate_call(fixture())
    def test_bad_cache_or_synthetic_tokens_rejected(self):
        for field,value in [('cached_replay_logprobs',[-1.2,-2.]),('policy_token_mask',[0,1]),
                            ('token_ids',[True,4]),('behavior_logprobs',[float('nan'),-2.])]:
            c=fixture();c['response'][field]=value
            with self.assertRaises(ValueError):validate_call(c)
    def test_wrong_role_sampling_or_messages_rejected(self):
        c=fixture();c['request']['role']='final'
        with self.assertRaises(ValueError):validate_call(c)
        c=fixture();c['response']['sampling']['effective']['temperature']=.6
        with self.assertRaises(ValueError):validate_call(c)
        c=fixture();c['request']['messages'][0]['content']='changed'
        with self.assertRaises(ValueError):validate_call(c)
    def test_paths_config_and_one_pass(self):
        for bad in ('../x','/x','C:/x','x\\y'):
            with self.assertRaises(ValueError):safe(Path('.'),bad)
        c=json.loads((Path(__file__).parent/'CONFIG.json').read_text());validate_config(c)
        c['passes']=2
        with self.assertRaises(ValueError):validate_config(c)
    def test_group_null_and_mean_advantage_verified(self):
        with temporary() as tmp:
            p=Path(tmp);save(p/'c.json',[fixture()]);rows=[]
            for n in range(4):rows.append(dict(rollout=n,R=None if n==0 else .5,A=None,
                calls_file='c.json',policy_identity='fixture',policy_tokens=2))
            save(p/'GROUPS.json',[dict(question_id='q',held=True,rollouts=rows)])
            save(p/'MANIFEST.json',dict(questions=1,records=4,files={n:sha(p/n) for n in ('GROUPS.json','c.json')}))
            self.assertTrue(load_data(p)[0][0]['held'])
            x=json.loads((p/'GROUPS.json').read_text());x[0]['rollouts'][1]['A']=.2;save(p/'GROUPS.json',x)
            m=json.loads((p/'MANIFEST.json').read_text());m['files']['GROUPS.json']=sha(p/'GROUPS.json');save(p/'MANIFEST.json',m)
            with self.assertRaises(ValueError):load_data(p)
    def test_plan_excludes_hold_preserves_ties_and_longest(self):
        gs=[dict(question_id=str(i),held=i==9,max_sequence_tokens=i,stratum='existing600_train_short' if i%2 else 'external_long_liveqa_train',
            rollouts=[dict(process_task_failure=i==4)]*4) for i in range(10)]
        c=json.loads((Path(__file__).parent/'CONFIG.json').read_text())
        p=batches(gs,c);self.assertEqual(sum(map(len,p)),9)
        self.assertEqual(p,batches(gs,c));self.assertNotIn(gs[9],[g for b in p for g in b])
        chosen=acceptance_groups(gs);self.assertEqual(len(chosen),8);self.assertIn(gs[8],chosen)


class GradientTests(unittest.TestCase):
    def test_deterministic_setup_is_strict_and_requires_workspace(self):
        before=(torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)
        try:
            with patch.dict('os.environ',{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}):
                rt.deterministic_setup(torch)
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
                self.assertFalse(torch.backends.cudnn.benchmark)
                self.assertTrue(torch.backends.cudnn.deterministic)
            with patch.dict('os.environ',{'CUBLAS_WORKSPACE_CONFIG':':16:8'}):
                with self.assertRaises(ValueError):rt.deterministic_setup(torch)
        finally:
            torch.use_deterministic_algorithms(before[0],warn_only=before[1])
            torch.backends.cudnn.benchmark=before[2];torch.backends.cudnn.deterministic=before[3]
    def test_is_algebra_detachment_and_zero_update_ratio(self):
        b=torch.tensor([-2.,-3.]);o=torch.tensor([-1.9,-3.1],requires_grad=True);v=o.detach().clone().requires_grad_()
        losses,s=terms(v,o,b,1.)
        self.assertFalse(s['raw_weight'].requires_grad);self.assertTrue(torch.equal(s['ratio'],torch.ones(2)))
        self.assertTrue(torch.allclose(s['raw_weight']*s['ratio'],(v.detach()-b).exp()))
        losses.mean().backward();self.assertIsNone(o.grad);self.assertTrue((v.grad<0).all())
    def test_negative_advantage_gradient_and_zero_advantage(self):
        v=torch.tensor([-2.],requires_grad=True);terms(v,[-2.],[-2.],-1.)[0].sum().backward()
        self.assertGreater(v.grad.item(),0)
        v=torch.tensor([-2.],requires_grad=True);terms(v,[-2.],[-2.],0.)[0].sum().backward()
        self.assertEqual(v.grad.item(),0)
    def test_ppo_clipping_both_signs_and_is_cap(self):
        v=torch.tensor([-1.],requires_grad=True);loss,s=terms(v,[-2.],[-4.],1.)
        loss.sum().backward();self.assertEqual(v.grad.item(),0);self.assertAlmostEqual(loss.item(),-2.4,places=5)
        v=torch.tensor([-3.],requires_grad=True);loss,_=terms(v,[-2.],[-2.],-1.)
        loss.sum().backward();self.assertEqual(v.grad.item(),0)
    def test_nan_and_shape_rejected(self):
        for v in (torch.tensor([float('nan')]),torch.tensor([-1.,-2.])):
            with self.assertRaises(ValueError):terms(v,[-1.],[-1.],1.)
    def test_resume_report_distinguishes_parameter_and_loss_failures(self):
        initial={'lora':torch.tensor([1.])};other={'lora':torch.tensor([1.0001])}
        opt={'0':{'step':torch.tensor(2.)}}
        r=rt.resume_comparison(initial,other,{'loss':.3},{'loss':.3},opt,opt)
        self.assertFalse(r['parameter_passed']);self.assertTrue(r['loss_passed'])
        self.assertEqual(r['adapter_tensors_above_tolerance'],1)
        r=rt.resume_comparison(initial,initial,{'loss':.3},{'loss':.3001},opt,opt)
        self.assertTrue(r['parameter_passed']);self.assertFalse(r['loss_passed'])
        with temporary() as tmp:
            save(Path(tmp)/'RESUME_COMPARISON.json',r)
            self.assertFalse(json.loads((Path(tmp)/'RESUME_COMPARISON.json').read_text())['thresholds_changed'])
    def test_equal_rollout_not_equal_call_normalization(self):
        # Two calls with 1 and 3 targets represent one four-token rollout.
        x=torch.tensor([-2.]*4,requires_grad=True)
        a=terms(x[:1],[-2.],[-2.],1.)[0].sum()/4
        b=terms(x[1:],[-2.]*3,[-2.]*3,1.)[0].sum()/4
        (a+b).backward();self.assertTrue(torch.equal(x.grad,torch.tensor([-.25]*4)))
    def test_decoder_and_vocab_checkpoint_gradients_match(self):
        torch.manual_seed(4);a=Toy().eval();b=copy.deepcopy(a);rt.install_checkpointing(b)
        c=fixture();va=rt.logprobs(a,c,gradient=True,chunk=1);vb=rt.logprobs(b,c,gradient=True,chunk=1)
        self.assertTrue(torch.equal(va,vb));va.sum().backward();vb.sum().backward()
        for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
            if pa.requires_grad:self.assertTrue(torch.allclose(pa.grad,pb.grad,atol=1e-8,rtol=1e-6),na)
            else:self.assertIsNone(pb.grad)
        self.assertFalse(any(m.training for m in b.modules()))
    def test_no_grad_and_grad_forward_align(self):
        m=Toy().eval();rt.install_checkpointing(m);c=fixture()
        self.assertTrue(torch.equal(rt.logprobs(m,c),rt.logprobs(m,c,gradient=True).detach()))
    def test_real_backward_changes_only_adapter(self):
        torch.manual_seed(3);m=Toy().eval();rt.install_checkpointing(m);c=fixture()
        old=rt.logprobs(m,c).tolist();c['response']['behavior_logprobs']=old
        params=[p for p in m.parameters() if p.requires_grad];frozen={n:p.clone() for n,p in m.named_parameters() if not p.requires_grad}
        optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=0)
        backward_call(m,c,old,1.,2,dict(epsilon=.2,is_cap=2),rt.logprobs)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
        before=[p.detach().clone() for p in params];optimizer.step()
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,params)))
        self.assertTrue(all(torch.equal(frozen[n],p) for n,p in m.named_parameters() if n in frozen))
    def test_sealed_save_restore_optimizer_and_resume_equivalence(self):
        torch.manual_seed(3);m=Toy().eval();rt.install_checkpointing(m);c=fixture();old=rt.logprobs(m,c).tolist()
        c['response']['behavior_logprobs']=old;opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
        def step():
            opt.zero_grad(set_to_none=True);backward_call(m,c,old,1.,2,dict(epsilon=.2,is_cap=2),rt.logprobs);opt.step()
        with temporary() as tmp,patch.dict('sys.modules',{'peft':fake_peft()}),\
             patch.object(torch.cuda,'get_rng_state_all',return_value=[]),patch.object(torch.cuda,'set_rng_state_all'):
            step();cp=rt.save_checkpoint(m,opt,Path(tmp),1,dict(fixture=True),{})
            step();expected=rt.state_digest(rt.adapter_state(m));expected_opt=copy.deepcopy(opt.state_dict())
            self.assertEqual(rt.restore(m,opt,cp,dict(fixture=True)),1);step()
            self.assertEqual(rt.state_digest(rt.adapter_state(m)),expected)
            for key,s in expected_opt['state'].items():
                for k,v in s.items():self.assertTrue(torch.equal(v,opt.state_dict()['state'][key][k]))
            with self.assertRaises(ValueError):rt.restore(m,opt,cp,dict(fixture=False))
            (cp/'adapter_config.json').write_text('{}')
            with self.assertRaises(ValueError):rt.restore(m,opt,cp,dict(fixture=True))


class AlignmentDiagnosticTests(unittest.TestCase):
    def prepare(self,old_offset=0.,zero_advantage=False):
        torch.manual_seed(61);model=Toy().eval()
        calls={};rollouts=[]
        for index,advantage in enumerate((-.3,-.1,.1,.3)):
            call=fixture();call['call_id']=str(index)+'_state'
            call['response']['token_ids']=[index+3,index+4]
            old=rt.logprobs(model,call).tolist()
            call['response']['behavior_logprobs']=old
            call['response']['old_train_logprobs']=[v+old_offset for v in old]
            name=str(index)+'.json';calls[name]=[call]
            rollouts.append(dict(calls_file=name,policy_tokens=2,
                A=0. if zero_advantage else advantage))
        params={n:p for n,p in model.named_parameters() if p.requires_grad}
        optimizer=torch.optim.AdamW(list(params.values()),lr=1e-6,weight_decay=0)
        config=json.loads((Path(__file__).parent/'CONFIG.json').read_text())
        return model,params,optimizer,[dict(rollouts=rollouts)],calls,config

    def update(self,prepared,**kwargs):
        model,params,optimizer,batch,calls,config=prepared
        with patch.object(rt,'call_path',side_effect=lambda data,r:r['calls_file']),\
             patch.object(rt,'read',side_effect=lambda path:calls[path]),\
             patch.object(rt,'reference_values',side_effect=lambda ref,r,c:c['response']['old_train_logprobs']),\
             patch.object(torch.cuda,'max_memory_allocated',return_value=0):
            return rt.update_batch(model,params,optimizer,None,None,batch,config,**kwargs)

    def test_unmeasured_alignment_is_null_not_zero(self):
        metrics=self.update(self.prepare())
        self.assertTrue(metrics['optimizer_step'])
        self.assertFalse(metrics['initial_alignment_checked'])
        self.assertEqual(metrics['initial_alignment_calls'],0)
        self.assertIsNone(metrics['initial_alignment_max_abs'])
        self.assertIn('"initial_alignment_max_abs": null',json.dumps(metrics))

    def test_measured_zero_and_nonzero_alignment_are_distinct(self):
        for offset in (0.,5e-5):
            with self.subTest(offset=offset):
                metrics=self.update(self.prepare(old_offset=offset),forward_check=True)
                self.assertTrue(metrics['initial_alignment_checked'])
                self.assertEqual(metrics['initial_alignment_calls'],4)
                if offset==0:self.assertEqual(metrics['initial_alignment_max_abs'],0.)
                else:
                    self.assertGreater(metrics['initial_alignment_max_abs'],0.)
                    self.assertLessEqual(metrics['initial_alignment_max_abs'],1e-4)

    def test_empty_and_zero_advantage_batches_report_only_actual_probes(self):
        for empty in (False,True):
            for enabled in (False,True):
                with self.subTest(empty=empty,enabled=enabled):
                    prepared=self.prepare(zero_advantage=True)
                    if empty:prepared[3].clear()
                    before={n:p.detach().clone() for n,p in prepared[0].named_parameters()}
                    metrics=self.update(prepared,forward_check=enabled)
                    measured=enabled and not empty
                    self.assertFalse(metrics['optimizer_step'])
                    self.assertEqual(metrics['initial_alignment_checked'],measured)
                    self.assertEqual(metrics['initial_alignment_calls'],4 if measured else 0)
                    self.assertEqual(metrics['initial_alignment_max_abs'],0. if measured else None)
                    self.assertFalse(prepared[2].state)
                    for n,p in prepared[0].named_parameters():
                        self.assertTrue(torch.equal(p,before[n]),n)

    def test_probe_does_not_change_loss_or_optimizer_update(self):
        unchecked=self.prepare();checked=self.prepare()
        a=self.update(unchecked);b=self.update(checked,forward_check=True)
        self.assertEqual(a['loss'],b['loss'])
        self.assertEqual(a['gradient_norm'],b['gradient_norm'])
        for (na,pa),(nb,pb) in zip(unchecked[0].named_parameters(),checked[0].named_parameters()):
            self.assertEqual(na,nb);self.assertTrue(torch.equal(pa,pb),na)
        for key,state in unchecked[2].state_dict()['state'].items():
            for n,value in state.items():
                self.assertTrue(torch.equal(value,checked[2].state_dict()['state'][key][n]))

    def test_interruption_keeps_none_contract_and_discards_partial_gradients(self):
        for completed in (0,1):
            with self.subTest(completed=completed):
                prepared=self.prepare();seen=0
                before={n:p.detach().clone() for n,p in prepared[0].named_parameters()}
                def stop():
                    nonlocal seen
                    seen+=1
                    return seen>completed
                self.assertIsNone(self.update(prepared,forward_check=True,stop=stop))
                self.assertFalse(prepared[2].state)
                for n,p in prepared[0].named_parameters():
                    self.assertTrue(torch.equal(p,before[n]),n)
                    self.assertIsNone(p.grad)

    def test_alignment_tolerance_hold_still_precedes_optimizer_step(self):
        prepared=self.prepare(old_offset=.001)
        before={n:p.detach().clone() for n,p in prepared[0].named_parameters()}
        with self.assertRaisesRegex(ValueError,'initial_grad_forward_alignment_hold'):
            self.update(prepared,forward_check=True)
        self.assertFalse(prepared[2].state)
        for n,p in prepared[0].named_parameters():
            self.assertTrue(torch.equal(p,before[n]),n)


if __name__=='__main__':unittest.main(verbosity=2)
