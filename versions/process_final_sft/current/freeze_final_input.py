"""Prepare the real Final request without generating a Final or changing a trajectory."""
import copy
import sys
from pathlib import Path
from common import read, save, sha

sys.path.insert(0, str(Path(__file__).resolve().parent/'runtime/src'))
from trainable_agent.context import ChatCodec, ContextBuilder
from trainable_agent.compression import SemanticSelector
from trainable_agent.context_processing.fallback import ScopedFallback
from trainable_agent.model import EndpointModel
from trainable_agent.storage import Store, digest

async def prepare_request(state, context, store, compressor=None):
    before=digest(state)
    selector=SemanticSelector(compressor, context.codec, store, state['run_id'],10240) if compressor else None
    fallback=ScopedFallback(compressor, context.codec, store, state['run_id'],10240,
                            max_calls=context.compression_max_calls) if compressor else None
    request=await context.prepare('final',copy.deepcopy(state),selector,fallback=fallback)
    assert digest(state)==before, 'source_state_modified'
    assert request['input_sha256']==digest(request['messages'])
    assert context.codec.encode(request['messages'])==request['prompt_token_ids']
    request.update(stage='final', role='final')
    return request

async def freeze(folder, agent_config):
    from transformers import AutoTokenizer
    folder=Path(folder); run=folder/'run'; target=folder/'teacher_input'
    state=read(run/'TRAJECTORY.json'); config=read(agent_config)
    assert state['process_only'] and state['final'] is None and state['final_status']=='not_requested'
    if state['run_status']!='process_complete' or state.get('completion_mode','normal')!='normal':
        raise ValueError('normal_complete_Process_required_for_teacher_input')
    if target.exists():
        receipt=read(target/'INPUT_IDENTITY.json')
        assert receipt['trajectory_sha256']==sha(run/'TRAJECTORY.json')
        for n,h in receipt['files'].items(): assert sha(target/n)==h
        return receipt
    codec=ChatCodec(AutoTokenizer.from_pretrained(config['tokenizer'],local_files_only=True))
    context=ContextBuilder(codec,context_tokens=12288,soft_context_tokens=12288,hard_context_tokens=15000,
        repair_reserve_tokens=384,safety_margin=128,compression_max_calls=12)
    compressor=EndpointModel(capture_dir=folder/'final_input_compression',**config['compression']) if config.get('compression') else None
    store=Store(run/'trajectory.sqlite')
    try:
        event_start=len(store.events(state['run_id']))
        request=await prepare_request(state,context,store,compressor)
        freeze_events=store.events(state['run_id'])[event_start:]
    finally: store.db.close()
    target.mkdir(exist_ok=False)
    save(target/'FINAL_REQUEST.json',request)
    save(target/'PREPARATION_EVENTS.json',freeze_events)
    save(target/'MESSAGES.json',request['messages'])
    parts=[f"[{m['role']}]\n{m['content']}" for m in request['messages']]
    (target/'TEACHER_INPUT.txt').write_text('\n\n'.join(parts)+'\n',encoding='utf8')
    tokens=len(request['prompt_token_ids']); reserve=request['max_new_tokens']
    assert tokens+reserve+384+128<=15000
    receipt=dict(schema='frozen_final_input_v1',trajectory_sha256=sha(run/'TRAJECTORY.json'),
        raw_prefinal_sha256=sha(run/'PRE_FINAL_PACKAGE.json'),messages_sha256=digest(request['messages']),
        input_tokens=tokens,output_reserve=reserve,repair_reserve=384,safety_margin=128,hard_limit=15000,
        exact_qwen_tokenizer=True,teacher_generated=False,semantic_review='PENDING',
        evidence_count=len(state['evidence']),quality_hold=not bool(state['evidence']),
        files={n:sha(target/n) for n in ['FINAL_REQUEST.json','MESSAGES.json','TEACHER_INPUT.txt','PREPARATION_EVENTS.json']})
    save(target/'INPUT_IDENTITY.json',receipt)
    return receipt
