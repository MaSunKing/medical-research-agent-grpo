import json
from pathlib import Path
from core import sha,validate_record
def verify_data(data):
    audit=data.parent/'audit'
    exact=json.loads((audit/'EXACT_RETOKENIZATION_ACCEPTANCE.json').read_text(encoding='utf8'))
    lineage=json.loads((audit/'ROW_LINEAGE.json').read_text(encoding='utf8'))
    selected=json.loads((audit/'SELECTION_ACCEPTANCE.json').read_text(encoding='utf8'))
    if exact['status']!='passed' or exact['text_sha256']!=sha(data/'train.history_aware.jsonl') or exact['encoded_sha256']!=sha(data/'stepwise.jsonl'):
        raise ValueError('Exact tokenizer receipt mismatch')
    if len(lineage)!=6741 or len({r['question_id'] for r in lineage})!=700 or any(r['stage']=='final' for r in lineage):
        raise ValueError('Process-only lineage mismatch')
    for line in (data/'stepwise.jsonl').open(encoding='utf8'):validate_record(json.loads(line))
    for line in (data/'train.history_aware.jsonl').open(encoding='utf8'):
        if set(json.loads(line))!={'messages','assistant_prefix','completion','template_options'}:
            raise ValueError('Unclean training fields')
