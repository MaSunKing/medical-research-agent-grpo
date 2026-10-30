"""Recompute public core-answer accuracy, denominators and information indicators."""
import csv
import itertools
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ARMS=('base','sft','grpo13','grpo25')
NAME=dict(zip(ARMS,('Base','Process-SFT','GRPO13','GRPO25')))
def load(name):return json.loads((HERE/name).read_text(encoding='utf8'))

def verify():
    rows=load('accuracy_records_400.json');summary=load('accuracy_summary.json')
    questions={q['index']:q for q in load('questions_100.json')}
    criteria={q['index']:q for q in load('answer_requirements_100.json')}
    originals={(r['index'],r['arm']):r for r in load('records_400.json')}
    assert len(rows)==400 and len(criteria)==len(questions)==100
    assert {(r['index'],r['arm']) for r in rows}==set(itertools.product(range(1,101),ARMS))
    with (HERE/'accuracy_scores_400.csv').open(encoding='utf8',newline='') as f:csvrows=list(csv.DictReader(f))
    assert len(csvrows)==400
    ci={(int(r['index']),r['arm']):r for r in csvrows}
    for r in rows:
        q=questions[r['index']];o=originals[r['index'],r['arm']];st=criteria[r['index']]
        assert r['question_id']==q['question_id']==st['question_id']
        assert r['question_sha256']==q['question_sha256']==st['question_sha256']
        assert st['required_core_content'] and st['source_urls']
        assert r['core_accuracy'] in (0,1) and r['accuracy_reason'].strip()
        assert r['prefinal_information_complete'] in (0,1)
        assert bool(r['prefinal_information_complete'])==all(p['support_score']==2 for p in r['prefinal_point_support'])
        assert all(p['support_score'] in (0,1,2) and p['reason'].strip() for p in r['prefinal_point_support'])
        assert r['native_answer_score']==o['benchmark_score']
        assert r['status_original']==o['status_original']
        assert r['original_process_score']==o['process_score'] and r['original_quality_value']==o['quality_value']
        assert r['original_major']==o['major']
        assert len(r['answer_sha256'])==64 and all(c in '0123456789abcdef' for c in r['answer_sha256'])
        assert 'answer' not in r and 'actual_context' not in r and 'quotes' not in r
        if r['status_original']!='complete':assert r['core_accuracy']==0
        c=ci[r['index'],r['arm']]
        assert int(c['core_accuracy'])==r['core_accuracy']
        assert int(c['prefinal_information_complete'])==r['prefinal_information_complete']
        assert c['accuracy_reason']==r['accuracy_reason'] and c['question_sha256']==r['question_sha256']
    for bench in ('healthbench','medbrowsecomp'):
        subset=[r for r in rows if r['benchmark']==bench]
        assert len(subset)==200
        for arm in ARMS:
            rs=[r for r in subset if r['arm']==arm];s=summary['groups'][bench][arm]
            assert len(rs)==s['total']==50
            assert sum(r['core_accuracy'] for r in rs)==s['correct']
            assert s['accuracy_percent']==100*s['correct']/50
            assert sum(r['prefinal_information_complete'] for r in rs)==s['information_complete']
            assert s['information_complete_percent']==2*s['information_complete']
        for a,b in itertools.combinations(ARMS,2):
            am={r['index']:r['core_accuracy'] for r in subset if r['arm']==a};bm={r['index']:r['core_accuracy'] for r in subset if r['arm']==b}
            pair=summary['paired_core_accuracy'][bench][a+'__'+b]
            assert pair=={'a_wins':sum(am[i]>bm[i] for i in am),'b_wins':sum(am[i]<bm[i] for i in am),'ties':sum(am[i]==bm[i] for i in am)}
    for i in range(1,101):
        page=(HERE/'questions'/f'Q{i:02}.md').read_text(encoding='utf8')
        assert page.count('## 核心答案与必要信息')==1
        for r in (r for r in rows if r['index']==i):
            assert f'|{NAME[r["arm"]]}|{r["accuracy_result"]}（{r["core_accuracy"]}）|{r["prefinal_information_complete"]}|' in page
            assert r['accuracy_reason'].replace('|','\\|').replace('\n',' ') in page
    assert summary['tool_budget']=={'normal_charged_calls':6,'environment_failure_exemptions_max':5,'actual_calls_max':11,'search_and_browse_share_budget':True}
    mb=[r for r in rows if r['benchmark']=='medbrowsecomp']
    assert sum(r['prefinal_information_complete'] for r in mb)==10
    assert sum(r['status_original']=='complete' for r in mb)==195
    print('PUBLIC_ACCURACY=passed; 400 decisions, fixed denominators, 100 question pages, original scores retained; MODEL=0; API=0; TRAINING=0')

if __name__=='__main__':verify()
