"""CPU-only verification, no models or external API."""
import argparse
from pathlib import Path
from eval_common import read, require, sha, verify_files, verify_package
from prepare_data import validate_rows
from grading import load_legacy

def check(package):
    package=Path(package);identity=verify_package(package)
    rows=read(package/'data/questions.json')
    validate_rows(rows,read(package/'data/KNOWN_EXCLUSIONS.json')['records'])
    meta=read(package/'data/DATASET.json')
    require(meta['planned_records']==400 and meta['fresh_arms']==['base','sft','grpo13','grpo25'],
            'four_fresh_arms_required')
    require(meta['historical_result_reuse'] is False,'historical_results_forbidden')
    require(not list((package/'data').glob('HISTORICAL*')),'historical_results_forbidden')
    verify_files(package/'runtime',read(package/'SOURCE_IDENTITY.json')['runtime_files'])
    load_legacy(package)
    print('OFFLINE_VERIFIED; benchmarks=2; questions=100; fresh_records=400; MODEL=0; API=0',flush=True)
    return dict(package_sha256=identity,questions=100,records=400)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True)
    check(p.parse_args().package)
