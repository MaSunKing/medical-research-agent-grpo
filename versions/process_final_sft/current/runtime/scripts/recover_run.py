"""Inspect by default; explicit resume only after uncertain calls are reconciled."""
import argparse,json
from pathlib import Path
from trainable_agent.storage import Store
from trainable_agent.recovery import inspect,prepare_resume

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--database",type=Path,required=True)
    p.add_argument("--run-id",required=True)
    p.add_argument("--prepare-resume",action="store_true")
    p.add_argument("--retry-failed-key",action="append",default=[])
    a=p.parse_args()
    if not a.database.is_file(): raise SystemExit("existing_database_required")
    store=Store(a.database)
    try:
        if a.prepare_resume: prepare_resume(store,a.run_id,a.retry_failed_key)
        print(json.dumps(inspect(store,a.run_id),ensure_ascii=False,indent=2))
    finally: store.db.close()
if __name__=="__main__":main()
