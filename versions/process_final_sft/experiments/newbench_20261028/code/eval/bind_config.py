"""Extract path-only retrieval configuration on the server, never copy credentials."""
import argparse
import copy
from pathlib import Path
from eval_common import read,save,require
from evaluate import validate_config

def build(collection_config,training_root,output,base_model,original_adapter):
    source=read(collection_config)
    retrieval=copy.deepcopy(source['retrieval'])
    retrieval['snapshot']='retrieval_snapshot_v2'
    config=dict(base_model=str(base_model),original_adapter=str(original_adapter),
        training_output=str(Path(training_root)/'train'),retrieval=retrieval,seed=42)
    validate_config(config)
    require(not Path(output).exists(),'new_server_config_required')
    save(output,config,exclusive=True)
    return config

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection-config',type=Path,required=True)
    p.add_argument('--training-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base-model',type=Path,required=True)
    p.add_argument('--original-adapter',type=Path,required=True)
    a=p.parse_args();build(a.collection_config,a.training_root,a.output,a.base_model,a.original_adapter)
    print('SERVER_CONFIG_WRITTEN; path references only; secrets not printed; MODEL=0; API=0')
