"""Offline source-recovery contracts, including actual subprocess restarts."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'retrieval_snapshot_v2'
if str(SNAPSHOT) not in sys.path:sys.path.insert(0, str(SNAPSHOT))
from source_registry import restore_completed_searches, restore_search_result, restore_web_search_rows
from paired_receipts import PairedReceiptStore
from trainable_agent.retrieval_deadlines import OperationJournal, reconcile_operation

SPEC = importlib.util.spec_from_file_location('source_registry_test_dual',
    SNAPSHOT / 'backend/agent/dr_agent/mcp_backend/apis/dual_paper.py')
DUAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DUAL)
WEB_URL = 'https://www.nice.org.uk/guidance/ng136'
PAPER_ID = 'a' * 40


def web_functions():
    """Load real registration and URL-safety definitions, without API clients."""
    path = SNAPSHOT / 'backend/agent/dr_agent/mcp_backend/apis/medical_web_apis.py'
    names = {'SOURCE_TYPE_DOMAINS', '_MEDICAL_WEB_SOURCE_URLS', '_MEDICAL_WEB_SOURCE_METADATA',
             '_MEDICAL_WEB_SOURCE_URLS_LOCK', 'medical_web_source_id', 'register_medical_web_sources',
             'get_medical_web_source_metadata', 'resolve_medical_web_source', '_host_matches',
             'is_allowed_medical_url', 'is_safe_public_web_url', 'is_supported_medical_web_url'}
    selected = []
    for node in ast.parse(path.read_text(encoding='utf8')).body:
        found = ({node.name} if isinstance(node, ast.FunctionDef) else
                 {t.id for t in node.targets if isinstance(t, ast.Name)} if isinstance(node, ast.Assign) else
                 {node.target.id} if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) else set())
        if found & names:selected.append(node)
    package = types.ModuleType('source_registry_fixture_web')
    package.__path__ = [str(path.parent)]
    sys.modules[package.__name__] = package
    scope = dict(__name__=package.__name__ + '.registry', __package__=package.__name__)
    exec('import hashlib, threading, os, re, ipaddress\nfrom typing import Dict, Iterable, List\n'
         'from urllib.parse import urlparse, parse_qs', scope)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), 'exec'), scope)
    return scope


def registries():
    web, paper = web_functions(), DUAL.Registry()
    def register_web(rows):
        # Same broad discovery scope used by the real worker Search and Browse.
        previous = os.environ.get('MEDGAP_WEB_DISCOVERY_MODE')
        os.environ['MEDGAP_WEB_DISCOVERY_MODE'] = 'broad'
        try:restore_web_search_rows(rows, lookup=web['get_medical_web_source_metadata'],
                                   register=web['register_medical_web_sources'])
        finally:
            if previous is None:os.environ.pop('MEDGAP_WEB_DISCOVERY_MODE', None)
            else:os.environ['MEDGAP_WEB_DISCOVERY_MODE'] = previous
    def restore(value):
        return restore_search_result(value, register_web=register_web, register_paper=paper.restore)
    return web, paper, restore


def values():
    web = web_functions()
    return ({'data':[{'source_id':web['medical_web_source_id'](WEB_URL), 'url':WEB_URL,
                      'title':'Hypertension in adults', 'snippet':'Synthetic search preview'}]},
            {'data':[{'source_id':'S2:'+PAPER_ID, 'paperId':PAPER_ID, 'pmid':'12345',
                      'externalIds':{'PubMed':'12345'}, 'title':'Synthetic original paper',
                      'abstract':'Original synthetic abstract',
                      'url':'https://www.semanticscholar.org/paper/'+PAPER_ID,
                      'openAccessPdf':{'url':'https://example.org/original.pdf'},
                      'identity_aliases':['S2:'+PAPER_ID,'PMID:12345']}]})


def fixture_worker(output, skip_startup=False):
    """Synthetic fetcher with actual journal, registries and recovery component."""
    web, paper, restore = registries()
    if not skip_startup:restore_completed_searches(output, restore)
    store = PairedReceiptStore(output/'paired', release_id='synthetic-source-fixture', mode='freeze')
    print(json.dumps({'status':'ready'}), flush=True)
    for line in sys.stdin:
        message = json.loads(line)
        op, args, identity = message['operation'], message['arguments'], message['operation_id']
        try:
            journal = OperationJournal(output, identity, op, args)
            value, _ = reconcile_operation(output, identity, op, args)
            disposition = 'completed_receipt' if value is not None else None
            if value is None:
                journal.start()
                def execute():
                    if op == 'search':
                        count = output/'search_count.txt'
                        count.write_text(str(int(count.read_text())+1 if count.exists() else 1))
                        return copy.deepcopy(values()[0 if args['tool']=='medical_web_search' else 1])
                    sid = args['source_id']
                    if sid.startswith('WEB:'):
                        return {'data':[{'url':web['resolve_medical_web_source'](sid),
                                         'metadata':web['get_medical_web_source_metadata'](sid)}]}
                    if sid not in paper.rows:raise ValueError('unregistered_paper_source')
                    return {'data':[copy.deepcopy(paper.rows[sid])]}
                value, receipt = store.resolve(op, args, execute)
                disposition = receipt['disposition']
                if op == 'search':restore(value)
                journal.finish(value)
            elif op == 'search':restore(value)
            print(json.dumps({'status':'ok','value':value,'disposition':disposition}), flush=True)
        except Exception as exc:
            print(json.dumps({'status':'error','error':str(exc)}), flush=True)


class FixtureCase(unittest.TestCase):
    def setUp(self):
        scratch = ROOT/'tmp/worker-source-registry-tests'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.web, self.paper, self.restore = registries()

    def journal(self, value, operation='search', identity='1'*64):
        journal = OperationJournal(self.root, identity, operation, {'tool':'synthetic'})
        journal.start();journal.finish(value)
        return journal


class RegistryTests(FixtureCase):
    def test_native_identity_graph_stays_out_of_model_candidate_projection(self):
        from trainable_agent.tools import normalize
        row = DUAL.Registry().fuse({'semantic_scholar':[
            {'paperId':PAPER_ID, 'externalIds':{'PubMed':'12345'},
             'title':'Synthetic paper', 'abstract':'Synthetic preview'}]})[0]
        self.assertTrue(row['native_identity_groups'])
        projected = normalize({'data':[row]}, 'paper')
        without_graph = copy.deepcopy(row)
        without_graph.pop('native_identity_groups')
        self.assertEqual(projected, normalize({'data':[without_graph]}, 'paper'))
        self.assertNotIn('native_identity_groups', projected['candidates'][0])

    def paper_merge_history(self):
        registry = DUAL.Registry()
        first = copy.deepcopy(registry.fuse({'pubmed':[
            {'pmid':'12345', 'title':'Original paper'}]}))
        second = copy.deepcopy(registry.fuse({'semantic_scholar':[
            {'paperId':PAPER_ID, 'doi':'10.1000/original',
             'title':'Provider title', 'abstract':'Provider abstract'}]}))
        merged = copy.deepcopy(registry.fuse({'semantic_scholar':[
            {'paperId':PAPER_ID, 'externalIds':{'PubMed':'12345'}}]}))
        self.assertEqual(first[0]['source_id'], 'PMID:12345')
        self.assertEqual(second[0]['source_id'], 'S2:'+PAPER_ID)
        self.assertEqual(merged[0]['source_id'], 'PMID:12345')
        return registry, (first, second, merged)

    def test_independent_sources_merge_then_restart_and_search_keep_canonical(self):
        live, history = self.paper_merge_history()
        for i, rows in enumerate(history, 1):
            j = self.journal({'data':rows}, identity=f'{i:064x}')
            os.utime(j.folder/'state.json', ns=(i*1_000_000_000, i*1_000_000_000))
        restore_completed_searches(self.root, self.restore)
        self.assertEqual(self.paper.by_alias, live.by_alias)
        # Both already-returned IDs remain directly readable after the merge.
        for rows in history[:2]:
            sid = rows[0]['source_id']
            for key, value in live.rows[sid].items():
                # Optional unknown fields may be absent instead of null.
                self.assertEqual(self.paper.rows[sid].get(key), value)
        query = {'semantic_scholar':[{'paperId':PAPER_ID}]}
        self.assertEqual(self.paper.fuse(query), live.fuse(query))
        self.assertEqual(self.paper.by_alias['S2:'+PAPER_ID], 'PMID:12345')

    def test_old_search_and_paired_replay_do_not_undo_native_identity_merge(self):
        live, history = self.paper_merge_history()
        store = PairedReceiptStore(self.root/'paired', release_id='synthetic-paper-merge', mode='freeze')
        arguments = {'tool':'pubmed_search', 'query':'synthetic provider query'}
        store.resolve('search', arguments, lambda:{'data':history[1]})
        for rows in history:self.restore({'data':rows})
        expected = copy.deepcopy(self.paper.by_alias)
        self.restore({'data':history[1]})
        replay, receipt = store.resolve('search', arguments,
                                       lambda:self.fail('cached Search must not execute again'))
        self.assertEqual(receipt['disposition'], 'replayed')
        self.restore(replay)
        self.assertEqual(self.paper.by_alias, expected)
        self.assertEqual(self.paper.by_alias, live.by_alias)
        self.assertEqual(self.paper.fuse({'semantic_scholar':[{'paperId':PAPER_ID}]}),
                         live.fuse({'semantic_scholar':[{'paperId':PAPER_ID}]}))

    def test_transitive_native_aliases_follow_merge_even_when_absent_from_new_row(self):
        live, history = self.paper_merge_history()
        self.assertNotIn('DOI:10.1000/original', DUAL.aliases(history[2][0]))
        for rows in history:self.paper.restore(rows)
        # The old S2 DOI is not in the final row, but its existing verified
        # component was joined; a DOI-only follow-up must use the same ID.
        self.assertEqual(live.by_alias['DOI:10.1000/original'], 'PMID:12345')
        self.assertEqual(self.paper.by_alias, live.by_alias)
        query = {'semantic_scholar':[{'doi':'10.1000/original'}]}
        self.assertEqual(self.paper.fuse(query), live.fuse(query))

    def test_s2_discovered_before_pmid_keeps_its_returned_canonical(self):
        live = DUAL.Registry()
        first = copy.deepcopy(live.fuse({'semantic_scholar':[{'paperId':PAPER_ID}]}))
        merged = copy.deepcopy(live.fuse({'semantic_scholar':[
            {'paperId':PAPER_ID, 'externalIds':{'PubMed':'12345'}}]}))
        self.assertEqual(merged[0]['source_id'], 'S2:'+PAPER_ID)
        self.paper.restore(first);self.paper.restore(merged);self.paper.restore(first)
        self.assertEqual(self.paper.by_alias, live.by_alias)
        self.assertEqual(self.paper.by_alias['PMID:12345'], 'S2:'+PAPER_ID)

    def test_matching_titles_and_claimed_aliases_do_not_merge_unrelated_papers(self):
        other_id = 'b'*40
        registry = DUAL.Registry()
        first = copy.deepcopy(registry.fuse({'pubmed':[{'pmid':'12345', 'title':'Same title'}]}))
        second = copy.deepcopy(registry.fuse({'semantic_scholar':[
            {'paperId':other_id, 'title':'Same title'}]}))
        # A syntactically valid claimed alias is not native identity evidence.
        second[0]['identity_aliases'].append('PMID:12345')
        self.paper.restore(first);self.paper.restore(second)
        self.assertEqual(self.paper.by_alias['PMID:12345'], 'PMID:12345')
        self.assertEqual(self.paper.by_alias['S2:'+other_id], 'S2:'+other_id)

    def test_native_discovery_groups_keep_multiple_s2_and_doi_identifiers(self):
        live = DUAL.Registry()
        first = copy.deepcopy(live.fuse({'pubmed':[{'pmid':'12345'}]}))
        second = copy.deepcopy(live.fuse({'semantic_scholar':[
            {'paperId':PAPER_ID, 'doi':'10.1000/first',
             'externalIds':{'PubMed':'12345'}}]}))
        extra = 'c'*40
        third = copy.deepcopy(live.fuse({'semantic_scholar':[
            {'paperId':extra, 'externalIds':{'PubMed':'12345','DOI':'10.1000/second'}}]}))
        # The display/native row deliberately retains its first useful values.
        self.assertEqual(third[0]['paperId'], PAPER_ID)
        self.assertEqual(third[0]['doi'], '10.1000/first')
        self.assertNotIn('S2:'+extra, DUAL.aliases(third[0]))
        self.assertNotIn('DOI:10.1000/second', DUAL.aliases(third[0]))
        # A single latest cached row carries the actual provider-ID history.
        self.paper.restore(third)
        self.assertEqual(self.paper.by_alias, live.by_alias)
        self.assertEqual(self.paper.rows['PMID:12345'], third[0])
        # Restoring the older result may not erase either later identity link.
        self.paper.restore(first);self.paper.restore(second)
        self.assertEqual(self.paper.by_alias, live.by_alias)
        query = {'semantic_scholar':[{'paperId':extra}]}
        self.assertEqual(self.paper.fuse(query), live.fuse(query))

    def test_fuse_projects_only_native_id_fields_and_ignores_supplied_groups(self):
        rows = DUAL.Registry().fuse({'semantic_scholar':[
            {'paperId':PAPER_ID, 'externalIds':{'PubMed':'12345','DOI':'10.1000/example',
                                             'CorpusId':'unused'},
             'title':'Original title', 'native_identity_groups':[{'paperId':'b'*40}]}]})
        self.assertEqual(rows[0]['native_identity_groups'], [
            {'paperId':PAPER_ID, 'externalIds':{'PubMed':'12345','DOI':'10.1000/example'}}])
        self.paper.restore(rows)
        self.assertNotIn('S2:'+'b'*40, self.paper.by_alias)

    def test_invalid_or_disconnected_native_groups_are_rejected_atomically(self):
        valid = DUAL.Registry().fuse({'pubmed':[{'pmid':'12345'}]})[0]
        bad_values = [None, 'not-a-list', [None], [{}],
                      [{'paperId':'not-an-id'}],
                      [{'pmid':'12345','title':'not an ID field'}],
                      [{'pmid':'12345','externalIds':{'PubMed':'99999'}}],
                      [{'pmid':'12345','externalIds':{'CorpusId':'unused'}}],
                      [{'paperId':'b'*40}],
                      [{'pmid':'99999','paperId':'b'*40}]]
        for groups in bad_values:
            with self.subTest(groups=groups):
                bad = copy.deepcopy(valid);bad['native_identity_groups'] = groups
                target = DUAL.Registry()
                with self.assertRaises(ValueError):target.restore([valid, bad])
                self.assertEqual(target.rows, {})
                self.assertEqual(target.by_alias, {})

    def test_restore_web_and_canonical_s2_with_later_pmid_without_refusing_id(self):
        web, paper = values()
        self.journal(web, identity='1'*64);self.journal(paper, identity='2'*64)
        self.assertEqual(restore_completed_searches(self.root, self.restore), 2)
        sid = web['data'][0]['source_id']
        self.assertEqual(self.web['resolve_medical_web_source'](sid), WEB_URL)
        row = self.paper.rows['S2:'+PAPER_ID]
        self.assertEqual(row, paper['data'][0])
        self.assertEqual(self.paper.by_alias['PMID:12345'], 'S2:'+PAPER_ID)

    def test_canonical_pmid_survives_later_s2_only_discovery_aliases(self):
        registry=DUAL.Registry()
        registry.fuse({'pubmed':[{'pmid':'12345','doi':'10.1000/example','title':'Original paper'}]})
        rows=registry.fuse({'semantic_scholar':[{'paperId':PAPER_ID,
             'externalIds':{'DOI':'10.1000/example'},'abstract':'Provider abstract'}]})
        row=rows[0]
        self.assertEqual(row['source_id'],'PMID:12345')
        self.assertNotIn('PMID:12345',row['identity_aliases'])
        restored=DUAL.Registry()
        restored.restore(rows)
        self.assertEqual(restored.rows['PMID:12345'],row)
        self.assertEqual(restored.by_alias['S2:'+PAPER_ID],'PMID:12345')

    def test_repeated_paper_restore_keeps_latest_discovery_provenance(self):
        registry=DUAL.Registry()
        first=copy.deepcopy(registry.fuse({'pubmed':[{'pmid':'12345','doi':'10.1000/example',
              'title':'Original paper'}]}))
        second=copy.deepcopy(registry.fuse({'semantic_scholar':[{'paperId':PAPER_ID,
               'externalIds':{'DOI':'10.1000/example'},'abstract':'Provider abstract'}]}))
        restored=DUAL.Registry()
        restored.restore(first);restored.restore(second)
        self.assertEqual(restored.rows['PMID:12345'],second[0])
        self.assertEqual(restored.by_alias['S2:'+PAPER_ID],'PMID:12345')

    def test_fresh_web_public_projection_keeps_private_discovery_metadata(self):
        public=values()[0]
        full=copy.deepcopy(public['data'][0])
        full.update(discovery_query='original search focus',original_question='complete original question',
                    candidate_score=.9,candidate_rank=1)
        previous=os.environ.get('MEDGAP_WEB_DISCOVERY_MODE')
        os.environ['MEDGAP_WEB_DISCOVERY_MODE']='broad'
        try:self.web['register_medical_web_sources']([full])
        finally:
            if previous is None:os.environ.pop('MEDGAP_WEB_DISCOVERY_MODE',None)
            else:os.environ['MEDGAP_WEB_DISCOVERY_MODE']=previous
        self.restore(public)
        saved=self.web['get_medical_web_source_metadata'](full['source_id'])
        for key in ('discovery_query','original_question','candidate_score','candidate_rank'):
            self.assertEqual(saved[key],full[key])
        self.assertEqual(saved['url'],WEB_URL)

    def test_incomplete_and_browse_journals_do_not_register_any_source(self):
        self.journal(values()[0], operation='browse')
        OperationJournal(self.root, '2'*64, 'search', {}).start()
        self.assertEqual(restore_completed_searches(self.root, self.restore), 0)
        self.assertEqual(self.paper.rows, {})

    def test_completion_order_not_operation_hash_order_preserves_web_metadata(self):
        old=values()[0];old['data'][0]['title']='First discovery metadata'
        new=values()[0];new['data'][0]['title']='Latest discovery metadata'
        first=self.journal(old, identity='f'*64)
        second=self.journal(new, identity='0'*64)
        # Deterministic file metadata: the later Search has a lexically earlier
        # operation ID, so sorting IDs would incorrectly restore the old title.
        os.utime(first.folder/'state.json', ns=(1_000_000_000,1_000_000_000))
        os.utime(second.folder/'state.json', ns=(2_000_000_000,2_000_000_000))
        restore_completed_searches(self.root,self.restore)
        sid=old['data'][0]['source_id']
        self.assertEqual(self.web['get_medical_web_source_metadata'](sid)['title'],'Latest discovery metadata')

    def test_production_worker_wires_startup_completed_and_paired_restore_paths(self):
        tree=ast.parse((SNAPSHOT/'worker.py').read_text(encoding='utf8'))
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        calls=[n for n in ast.walk(main) if isinstance(n,ast.Call)]
        startup=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='restore_completed_searches')
        ready=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='emit'
                   and n.args and isinstance(n.args[0],ast.Call)
                   and any(k.arg=='status' and isinstance(k.value,ast.Constant) and k.value.value=='ready'
                           for k in n.args[0].keywords))
        self.assertLess(startup.lineno,ready.lineno)
        recovered=next(n for n in ast.walk(main) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
                       and isinstance(n.test.left,ast.Name) and n.test.left.id=='recovered')
        recovered_calls=[n for n in ast.walk(recovered) if isinstance(n,ast.Call)]
        self.assertTrue(any(isinstance(n.func,ast.Name) and n.func.id=='restore_sources' for n in recovered_calls))
        resolve=next(n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='resolve'
                     and isinstance(n.func.value,ast.Name) and n.func.value.id=='receipt_store')
        restore_value=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=='restore_sources'
                           and n.args and isinstance(n.args[0],ast.Name) and n.args[0].id=='value')
        finish=next(n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='finish'
                    and isinstance(n.func.value,ast.Name) and n.func.value.id=='journal')
        self.assertLess(resolve.lineno,restore_value.lineno)
        self.assertLess(restore_value.lineno,finish.lineno)

    def test_corrupt_hash_is_rejected_before_any_registry_callback(self):
        self.journal(values()[0], identity='1'*64)
        bad = self.journal(values()[1], identity='2'*64)
        (bad.folder/'result.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'result_hash_mismatch'):
            restore_completed_searches(self.root, self.restore)
        self.assertEqual(self.web['_MEDICAL_WEB_SOURCE_URLS'], {})

    def test_changed_journal_identity_and_invalid_operation_directory_are_rejected(self):
        j = self.journal(values()[0]);p=j.folder/'state.json';state=json.loads(p.read_text())
        state['operation_id']='2'*64;p.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError,'identity_mismatch'):
            restore_completed_searches(self.root, self.restore)
        p.write_text('{}')
        (self.root/'worker_operations/foreign-path').mkdir()
        # Invalid path is checked regardless of operation state.
        with self.assertRaises(ValueError):restore_completed_searches(self.root, self.restore)

    def test_invalid_source_id_and_web_hash_url_mismatch_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'invalid_source_id'):
            self.restore({'data':[{'source_id':'WEB:../escape','url':WEB_URL}]})
        value=values()[0];value['data'][0]['url']='https://example.org/changed'
        with self.assertRaisesRegex(ValueError,'does not match'):
            self.restore(value)

    def test_private_web_and_paper_urls_and_false_paper_aliases_are_rejected(self):
        web = values()[0];web['data'][0]['url']='http://127.0.0.1/private'
        web['data'][0]['source_id']=self.web['medical_web_source_id'](web['data'][0]['url'])
        with self.assertRaisesRegex(ValueError,'unsafe or unsupported'):self.restore(web)
        paper=values()[1];paper['data'][0]['openAccessPdf']['url']='https://192.168.1.1/private.pdf'
        with self.assertRaisesRegex(ValueError,'unsafe_restored_paper_url'):self.restore(paper)
        paper=values()[1];paper['data'][0]['paperId']='b'*40
        with self.assertRaisesRegex(ValueError,'identity_mismatch'):self.restore(paper)
        paper=values()[1];paper['data'][0]['identity_aliases']=['S2:not-a-real-id']
        with self.assertRaisesRegex(ValueError,'invalid_restored_paper_aliases'):self.restore(paper)

    def test_symlinked_result_and_operation_directory_are_rejected(self):
        j=self.journal(values()[0]);original=j.folder/'original.json'
        (j.folder/'result.json').rename(original)
        try:(j.folder/'result.json').symlink_to(original)
        except OSError:self.skipTest('symlink privilege unavailable')
        with self.assertRaisesRegex(ValueError,'unsafe_receipt_path'):
            restore_completed_searches(self.root,self.restore)

    def test_symlink_guards_fail_before_callback_even_without_os_symlink_privilege(self):
        j=self.journal(values()[0]);original=Path.is_symlink
        for protected in (j.folder/'result.json', j.folder, self.root/'worker_operations'):
            with self.subTest(path=protected.name), patch.object(Path,'is_symlink',
                    lambda p: p==protected or original(p)):
                with self.assertRaisesRegex(ValueError,'unsafe_receipt_path|invalid_operation_path'):
                    restore_completed_searches(self.root,self.restore)
        self.assertEqual(self.web['_MEDICAL_WEB_SOURCE_URLS'],{})


class SubprocessRecoveryTests(FixtureCase):
    def start_worker(self, skip=False):
        env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
                 PYTHONPATH=os.pathsep.join([str(ROOT/'src'),str(SNAPSHOT)]))
        command=[sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'--fixture-worker',str(self.root)]
        if skip:command.append('--skip-startup')
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                                 text=True,encoding='utf8',cwd=ROOT,env=env)
        def stop():
            if process.poll() is None:
                process.stdin.close();process.wait(timeout=15)
            process.stdout.close();process.stderr.close()
        self.addCleanup(stop)
        self.assertEqual(json.loads(process.stdout.readline()),{'status':'ready'})
        return process

    def call(self, process, operation, arguments, n):
        process.stdin.write(json.dumps(dict(operation=operation,arguments=arguments,operation_id=f'{n:064x}'))+'\n')
        process.stdin.flush()
        return json.loads(process.stdout.readline())

    def seed(self):
        first=self.start_worker()
        for n,tool in ((1,'medical_web_search'),(2,'pubmed_search')):
            self.assertEqual(self.call(first,'search',{'tool':tool},n)['status'],'ok')
        first.stdin.close();first.wait(timeout=15)
        self.assertEqual((self.root/'search_count.txt').read_text(),'2')

    def assert_browse_both(self, process):
        sid=values()[0]['data'][0]['source_id']
        self.assertEqual(self.call(process,'browse',{'source_id':sid},3)['value']['data'][0]['url'],WEB_URL)
        row=self.call(process,'browse',{'source_id':'S2:'+PAPER_ID},4)['value']['data'][0]
        self.assertEqual(row['source_id'],'S2:'+PAPER_ID)
        self.assertEqual(row['abstract'],'Original synthetic abstract')
        self.assertEqual(row['pmid'],'12345')
        self.assertEqual((self.root/'search_count.txt').read_text(),'2')

    def test_real_subprocess_restart_restores_web_s2_and_does_not_search_again(self):
        self.seed();second=self.start_worker();self.assert_browse_both(second)

    def test_completed_operation_replay_reregisters_sources_without_search(self):
        self.seed();second=self.start_worker(skip=True)
        for n,tool in ((1,'medical_web_search'),(2,'pubmed_search')):
            result=self.call(second,'search',{'tool':tool},n)
            self.assertEqual(result['disposition'],'completed_receipt')
        self.assert_browse_both(second)

    def test_paired_receipt_replay_reregisters_sources_without_search(self):
        self.seed();second=self.start_worker(skip=True)
        for n,tool in ((11,'medical_web_search'),(12,'pubmed_search')):
            result=self.call(second,'search',{'tool':tool},n)
            self.assertEqual(result['disposition'],'replayed')
        self.assert_browse_both(second)

    def test_missing_receipt_browse_fails_without_searching(self):
        process=self.start_worker()
        result=self.call(process,'browse',{'source_id':'S2:'+PAPER_ID},1)
        self.assertEqual(result,{'status':'error','error':'unregistered_paper_source'})
        self.assertFalse((self.root/'search_count.txt').exists())


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--fixture-worker':
        fixture_worker(Path(sys.argv[2]),'--skip-startup' in sys.argv)
    else:unittest.main()
