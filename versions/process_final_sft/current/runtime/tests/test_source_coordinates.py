import copy,importlib.util,json,random,sys,unittest
from pathlib import Path
from trainable_agent.context_processing.browser import clean_text,normalized_source_blocks,preprocess_document
from test_cleaning_adaptation import parser,handoff
from trainable_agent.prompts import PROMPTS
from trainable_agent.contracts import validate_action

class Coordinates(unittest.TestCase):
    def test_parser_cleans_before_block_coordinates(self):
        doc=parser.parse_medical_markdown('# Results\nPatients did not improve.\x01\n12% had adverse events.\x03')
        original=copy.deepcopy(doc);out=preprocess_document(doc)
        self.assertEqual(doc,original)
        for section in out['sections']:
            for block in section['source_blocks']:
                self.assertLessEqual(block['end_char'],len(section['text']))
        self.assertIn('did not improve',out['text']);self.assertIn('12%',out['text'])
    def test_legacy_cached_blocks_rebased_and_audited(self):
        left='Patients did not improve.\x01';right='12% had adverse events.\x03'
        original=left+' '+right
        blocks=[dict(start_char=0,end_char=len(left)),dict(start_char=len(left)+1,end_char=len(original))]
        doc={'sections':[{'heading':'Results','text':original,'source_blocks':blocks}]}
        out=preprocess_document(doc);s=out['sections'][0]
        self.assertEqual([s['text'][b['start_char']:b['end_char']] for b in s['source_blocks']],['Patients did not improve.','12% had adverse events.'])
        self.assertEqual(s['source_block_normalization']['original_blocks'],blocks)
        again=preprocess_document(out)
        self.assertEqual(again['sections'],out['sections'])
    def test_all_ascii_controls_and_whitespace_match_shared_normalizer(self):
        for char in [chr(i) for i in range(32)]+['\x7f','\u00a0','\u2003']:
            text='  Patients'+char+'  did not improve.  '
            body,blocks,audit=normalized_source_blocks(text,[dict(start_char=0,end_char=len(text))])
            self.assertEqual(body,clean_text(text));self.assertEqual(blocks,[dict(start_char=0,end_char=len(body))])
    def test_empty_control_only_block_removed_without_neighbor_loss(self):
        text='\x01\x03 Patients did not improve.'
        body,blocks,audit=normalized_source_blocks(text,[dict(start_char=0,end_char=2),dict(start_char=3,end_char=len(text))])
        self.assertEqual(len(blocks),1);self.assertEqual(body,'Patients did not improve.')
        self.assertEqual(len(audit['empty_blocks']),1)
    def test_genuinely_invalid_original_coordinates_still_rejected(self):
        for a,b in [(-1,5),(0,999),(5,5),(True,5)]:
            with self.assertRaisesRegex(ValueError,'invalid_source_block_coordinates'):
                normalized_source_blocks('Patients did not improve.',[dict(start_char=a,end_char=b)])
    def test_normalization_random_shapes_and_idempotence(self):
        rand=random.Random(17)
        for _ in range(100):
            text=''.join(rand.choice('abc 12%.\x01\x03\x7f\t\n\r') for _ in range(100))
            body,blocks,_=normalized_source_blocks(text,[dict(start_char=0,end_char=len(text))])
            self.assertEqual(body,clean_text(text))
            self.assertEqual(normalized_source_blocks(body,blocks)[:2],(body,blocks))
    def test_prompt_matches_action_requirement_limit(self):
        self.assertIn('1–2 existing R IDs',PROMPTS['decision'])
        self.assertIn('source aliases',PROMPTS['decision'])


class ActionBindingContract(unittest.TestCase):
    def test_one_or_two_bindings_allowed_three_rejected_for_both_actions(self):
        from pydantic import ValidationError
        state=dict(requirements={'R1':'guidance','R2':'safety','R3':'effect'},candidates=[{'source_id':'PMID:1'}],history=[])
        for action in [dict(action='search',channel='paper',query='focused evidence'),dict(action='browse',source_id='PMID:1',reading_focus='focused evidence')]:
            for ids in [['R1'],['R1','R2']]:
                self.assertEqual(validate_action(dict(action,requirement_ids=ids),state)['requirement_ids'],ids)
            for ids in [[],['R1','R2','R3']]:
                with self.assertRaises(ValidationError):validate_action(dict(action,requirement_ids=ids),state)


class SafeFailure(unittest.TestCase):
    def test_worker_locations_exclude_exception_message_and_locals(self):
        root=Path(__file__).resolve().parents[1];snap=root/'retrieval_snapshot_v2'
        sys.path.insert(0,str(snap))
        try:
            spec=importlib.util.spec_from_file_location('safe_worker_diagnostic',snap/'worker.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        finally:sys.path.remove(str(snap))
        private='DO_NOT_SERIALIZE_PRIVATE_VALUE'
        try:raise ValueError(private)
        except ValueError as exc:rows=module.safe_error_locations(exc,root)
        self.assertTrue(rows);self.assertNotIn(private,json.dumps(rows))
        self.assertEqual(set(rows[-1]),{'file','function','line'})
