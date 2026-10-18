import unittest
from trainable_agent.context import ChatCodec

class TokenizerReturnTests(unittest.TestCase):
    def test_explicit_nonmapping_return(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                return [11,12,13] if kwargs.get('return_dict') is False else {'input_ids':[11,12,13],'attention_mask':[1,1,1]}
        self.assertEqual(ChatCodec(Tokenizer()).encode([{'role':'user','content':'fixture'}]),[11,12,13])

    def test_mapping_never_counted_as_two_tokens(self):
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs):return {'input_ids':[1,2],'attention_mask':[1,1]}
        with self.assertRaisesRegex(ValueError,'flat_integer'):ChatCodec(Tokenizer()).encode([])

    def test_noninteger_nested_or_empty_rejected(self):
        for output in [[],['input_ids','attention_mask'],[[1,2]],[True,3],[-1,2]]:
            with self.subTest(output=output):
                class Tokenizer:
                    def apply_chat_template(self,*args,**kwargs):return output
                with self.assertRaisesRegex(ValueError,'flat_integer'):ChatCodec(Tokenizer()).encode([])
