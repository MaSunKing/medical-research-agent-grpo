"""Synthetic interface-only example; no medical answer or evaluation question."""
import json

QUESTION=('What benefits does treatment A offer adults with disease B? '
          'What serious adverse events does treatment A cause in adults with disease B? '
          'Use guideline webpages and comparative research papers. '
          'Answer briefly in Chinese.')
CONSTRAINTS=['Use guideline webpages and comparative research papers.','Answer briefly in Chinese.']
BAD={'anchors':[
    ['What benefits does treatment A offer adults with disease B?'],
    ['What serious adverse events does treatment A cause in adults with disease B?']]}
GOOD={'anchors':[fragments+CONSTRAINTS for fragments in BAD['anchors']]}

def example():
    return ('\nUnrelated format example; do not copy its topic or requirements into the actual task.'
            '\nExample question: '+QUESTION+'\nCorrect output: '+json.dumps(GOOD,separators=(',',':'))+
            '\nEach item contains a complete request sentence. Shared constraints apply to both '
            'requests; all quotes come from this example question. A and B are fictional '
            'placeholders, not topics to add to the actual task.')

def repair_example():
    return ('\nUnrelated correction example (not a tool result): omitting the two final sentences from '
            'the treatment A example is incomplete. Keep both complete request sentences and append the exact source '
            'and answer-language sentences to each applicable anchors list, as in its correct output. '
            'Do not repeat an unchanged rejected output. Apply this procedure to the actual question only.')
