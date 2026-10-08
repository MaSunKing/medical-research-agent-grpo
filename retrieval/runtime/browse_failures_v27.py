"""Trajectory-local acquisition failures. No relevance/repetition blacklist."""
class BrowseFailures:
    def __init__(self):
        self.records = {}

    @staticmethod
    def key(search_tool, source_id):
        # This is the configured acquisition pipeline, not an individual PMC/PDF hop.
        # Different source aliases/routes are NOT merged or domain-blocked.
        return (search_tool, source_id)

    def check(self, search_tool, source_id, turn):
        r = self.records.get(self.key(search_tool, source_id))
        if not r or (r['retry_after_turn'] is not None and turn >= r['retry_after_turn']):
            return None
        return dict(kind='browse_acquisition_paused', source_id=source_id,
                    route=search_tool, reason=r['reason'], executed=False,
                    retry_after_turn=r['retry_after_turn'], scope='current_trajectory',
                    guidance='This acquisition route was not executed. Choose another available source or route, wait until the indicated decision turn, or finish. This is not evidence of irrelevance.')

    def observe(self, search_tool, source_id, output, evidence, turn):
        key=self.key(search_tool, source_id)
        if not evidence.get('failed') and any(c.get('text','').strip() for c in evidence.get('chunks',[])):
            self.records.pop(key,None)
            return
        # Whitelist structured error codes; never scan arbitrary document text/errors.
        code=output.get('error_code') or output.get('error') or output.get('failure_type')
        if not isinstance(code,str):return
        stable={'source_removed','http_410','access_denied','robots_denied'}
        temporary={'no_readable_biomedical_document','no_readable_webpage','timeout',
                   'rate_limited','connection_error','http_429','http_503'}
        if code not in stable|temporary:return
        previous=self.records.get(key,{})
        n=previous.get('failures',0)+1
        self.records[key]=dict(reason=code, failures=n,
            retry_after_turn=None if code in stable else turn+min(2**n,8),
            scope='current_trajectory')
