"""Trajectory-local successful-query ledger plus acquisition failures."""
from browse_failures_v27 import BrowseFailures

class BrowseState(BrowseFailures):
    def __init__(self):
        super().__init__()
        self.successful=set()

    @staticmethod
    def success_key(search_tool, source_id, query):
        if not isinstance(query,str) or not query.strip():
            raise ValueError('effective_browse_query_required')
        return (search_tool, source_id, ' '.join(query.split()).casefold())

    def check(self, search_tool, source_id, turn, query):
        failure=super().check(search_tool,source_id,turn)
        if failure:return failure
        if self.success_key(search_tool,source_id,query) in self.successful:
            return dict(kind='browse_already_read',source_id=source_id,route=search_tool,
                effective_query=query,executed=False,scope='current_trajectory',
                reason='same_source_and_effective_query_already_read',
                guidance='Not executed: this source was already read with this query. Use the existing evidence, select another source, supply an explicit different focused query to read this source, or finish. Omitting query reuses the inherited query.')
        return None

    def observe(self, search_tool, source_id, output, evidence, turn, query):
        super().observe(search_tool,source_id,output,evidence,turn)
        if not evidence.get('failed') and any(c.get('text','').strip() for c in evidence.get('chunks',[])):
            self.successful.add(self.success_key(search_tool,source_id,query))
