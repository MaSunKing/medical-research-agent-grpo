"""Per-trajectory resource state; permits consecutive reads and historical windows."""
import os
from browse_state_v28 import BrowseState as Previous
class BrowseState(Previous):
    def __init__(self,limit=None):
        super().__init__()
        self.limit=int(os.environ.get('V71_BROWSE_LIMIT','4')) if limit is None else limit
        if not isinstance(self.limit,int) or self.limit<1:raise ValueError('invalid browse limit')
        self.total=0;self.since_search=0;self.last_browse_turn=None
    def search_executed(self):self.since_search=0
    def begin(self,turn):
        if self.total>=self.limit:raise ValueError('browse budget exhausted')
        self.total+=1;self.since_search+=1;self.last_browse_turn=turn
    def check(self,search_tool,source_id,turn,query):
        blocked=super().check(search_tool,source_id,turn,query)
        if blocked:return blocked
        if self.total>=self.limit:
            return dict(kind='browse_budget_exhausted',source_id=source_id,executed=False,
                reason='per_trajectory_browse_limit',scope='current_trajectory',
                guidance='Not executed: Browse budget is exhausted. Use opened evidence to answer with limitations. Search cannot restore Browse budget.')
        return None
    def view(self,turn,candidates):
        unread=[]
        read_ids={key[1] for key in self.successful}
        for c in candidates:
            if c['source_id'] not in read_ids and not self.check(c['search_tool'],c['source_id'],turn,c['query']):unread.append(c['source_id'])
        paused=[dict(source_id=sid,route=route,**r) for (route,sid),r in self.records.items()
                if r['retry_after_turn'] is None or turn<r['retry_after_turn']]
        return dict(browse_executed_total=self.total,browse_limit=self.limit,browse_remaining=max(0,self.limit-self.total),
            browses_since_last_search=self.since_search,browse_this_turn=self.last_browse_turn==turn,
            already_browsed=sorted(read_ids),paused_sources=paused,available_unread_ids=unread,
            candidate_scope='current_window_from_cumulative_search_history',
            guidance='Consecutive Browse is allowed. Use returned candidate IDs only. A repeated successful read needs an explicit different focused query; no new Search is required.')
