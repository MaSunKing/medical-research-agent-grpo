"""Require an explicit new focus only for already successfully read sources."""
from browse_budget_v34 import BrowseState as Previous
class BrowseState(Previous):
    def gate(self,route,sid,turn,query,explicit):
        read=any(key[1]==sid for key in self.successful)
        if read and not explicit:
            return dict(kind='browse_focus_required',source_id=sid,executed=False,
                reason='already_read_missing_explicit_query',scope='current_trajectory',
                guidance='Not executed: this source was already read. To reread, explicitly include query="the specific new question to inspect" on the Browse call. Do not reuse the previous focus. Otherwise use existing evidence, choose an unread source, or finish.')
        blocked=self.check(route,sid,turn,query)
        if blocked and blocked['kind']=='browse_already_read':
            blocked=dict(blocked,reason='explicit_focus_already_read',
                guidance='Not executed: this explicit query was already read for this source. Supply a genuinely different focus, choose an unread source, or finish.')
        return blocked
