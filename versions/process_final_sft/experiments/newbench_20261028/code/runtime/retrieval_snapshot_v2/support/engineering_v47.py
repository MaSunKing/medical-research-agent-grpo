"""Runtime-only metadata and resource-feedback corrections; no policy changes."""
import re

def install_metadata(module):
    original = module.infer_medical_source_metadata
    if getattr(original, 'no_design_v58', False):
        return
    def infer(**kwargs):
        kwargs = dict(kwargs)
        kwargs['publication_types'] = list(kwargs.get('publication_types') or [])
        result = original(**kwargs)
        identity = ' '.join([str(kwargs.get('title', '')), *kwargs['publication_types']]).casefold()
        methods = bool(re.search(r'\b(?:reporting guideline|methodological standard|consort|strobe statement|prisma statement)\b', identity))
        clinical_guideline = bool(re.search(r'\bguidelines?\b', identity))
        if not methods and 'consensus statement' in identity:
            result['authority_type'] = 'guideline' if clinical_guideline else 'none'
            if result['study_design'] == 'methodological_standard':
                result['study_design'] = 'unknown'
        # Preserve the internal schema without treating keyword matches as facts.
        # Native publication_types remain on the original source record.
        result['study_design'] = 'unknown'
        return result
    infer.no_design_v58 = True
    module.infer_medical_source_metadata = infer

def install_budget(cls):
    gate, view = cls.gate, cls.view
    def corrected_gate(self, route, sid, turn, query, explicit):
        if self.total >= self.limit:
            return dict(kind='browse_budget_exhausted', source_id=sid, executed=False,
                        reason='per_trajectory_browse_limit', scope='current_trajectory',
                        guidance='Not executed: no Browse calls remain. A different query or source cannot restore the quota. Finish using opened evidence and state any unresolved limitations.')
        return gate(self, route, sid, turn, query, explicit)
    def corrected_view(self, turn, candidates, tool_remaining):
        result = view(self, turn, candidates, tool_remaining)
        if result['effective_browse_remaining'] == 0:
            result['guidance'] = 'No Browse calls remain. Do not request another Browse. Search cannot restore the quota. Finish using opened evidence and state any unresolved limitations.'
        return result
    cls.gate, cls.view = corrected_gate, corrected_view
