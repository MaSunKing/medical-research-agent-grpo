"""Receipt-settled budgets: six charged calls plus three environment failures."""
ENVIRONMENT_CODES = {
    'timeout', 'rate_limited', 'connection_error', 'http_429', 'http_503',
    'http_403', 'http_410', 'access_denied', 'robots_denied', 'source_removed',
    'no_readable_biomedical_document', 'no_readable_webpage',
    'archive_transport_failed', 'pdf_transport_failed',
}
NON_ENVIRONMENT_CODES = {
    'search_no_results', 'no_supported_medical_web_results', 'empty_results',
    'invalid_args', 'invalid_arguments', 'invalid_tool_call', 'no_relevant_content',
}


def environment_failure(action, output, evidence=None):
    """Never classify empty/readable/irrelevant results by scanning prose."""
    codes = {output.get(key) for key in ('error_code', 'failure_type', 'error')
             if isinstance(output.get(key), str)}
    if codes & NON_ENVIRONMENT_CODES:
        return False
    if action == 'browse' and evidence and not evidence.get('failed'):
        if any(c.get('text', '').strip() for c in evidence.get('chunks', [])):
            return False
    return (output.get('environment_failure') is True
            or bool(codes & ENVIRONMENT_CODES)
            or (output.get('failed') is True
                and output.get('failure_type') == 'environment_failure'))


class ToolBudget:
    def __init__(self, limit=6, exemptions=3):
        if limit != 6 or exemptions != 3:
            raise ValueError('budget_contract_requires_6_plus_3')
        self.limit, self.exemption_limit = limit, exemptions
        self.charged = self.actual = self.exempted = 0
        self.receipts = []

    @property
    def remaining(self):
        return self.limit - self.charged

    def settle(self, action, output, evidence=None, browse_state=None):
        if self.remaining <= 0 or self.actual >= 9:
            raise ValueError('tool_budget_exhausted')
        environmental = environment_failure(action, output, evidence)
        exempt = environmental and self.exempted < self.exemption_limit
        if action == 'browse' and exempt and (browse_state is None or browse_state.total < 1):
            raise ValueError('browse_begin_required_before_settlement')
        self.actual += 1
        self.exempted += int(exempt)
        self.charged += int(not exempt)
        if action == 'browse' and exempt:
            browse_state.total -= 1
        receipt = dict(action=action, environment_failure=environmental,
                       exempted=exempt, charged=not exempt, **self.view())
        self.receipts.append(receipt)
        return receipt

    def view(self):
        return dict(actual_tool_calls=self.actual, charged_tool_calls=self.charged,
                    remaining_tool_calls=self.remaining,
                    environment_failure_exemptions_used=self.exempted,
                    environment_failure_exemptions_remaining=3-self.exempted,
                    max_actual_tool_calls=9)


def install_browse_state(cls):
    """Preserve quota, cooldown and success-query guards; cap failed retries."""
    if getattr(cls, '_environment_budget_v1', False):
        return
    original_begin, original_observe = cls.begin, cls.observe
    original_check, original_view = cls.check, cls.view

    def begin(self, turn):
        original_begin(self, turn)
        self.actual_browse_calls = getattr(self, 'actual_browse_calls', 0) + 1

    def observe(self, route, sid, output, evidence, turn, query):
        original_observe(self, route, sid, output, evidence, turn, query)
        if evidence.get('failed'):
            counts = getattr(self, '_failure_attempts_v1', {})
            key = (route, sid)
            counts[key] = counts.get(key, 0) + 1
            self._failure_attempts_v1 = counts
            if output.get('retryable') is False or counts[key] >= 2:
                self.records[key] = dict(reason=output.get('error_code') or output.get('failure_type') or 'acquisition_failure',
                    failures=counts[key], retryable=False, retry_after_turn=None,
                    scope='current_trajectory')

    def check(self, route, sid, turn, query):
        return original_check(self, route, sid, turn, query)

    def view(self, *args, **kwargs):
        result = original_view(self, *args, **kwargs)
        result['browse_charged_total'] = self.total
        result['browse_executed_total'] = getattr(self, 'actual_browse_calls', 0)
        result['guidance'] += ' Up to three environmental failures per trajectory are exempt from both call and Browse quotas; later failures consume normal quotas. A failed source can be retried at most once; nonretryable failures cannot be retried.'
        return result

    cls.begin, cls.observe, cls.check, cls.view = begin, observe, check, view
    cls._environment_budget_v1 = True
