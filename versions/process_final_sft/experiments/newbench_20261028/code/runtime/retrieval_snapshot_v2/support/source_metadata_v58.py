"""Do not expose a derived study-design label. Preserve native source facts."""
def public_source_facts(value):
    if isinstance(value, dict):
        return {k: public_source_facts(v) for k, v in value.items() if k != 'study_design'}
    if isinstance(value, list):
        return [public_source_facts(v) for v in value]
    if isinstance(value, tuple):
        return tuple(public_source_facts(v) for v in value)
    return value
