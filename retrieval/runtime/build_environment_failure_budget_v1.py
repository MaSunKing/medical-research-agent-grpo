"""Derive an isolated runtime without changing frozen snapshots."""
import ast
import hashlib
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def accessible(path):
    path = Path(path).resolve()
    if os.name == 'nt' and not str(path).startswith('\\\\?\\'):
        return Path('\\\\?\\' + str(path))
    return path


def patch_runner(source):
    replacements = [
        ('"--max-decision-turns", type=int, default=8)', '"--max-decision-turns", type=int, default=11)'),
        ('        remaining = args.max_tool_calls', '        budget = ToolBudget(args.max_tool_calls)\n        remaining = budget.remaining'),
        ('        browse_failures = BrowseState()', '        install_browse_state(BrowseState)\n        browse_failures = BrowseState()'),
        ('            remaining -= 1', '            # Settle quotas only after the real tool receipt.'),
        ('                additions = [', '                budget.settle("search", search_output)\n                remaining = budget.remaining\n                additions = ['),
        ('            evidence = collection.evidence_payload(browse_output)', '            evidence = collection.evidence_payload(browse_output)\n            budget.settle("browse", browse_output, evidence, browse_failures)\n            remaining = budget.remaining'),
        ('"formal_tool_calls": args.max_tool_calls - remaining,', '"formal_tool_calls": budget.actual, "charged_tool_calls": budget.charged, "tool_budget": budget.view(), "tool_budget_receipts": budget.receipts,'),
        ("runtime_feedback['browse_state'] = browse_failures.view", "runtime_feedback['tool_budget'] = budget.view()\n            runtime_feedback['browse_state'] = browse_failures.view"),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('runtime_patch_anchor_mismatch: ' + old)
        source = source.replace(old, new)
    tree = ast.parse(source)
    future_imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == '__future__']
    insertion = max((node.end_lineno for node in future_imports), default=0)
    if not insertion and tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
        insertion = tree.body[0].end_lineno
    lines = source.splitlines(keepends=True)
    lines.insert(insertion, 'from environment_failure_budget_v1 import ToolBudget, install_browse_state\n')
    source = ''.join(lines)
    compile(source, '<patched-runtime>', 'exec')
    return source


def build(base):
    base = accessible(base)
    module = (HERE/'environment_failure_budget_v1.py').read_text(encoding='utf-8')
    changes = {}
    for arm in ('title_strict', 'preview_strict', 'preview_relaxed'):
        path = f'variants/{arm}/script/run_medgap_v71_on_policy_retrieval.py'
        if not (base/path).is_file():
            continue
        changes[path] = patch_runner((base/path).read_text(encoding='utf-8'))
        changes[f'variants/{arm}/script/environment_failure_budget_v1.py'] = module
        prompt_path = f'variants/{arm}/script/collect_medgap_v71_decision_groups.py'
        prompt = (base/prompt_path).read_text(encoding='utf-8')
        old = 'each executed call consumes one of the six tool calls, including cache hits.'
        new = ('successful calls, empty results and cache hits consume the six charged tool calls. '
               'The first three structured environmental failures across Search and Browse are exempt '
               'from both total-call and Browse quotas; later environmental failures consume normal quotas '
               'and do not force an immediate Stop. At most nine actual calls can execute. '
               'Blocked or invalid actions are not executed. A failed source can be retried at most once '
               'after cooldown; nonretryable failures cannot be retried.')
        if prompt.count(old) > 1:
            raise ValueError('budget_prompt_anchor_mismatch')
        changes[prompt_path] = prompt.replace(old, new)
        ast.parse(changes[prompt_path])
    if not changes:
        raise ValueError('no_runtime_variants_found')
    digest = hashlib.sha256((str(base.resolve()) + ''.join(changes.values())).encode()).hexdigest()[:12]
    dest = accessible(HERE/('inference_environment_budget_v1_' + digest))
    if not dest.exists():
        shutil.copytree(base, dest, ignore=shutil.ignore_patterns('__pycache__'))
        for path, source in changes.items():
            (dest/path).write_text(source, encoding='utf-8')
    for path, source in changes.items():
        if (dest/path).read_text(encoding='utf-8') != source:
            raise ValueError('derived_runtime_integrity_failed: ' + path)
    return dest
