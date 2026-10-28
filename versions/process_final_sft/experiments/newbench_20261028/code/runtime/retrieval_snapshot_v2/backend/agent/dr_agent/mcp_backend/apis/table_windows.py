"""Deterministic, exact-source table row windows; never rewrite a cell.

Only parser-owned layouts are accepted. Flattened legacy/PDF tables without
verified row/header ownership retain the whole-table exclusion policy.
"""
import copy
import hashlib
import math
import re


VERSION = 'exact_table_rows_v1'


def _clean(value):
    return re.sub(r'\s+', ' ', str(value or '')).strip()


def table_section(rows, *, heading='', origin='', prefix='', footer='', header_count=0,
                  unresolved_notes=False):
    """Serialize complete original cells, retaining their exact normalized spans."""
    parts, indexed, cursor, active, widths = [], [], 0, {}, []
    reason = ''
    def append(text):
        nonlocal cursor
        text = _clean(text)
        if not text:
            return None
        if parts:
            cursor += 1
        span = dict(start_char=cursor, end_char=cursor+len(text))
        parts.append(text)
        cursor += len(text)
        return span
    caption_span = append(prefix)
    for index, row in enumerate(rows):
        cells, column = [], 0
        occupied = {c for c, end in active.items() if end >= index}
        for original in row:
            while column in occupied:
                column += 1
            try:
                colspan, rowspan = int(original.get('colspan', 1)), int(original.get('rowspan', 1))
            except (ValueError, TypeError):
                colspan = rowspan = 0
            if colspan < 1 or colspan > 256 or rowspan < 1 or index+rowspan > len(rows):
                reason = 'unverified_cell_span'
                colspan = rowspan = 1
            if any(c in occupied for c in range(column, column+colspan)):
                reason = 'overlapping_cell_span'
            cells.append(dict(text=_clean(original.get('text')), column=column,
                              colspan=colspan, rowspan=rowspan,
                              header=bool(original.get('header'))))
            for c in range(column, column+colspan):
                occupied.add(c)
                active[c] = index+rowspan-1
            column += colspan
        span = append(' | '.join(cell['text'] for cell in cells))
        if not span:
            reason = 'empty_table_row'
            span = dict(start_char=cursor, end_char=cursor)
        widths.append(max(occupied)+1 if occupied else 0)
        indexed.append(dict(row_index=index, cells=cells, **span))
    footer_span = append(footer)
    text = ' '.join(parts)
    if not 0 < header_count < len(indexed):
        reason = 'unverified_table_header'
    if not widths or len(set(widths)) != 1 or not widths[0]:
        reason = 'unverified_column_ownership'
    if any(i < header_count and i+c['rowspan'] > header_count
           for i, row in enumerate(indexed) for c in row['cells']):
        reason = 'header_data_rowspan_unsupported'
    if unresolved_notes:
        reason = 'unresolved_table_footnote'
    groups, index = [], header_count
    while index < len(indexed):
        end, scan = index+1, index
        while scan < min(end, len(indexed)):
            end = max([end]+[scan+c['rowspan'] for c in indexed[scan]['cells']])
            scan += 1
        groups.append(list(range(index, min(end, len(indexed)))))
        index = end
    layout = dict(version=VERSION, safe=not reason, reason=reason,
                  source_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                  header_count=header_count, column_count=widths[0] if widths else 0,
                  rows=indexed, row_groups=groups, caption_span=caption_span,
                  footer_span=footer_span, all_footnotes_retained=True)
    return dict(heading=_clean(heading), text=text, structure_kind='table',
                structure_origin=origin, table_structure_complete=not reason,
                table_layout=layout)


def _layout(chunk, document):
    section = document['sections'][chunk['section_index']]
    text = _clean(section['text'])
    layout = section.get('table_layout') or {}
    if (layout.get('version') != VERSION or not layout.get('safe')
            or hashlib.sha256(text.encode()).hexdigest() != layout.get('source_text_sha256')):
        return None, text
    for row in layout['rows']:
        expected = ' | '.join(c['text'] for c in row['cells'])
        if text[row['start_char']:row['end_char']] != expected:
            return None, text
    return layout, text


def _previous_rows(chunk, layout, source, opened):
    """Same-source and same-table row identities; never fuzzy/clinical dedup."""
    table_id = chunk.get('canonical_chunk_id') or chunk['chunk_id']
    used = set()
    for prior in opened:
        if prior.get('source_id') != source:
            continue
        projection = prior.get('table_window') or prior.get('metadata', {}).get('table_window') or {}
        if (projection.get('canonical_table_id') == table_id
                and projection.get('source_text_sha256') == layout['source_text_sha256']):
            used.update(projection.get('row_indices', ()))
        elif (str(prior.get('chunk_id','')).startswith(table_id+'-v')
                and hashlib.sha256(_clean(prior.get('text')).encode()).hexdigest() == layout['source_text_sha256']):
            used.update(range(layout['header_count'],len(layout['rows'])))
    return used


def row_window(chunk, document, *, tokenizer, max_tokens, max_chars, query='', opened=(), grow=True):
    """Return one complete contiguous data-row group, with exact header/notes.

    Grow adjacent groups only when the whole framed window fits. Header and
    notes are deliberately repeated to avoid detached numeric/comparator cells.
    """
    layout, original = _layout(chunk, document)
    if layout is None:
        return None, 'unverified_table_layout'
    source = str(chunk.get('canonical_chunk_id') or chunk['chunk_id']).split('#', 1)[0]
    used = _previous_rows(chunk, layout, source, opened)
    groups = [g for g in layout['row_groups'] if not any(i in used for i in g)]
    if not groups:
        return None, 'already_opened_table_rows'
    terms = set(re.findall(r'[\w]+', query.casefold()))
    # Lexical focus is only a row-window anchor inside the already ranked table,
    # not a replacement for chunk/source ranking or a semantic coverage label.
    group_terms = [set(re.findall(r'[\w]+', ' '.join(
        original[layout['rows'][i]['start_char']:layout['rows'][i]['end_char']] for i in g).casefold())) for g in groups]
    term_weights = {t: math.log(1+len(groups)/(1+sum(t in words for words in group_terms))) for t in terms}
    scores = [sum(term_weights[t] for t in terms & words) for words in group_terms]
    def score(n):
        return scores[n]
    order = sorted(range(len(groups)), key=lambda n: (-score(n), groups[n][0]))
    count = lambda value: len(tokenizer.encode(value, add_special_tokens=False)) if tokenizer else len(value)
    def frame(indices):
        spans = []
        if layout.get('caption_span'):
            spans.append(dict(layout['caption_span'], role='caption'))
        spans.extend(dict(start_char=r['start_char'], end_char=r['end_char'], role='header', row_index=r['row_index'])
                     for r in layout['rows'][:layout['header_count']])
        spans.extend(dict(start_char=layout['rows'][i]['start_char'], end_char=layout['rows'][i]['end_char'], role='data', row_index=i)
                     for i in sorted(indices))
        if layout.get('footer_span'):
            spans.append(dict(layout['footer_span'], role='footnotes'))
        value = ' '.join(original[s['start_char']:s['end_char']] for s in spans)
        return value, spans
    chosen = None
    for anchor in order:
        indices = list(groups[anchor])
        value, spans = frame(indices)
        if len(value) > max_chars or count(value) > max_tokens:
            continue
        # Adjacent rows only: do not stitch unrelated rows or omit a rowspan.
        changed = grow
        while changed:
            changed = False
            adjacent = [g for g in groups if g[-1]+1 == min(indices) or g[0]-1 == max(indices)]
            adjacent.sort(key=lambda g: (-score(groups.index(g)), g[0]))
            for group in adjacent:
                proposal, candidate_spans = frame(sorted(indices+group))
                if len(proposal) <= max_chars and count(proposal) <= max_tokens:
                    indices = sorted(indices+group)
                    value, spans = proposal, candidate_spans
                    changed = True
                    break
        chosen = value, spans, indices
        break
    if chosen is None:
        return None, 'complete_row_group_exceeds_budget'
    value, spans, indices = chosen
    for span in spans:
        span['text_sha256'] = hashlib.sha256(original[span['start_char']:span['end_char']].encode()).hexdigest()
    result = copy.deepcopy(chunk)
    projection = dict(version=VERSION, canonical_table_id=chunk.get('canonical_chunk_id') or chunk['chunk_id'],
                      source_text_sha256=layout['source_text_sha256'], row_indices=indices,
                      header_indices=list(range(layout['header_count'])), column_count=layout['column_count'],
                      source_spans=spans, separator=' ', complete_row_groups=True,
                      all_columns_retained=True, all_footnotes_retained=True,
                      complete_table=len(indices) == len(layout['rows'])-layout['header_count'],
                      selection='lexical_focus_contiguous_row_groups')
    result.update(text=value, start_char=min(s['start_char'] for s in spans),
                  end_char=max(s['end_char'] for s in spans), table_window=projection,
                  boundary_incomplete=False, structure_kind='table_like', table_integrity_verified=True,
                  original_chunk_text_sha256=chunk.get('text_sha256'),
                  text_sha256=hashlib.sha256(value.encode()).hexdigest(), boundary_policy=VERSION,
                  structure_audit=dict(classifier=VERSION, structure_kind='table_like', complete_rows=True))
    return result, 'kept_table_row_window'


def validate_window(chunk, document):
    layout, original = _layout(chunk, document)
    p = chunk.get('table_window') or {}
    if layout is None or p.get('version') != VERSION:
        raise ValueError('unverified_table_window')
    spans = p['source_spans']
    indices = p.get('row_indices', [])
    if (not indices or any(type(i) is not int for i in indices)
            or min(indices) < layout['header_count'] or max(indices) >= len(layout['rows'])
            or indices != list(range(min(indices), max(indices)+1))):
        raise ValueError('table_rows_not_contiguous')
    if (p.get('header_indices') != list(range(layout['header_count']))
            or p.get('canonical_table_id') != (chunk.get('canonical_chunk_id') or chunk['chunk_id'])
            or p.get('column_count') != layout['column_count']
            or p.get('source_text_sha256') != layout['source_text_sha256']
            or p.get('complete_row_groups') is not True
            or p.get('complete_table') is not (len(indices) == len(layout['rows'])-layout['header_count'])
            or not p.get('all_columns_retained') or not p.get('all_footnotes_retained')):
        raise ValueError('table_context_ownership_mismatch')
    if any(set(group) & set(indices) and not set(group) <= set(indices) for group in layout['row_groups']):
        raise ValueError('incomplete_table_rowspan_group')
    expected = []
    if layout.get('caption_span'):
        expected.append(dict(layout['caption_span'],role='caption'))
    expected.extend(dict(start_char=r['start_char'],end_char=r['end_char'],role='header',row_index=r['row_index'])
                    for r in layout['rows'][:layout['header_count']])
    expected.extend(dict(start_char=layout['rows'][i]['start_char'],end_char=layout['rows'][i]['end_char'],role='data',row_index=i)
                    for i in indices)
    if layout.get('footer_span'):
        expected.append(dict(layout['footer_span'],role='footnotes'))
    if [{k:v for k,v in s.items() if k != 'text_sha256'} for s in spans] != expected:
        raise ValueError('table_window_framing_mismatch')
    for s in spans:
        if not 0 <= s['start_char'] < s['end_char'] <= len(original):
            raise ValueError('invalid_table_source_span')
        if hashlib.sha256(original[s['start_char']:s['end_char']].encode()).hexdigest() != s['text_sha256']:
            raise ValueError('table_source_span_hash_mismatch')
    if ' '.join(original[s['start_char']:s['end_char']] for s in spans) != chunk['text']:
        raise ValueError('table_projection_text_mismatch')
    return copy.deepcopy(chunk)
