"""Bounded, validated original-ID selection shared by distinct compression roles."""
import copy
import json
from ..storage import canonical, digest


def text_tokens(codec, text):
    tokenizer = getattr(codec, 'tokenizer', None)
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(codec.encode([{'role':'user','content':text}]))


class CompressionBudget:
    def __init__(self, max_calls=12):
        if type(max_calls) is not int or not 1 <= max_calls <= 64:
            raise ValueError('invalid_compression_call_limit')
        self.max_calls, self.calls = max_calls, 0

    @property
    def exhausted(self):
        return self.calls >= self.max_calls


class SelectionRunner:
    def __init__(self, provider, codec, store, run_id, limit, budget=None):
        self.provider, self.codec, self.store, self.run_id, self.limit = provider, codec, store, run_id, limit
        self.budget = budget if budget is not None else CompressionBudget()

    def fits(self, messages, reserve):
        return len(self.codec.encode(messages)) + reserve + 128 <= self.limit

    async def ask(self, messages, reserve, validate, kind):
        base = copy.deepcopy(messages)
        error = ''
        for attempt in range(2):
            current = copy.deepcopy(base)
            if attempt:
                current.append({'role':'user','content':'Selection rejected: '+error+
                                ' Return corrected IDs within the supplied count and total text-token budget. Do not rewrite text.'})
            if not self.fits(current, reserve):
                self.store.event(self.run_id, kind, dict(accepted=False,error='compression_repair_or_input_overflow',attempt=attempt))
                return None
            request = dict(stage='compression',role='compression',messages=current,
                           prompt_token_ids=self.codec.encode(current),exact_tokenizer=self.codec.exact,max_new_tokens=reserve)
            key = digest([self.run_id,'bounded_original_selection_v2',self.provider.identity,request])
            if self.budget.exhausted:
                self.store.event(self.run_id,kind,dict(accepted=False,error='compression_call_limit',calls=self.budget.calls))
                return None
            async def invoke():
                self.budget.calls += 1
                return await self.provider.generate(request)
            response = await self.store.once(key,invoke)
            try:
                value = validate(json.loads(response['text']))
                error = ''
            except (ValueError,TypeError,KeyError) as exc:
                value,error=None,str(exc)[:800]
            self.store.event(self.run_id,kind,dict(receipt_key=key,request=request,response=response,
                             accepted=not error,error=error,attempt=attempt,calls=self.budget.calls,
                             max_calls=self.budget.max_calls))
            if not error:
                return value
        return None

    async def select(self, parts, make_messages, parse_ids, render, *, target, max_ids, reserve, kind, dependencies=None):
        """Split only at complete original boundaries; merge by another model selection."""
        if not parts:
            return None
        items=list(parts.items())
        batches=[];batch={}
        for name,text in items:
            required = set((dependencies or {}).get(name, [])) | {name}
            if not required <= set(parts):
                self.store.event(self.run_id,kind,dict(accepted=False,error='missing_original_dependency_in_input',passage_id=name))
                return None
            unit = {n:t for n,t in items if n in required}
            trial={**batch,**unit}
            if self.fits(make_messages(trial),reserve):
                batch=trial
                continue
            if batch:
                batches.append(batch);batch={}
            if not self.fits(make_messages(unit),reserve):
                # Do not silently omit an indivisible original paragraph.
                self.store.event(self.run_id,kind,dict(accepted=False,error='complete_passage_exceeds_compressor_input',passage_id=name))
                return None
            batch=unit
        if batch:batches.append(batch)

        async def choose(visible):
            def validate(obj):
                names=parse_ids(obj)
                if (not isinstance(names,list) or any(not isinstance(n,str) or n not in visible for n in names)
                        or len(names)!=len(set(names))):
                    raise ValueError('invalid_original_passage_ids')
                if len(names)>max_ids:
                    raise ValueError(f'too_many_passages: selected={len(names)} maximum={max_ids}')
                ordered=[n for n in parts if n in names]
                cost=text_tokens(self.codec,render(ordered)) if ordered else 0
                if cost>target:
                    raise ValueError(f'selected_text_exceeds_budget: selected_tokens={cost} maximum={target}')
                return ordered
            return await self.ask(make_messages(visible),reserve,validate,kind)

        selected=[]
        for batch in batches:
            names=await choose(batch)
            if names is None:return None
            selected.extend(n for n in names if n not in selected)
        selected=[n for n in parts if n in selected]
        if len(selected)<=max_ids and (not selected or text_tokens(self.codec,render(selected))<=target):
            return selected
        pool={n:parts[n] for n in selected}
        if not self.fits(make_messages(pool),reserve):
            self.store.event(self.run_id,kind,dict(accepted=False,error='selected_pool_exceeds_compressor_input'))
            return None
        return await choose(pool)


def excerpt_target(codec, text, *, available, deficit, round_index=0, current_text=None, dependency_aware=False):
    """Distribute deficits without targets smaller than every complete passage."""
    from ..compression import spans, passage_dependencies, card
    cost=text_tokens(codec,text if current_text is None else current_text)
    lengths=[text_tokens(codec,text[a:b]) for a,b in spans(text)]
    if dependency_aware:
        chunk = dict(text=text)
        deps = passage_dependencies(text)
        lengths = [text_tokens(codec,card(chunk, list(dict.fromkeys(deps.get(f'S{i}', [])+[f'S{i}'])))['text'])
                   for i,_ in enumerate(spans(text)) if len(set(deps.get(f'S{i}', [])+[f'S{i}'])) <= 3]
    if not lengths or available<min(lengths) or min(lengths)>=cost:return None
    # Begin gently; other chunks can share the deficit. Later rounds tighten.
    desired=int(max(1,cost-min(deficit+64,int(cost*.35)))*(.85**round_index))
    return min(available,max(min(lengths),desired))
