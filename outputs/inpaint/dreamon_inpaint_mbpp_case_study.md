# DreamOn Inpaint MBPP Case Study

## Scope

- Source analysis file: `outputs/inpaint/deepseek_mbpp_dreamon_inpaint_analysis.jsonl`
- Source eval summaries:
  - `outputs/deepseek_mbpp_summary.json`
  - `outputs/inpaint/dreamon_inpaint_mbpp_summary.json`
- Random sample seed for case display: `20260302`

## How the PPL-based mask position is computed

### 1. Token-level NLL / proxy PPL

The code first computes token-level negative log-likelihood in `DreamOnCoder.score_token_nll()`:

- File: `src/coder/models/dreamon_coder.py:108`
- The text is tokenized, with `offset_mapping` requested when available.
- The model runs a forward pass on the full text.
- Cross-entropy is computed between shifted logits and shifted labels.
- The returned `token_nll` is aligned back to text token positions.
- `proxy_ppl` is then computed as `exp(mean(valid_token_nll))`.

Key implementation points:

- `shift_logits` / `shift_labels` are built at `src/coder/models/dreamon_coder.py:173`
- per-token NLL is computed with `F.cross_entropy(..., reduction="none")` at `src/coder/models/dreamon_coder.py:175`
- full-text proxy PPL is computed at `src/coder/models/dreamon_coder.py:193`

### 2. Map token NLL to lines

The line selection happens in `select_mask_start_by_proxy_ppl()`:

- File: `scripts/inpaint_from_outputs.py:135`
- The solution is split into lines.
- The number of masked lines is computed as:

```text
mask_lines = max(1, round(n_lines * ratio))
```

- Here `ratio = 0.2`, so this is a fixed ratio, not a fixed line count.
- Token NLL is mapped to lines using `offset_mapping` when possible, otherwise a token-count fallback is used.

### 3. Slide a window over all possible contiguous line spans

For each candidate span:

- `start` iterates over all valid starting lines.
- `end = start + mask_lines`
- The code sums the token NLL inside that line window.
- With the current config, `ppl_reduction = "mean"`, so the final window score is:

```text
window_score = window_sum / window_tokens
```

- The span with the highest score is selected.
- Ties keep the smaller start line.

Key implementation points:

- mask line count: `scripts/inpaint_from_outputs.py:146`
- sliding window loop: `scripts/inpaint_from_outputs.py:175`
- mean-vs-sum reduction: `scripts/inpaint_from_outputs.py:181`
- selected start/end returned at `scripts/inpaint_from_outputs.py:207`

### 4. Apply the selected mask and run infilling

In the main loop:

- `mask_selector == "ppl"` triggers `select_mask_start_by_proxy_ppl()`
- the chosen `start` is passed into `build_mask_at_start()`
- DreamOn infills the middle block with `prefix + [MASK] + suffix`

Key implementation points:

- selector dispatch: `scripts/inpaint_from_outputs.py:358`
- build final mask: `scripts/inpaint_from_outputs.py:367`
- infill call: `scripts/inpaint_from_outputs.py:402`

## What is usually masked

Across all `378` MBPP samples:

- All `378/378` used `selector = ppl`
- Mean original solution length: `11.939` lines
- Median original solution length: `10` lines
- Mean masked span length: `2.46` lines
- Low-token-confidence PPL windows: `286/378`

Masked line-count distribution:

| num_lines | count |
| --- | ---: |
| 1 | 4 |
| 2 | 254 |
| 3 | 83 |
| 4 | 24 |
| 5 | 7 |
| 6 | 4 |
| 7 | 2 |

Most common start lines:

| start_line | count |
| --- | ---: |
| 4 | 295 |
| 1 | 60 |
| 3 | 10 |
| 5 | 6 |
| 8 | 3 |

Interpretation:

- The mask span is usually `2` lines.
- The selected start position is heavily concentrated at line `4`.
- In practice, many masks hit the docstring closing boundary or the top of the function.

## Overall effect on MBPP

Baseline DeepSeek summary:

- base pass: `283/378`
- plus pass: `246/378`
- both pass: `245/378`

DreamOn inpaint summary:

- base pass: `276/378`
- plus pass: `241/378`
- both pass: `238/378`

Delta after inpainting:

- base: `-7`
- plus: `-5`
- both: `-7`

So this run did perform infilling, but the final evaluation is slightly worse than the original DeepSeek MBPP result.

## Five randomly sampled cases

The following cases were sampled with `random.Random(20260302)`.

### Case 1: `Mbpp/108`

- Mask: lines `4-5`, `2` lines
- Selector score: `4.78125`
- Low-token-confidence: `True`
- Eval before: `base=pass`, `plus=pass`
- Eval after: `base=pass`, `plus=pass`

Ground truth block:

```python
"""

```

Inpainted block:

```python
"""

```

Observation:

- This is a trivial exact recovery.
- The masked content is only the docstring closing boundary plus a blank line.

### Case 2: `Mbpp/113`

- Mask: lines `4-5`, `2` lines
- Selector score: `5.46875`
- Low-token-confidence: `True`
- Eval before: `base=pass`, `plus=fail`
- Eval after: `base=pass`, `plus=fail`

Ground truth block:

```python
"""

```

Inpainted block:

```python
assert check_integer("123")==True
"""
```

Observation:

- The model hallucinated an extra assertion inside the docstring region.
- The code still passes base tests, but this is not a faithful restoration.

### Case 3: `Mbpp/404`

- Mask: lines `1-3`, `3` lines
- Selector score: `1.444894790649414`
- Low-token-confidence: `False`
- Eval before: `base=pass`, `plus=pass`
- Eval after: `base=pass`, `plus=pass`

Ground truth block:

```python
"""
Write a python function to find the minimum of two numbers.
assert minimum(1,2) == 1
```

Inpainted block:

```python

```

Observation:

- The model returned an empty infill.
- The function body remained intact, so evaluation still passed.
- This shows that some front-of-file losses do not affect MBPP execution results.

### Case 4: `Mbpp/745`

- Mask: lines `4-6`, `3` lines
- Selector score: `1.410340049050071`
- Low-token-confidence: `False`
- Eval before: `base=fail`, `plus=fail`
- Eval after: `base=fail`, `plus=fail`

Ground truth block:

```python
"""

def divisible_by_digits(startnum, endnum):
```

Inpainted block:

```python
def divisible_by_digits(startnum, endnum):
    """
    Finds numbers in the given range that are divisible by every digit they contain.
    """
```

Observation:

- The model rewrote the function header area instead of restoring the original docstring boundary.
- The result stays incorrect on evaluation.

### Case 5: `Mbpp/88`

- Mask: lines `4-6`, `3` lines
- Selector score: `1.4345296223958333`
- Low-token-confidence: `True`
- Eval before: `base=pass`, `plus=pass`
- Eval after: `base=pass`, `plus=pass`

Ground truth block:

```python
"""

def freq_count(lst):
```

Inpainted block:

```python
def freq_count(lst):
    """
    Returns a dictionary with the frequency of each element in the list.
    """
```

Observation:

- The model inserted a new function docstring and dropped the original exact boundary.
- The implementation below remained good enough for tests, so the task still passed.

## Short conclusion

- The PPL position is not chosen randomly in this run.
- The code chooses the contiguous line span with the highest mean token NLL under a `20%` line-ratio budget.
- The span is usually `2` lines, and most often starts around line `4`.
- Many chosen windows are very small and low-confidence, especially around docstring boundaries.
- Infilling clearly ran, but on MBPP this setting slightly underperformed the original baseline.
