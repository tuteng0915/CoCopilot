# DreamOn Token Inpaint MBPP Case Study

## Scope

- Eval result:
  `outputs/inpaint/deepseek_mbpp_dreamon_inpaint_token_eval-sanitized_eval_results.json`
- Analysis file:
  `outputs/inpaint/deepseek_mbpp_dreamon_inpaint_token_analysis.jsonl`
- Baseline:
  `outputs/deepseek_mbpp-sanitized_eval_results.json`
- Prior line-based inpaint:
  `outputs/inpaint/deepseek_mbpp_dreamon_inpaint_eval-sanitized_eval_results.json`

## Executive Summary

This run is **not a successful token-level inpainting experiment**.

The evaluation file looks strong only because it exactly matches the original DeepSeek baseline:

| run | base pass | plus pass | both pass |
| --- | ---: | ---: | ---: |
| DeepSeek baseline | 283 | 246 | 245 |
| DreamOn line inpaint | 276 | 241 | 238 |
| DreamOn token inpaint | 283 | 246 | 245 |

The reason is visible in the analysis file:

- `378 / 378` samples have `status = "error"`
- `378 / 378` samples have empty `inpainted_block`
- `378 / 378` samples fall back to the original `solution`

So the token-level run produced **baseline-equivalent eval numbers by complete fallback**, not by successful infilling.

## What happened

### 1. Token-level masking was applied

The analysis file shows the new selector did run:

- `selector = "ppl_token"`
- mean masked tokens: `22.458`
- median masked tokens: `20`
- min / max masked tokens: `8 / 94`
- mean text tokens per sample: `112.23`
- median text tokens per sample: `99.5`
- mean number of masked spans: `14.63`
- median number of masked spans: `13`
- max masked spans in one sample: `54`

This means the run is masking roughly the highest-perplexity `20%` of tokens, but those tokens are usually scattered across many short spans.

### 2. Every sample failed during generation

All recorded errors have the same form:

```text
infill_error(ValueError): Input length of input_ids is X, but max_length is set to Y
```

There are many concrete variants because the input lengths differ by sample, but the failure mode is the same.

Examples from the analysis file:

- `Mbpp/100`: `input_ids=86`, `max_length=70`
- `Mbpp/404`: `input_ids=66`, `max_length=54`
- `Mbpp/752`: `input_ids=161`, `max_length=130`
- `Mbpp/108`: `input_ids=277`, `max_length=223`

### 3. Why this happened

The run configuration for token mode in the script is:

- `initial_mask_tokens = gt_tok_len`
- `max_new_tokens = 1`
- `expand_budget = 0`

See:

- `scripts/inpaint_from_outputs.py:518`
- `scripts/inpaint_from_outputs.py:521`
- `scripts/inpaint_from_outputs.py:522`

At the same time, `DreamOnCoder.infill_masked_tokens()` builds a full masked sequence containing the entire original token sequence plus BOS/EOS, then sends that full sequence into diffusion generation:

- `src/coder/models/dreamon_coder.py:379`
- `src/coder/models/dreamon_coder.py:387`
- `src/coder/models/dreamon_coder.py:407`

So the generated `max_length` budget is too small for the actual masked input sequence length, and the model rejects it before generation starts.

## Evaluation Interpretation

### Against baseline

Compared with the original DeepSeek run:

- base: `283 -> 283`
- plus: `246 -> 246`
- both: `245 -> 245`

Transition counts:

- base: `283 pass->pass`, `95 fail->fail`
- plus: `246 pass->pass`, `132 fail->fail`

There are:

- `0` improvements
- `0` regressions

This is exactly what we expect from a full fallback run.

### Against the earlier line-based inpaint run

Compared with the earlier line-based inpaint result:

- base: token run is better on `18` tasks and worse on `11`
- plus: token run is better on `13` tasks and worse on `8`

But this does **not** mean token inpainting worked better.
It only means:

- line-based inpainting changed many solutions
- token-based run kept the original baseline solutions because it failed and fell back

## Error Pattern Summary

The error strings are not identical textually because the lengths vary, but semantically they are the same `max_length` validation error.

Representative examples:

- `Input length 55 > max_length 45`
- `Input length 66 > max_length 54`
- `Input length 86 > max_length 70`
- `Input length 113 > max_length 92`
- `Input length 277 > max_length 223`

The larger the original sample, the larger the same mismatch becomes.

## Five Representative Cases

## Case 1: `Mbpp/105`

- status: `error`
- masked tokens: `11 / 53`
- masked spans: `8`
- error:
  `Input length of input_ids is 55, but max_length is set to 45`

Ground truth masked content:

```text
"""
Write
<SPAN_SEP>
 python
<SPAN_SEP>
 count true bo
<SPAN_SEP>
 given
<SPAN_SEP>
assert
<SPAN_SEP>
,False
<SPAN_SEP>
"""

<SPAN_SEP>
(lst
```

Observation:

- Even a short sample fails before generation.
- The masked tokens are already fragmented across docstring, assertion, and function signature.

## Case 2: `Mbpp/100`

- status: `error`
- masked tokens: `17 / 84`
- masked spans: `10`
- error:
  `Input length of input_ids is 86, but max_length is set to 70`

Ground truth masked content begins with:

```text
"""
Write
<SPAN_SEP>
 find
<SPAN_SEP>
 smallest palindrome of
```

Observation:

- The highest-PPL tokens are spread across both prompt-like docstring text and executable code punctuation.
- Because generation failed, `inpainted_solution == original solution`.

## Case 3: `Mbpp/404`

- status: `error`
- masked tokens: `13 / 64`
- masked spans: `8`
- error:
  `Input length of input_ids is 66, but max_length is set to 54`

Ground truth masked content includes:

```text
"""
Write
<SPAN_SEP>
 python
<SPAN_SEP>
 find
<SPAN_SEP>
 numbers.
assert minimum
```

Observation:

- This sample shows token masking often breaks semantic units into small pieces rather than one coherent block.
- The run still falls back, so evaluation remains unchanged.

## Case 4: `Mbpp/752`

- status: `error`
- masked tokens: `32 / 159`
- masked spans: `18`
- error:
  `Input length of input_ids is 161, but max_length is set to 130`

Ground truth masked content includes:

```text
"""
Write
<SPAN_SEP>
 find
<SPAN_SEP>
 nth j
<SPAN_SEP>
st
<SPAN_SEP>
 number.
```

Observation:

- This is a medium-large program where token masking chops identifiers and docstring text into pieces like `nth j`, `st`, `_num(5`.
- Even before quality is evaluated, generation aborts.

## Case 5: `Mbpp/108`

- status: `error`
- masked tokens: `55 / 275`
- masked spans: `26`
- error:
  `Input length of input_ids is 277, but max_length is set to 223`

Ground truth masked content contains many tiny fragments such as:

```text
2
<SPAN_SEP>
1
<SPAN_SEP>
4
<SPAN_SEP>
5
<SPAN_SEP>
9
<SPAN_SEP>
],[19
```

Observation:

- On long, punctuation-heavy MBPP samples, token-level top-PPL masking becomes extremely fragmented.
- This case illustrates the operational difficulty of using dispersed token masks with a generation path tuned for contiguous infill-like behavior.

## Conclusion

This result file should be read as a **failed run**, not as a successful token-level inpainting benchmark.

The actual conclusions are:

- token-level top-20%-PPL masking was computed and recorded
- the masked positions are usually highly fragmented
- generation failed on all `378` samples with `max_length` validation errors
- evaluation matched baseline only because every sample fell back to the original solution

## Practical takeaway

Before re-running the experiment, the token-mode generation path needs to be fixed so that the diffusion call gets a valid length budget for the full masked sequence.
