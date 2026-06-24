# Tau Ablation Re-Run Spec

## Background

The current `tab_tau` table in the paper has two data-quality problems:

1. **τ=0.5 data file is missing.** The existing `remask_kodai/` directory contains only τ∈{0.7, 0.8, 0.9, 0.93, 0.95, 0.97, 0.99}. The 13.4% HumanEval+ figure in the table has no backing file.

2. **Inconsistent evaluation environment.** The `remask_kodai/` experiments were run by a collaborator on a different machine with a different EvalPlus version. Concretely, for 150 tasks whose solution is **identical** between the AR baseline and the τ=0.9 CoCoder output, the AR eval gives 88/150 pass while the CoCoder eval gives 108/150 pass — same code, different result. This inflates the reported gains.

**Goal:** re-run the complete tau ablation from a single, consistent AR input (`base_tuteng/deepseek_humaneval.jsonl` and `base_tuteng/deepseek_mbpp.jsonl`) using the **current** EvalPlus installation, and re-evaluate the AR baseline with the same pipeline.

---

## Working Directory

All commands run from:
```
/home/wjzhang/tt_workspace/model/CoCoder/CoCoder/
```

All outputs go into a new subdirectory `outputs/tau_rerun/`.

---

## Step 0: Verify environment

```bash
export ROOT_DIR=$(git rev-parse --show-toplevel)
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
mkdir -p outputs/tau_rerun

# Verify AR input files exist and have 164/500 tasks
python3 -c "
import json
he = [json.loads(l) for l in open('outputs/base_tuteng/deepseek_humaneval.jsonl')]
mb = [json.loads(l) for l in open('outputs/base_tuteng/deepseek_mbpp.jsonl')]
print(f'HumanEval tasks: {len(he)}')  # expect 164
print(f'MBPP tasks: {len(mb)}')       # expect 500
"
```

---

## Step 1: Re-evaluate AR baseline with current EvalPlus

The AR baseline files were sanitized previously; just re-run eval with the current EvalPlus version.

```bash
# Re-evaluate AR baseline (humaneval)
python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset humaneval \
  --samples outputs/base_tuteng/deepseek_humaneval-sanitized.jsonl \
  --summary_out outputs/tau_rerun/ar_humaneval_summary.json

# Re-evaluate AR baseline (mbpp)
python -m coder.scripts.eval_evalplus \
  --backend local \
  --dataset mbpp \
  --samples outputs/base_tuteng/deepseek_mbpp-sanitized.jsonl \
  --summary_out outputs/tau_rerun/ar_mbpp_summary.json
```

**Expected output:** `outputs/tau_rerun/ar_humaneval_summary.json` with updated pass rates.  
**Record these numbers** — they become the ground-truth AR baseline for the table.

---

## Step 2: Run remask for all τ values

Run on a single GPU (adjust `CUDA_VISIBLE_DEVICES` as needed). Each tau takes ~10–15 min for HumanEval, ~30–40 min for MBPP.

```bash
export CUDA_VISIBLE_DEVICES=0

for THRESH in 0.5 0.7 0.8 0.9 0.93 0.95 0.97 0.99; do

  # HumanEval
  python -m coder.scripts.gen_remask \
    --input outputs/base_tuteng/deepseek_humaneval.jsonl \
    --out   outputs/tau_rerun/remask_humaneval_t${THRESH}.jsonl \
    --confidence_threshold ${THRESH} \
    --temperature 0.0 --top_p 1.0 --seed 3407

  # MBPP
  python -m coder.scripts.gen_remask \
    --input outputs/base_tuteng/deepseek_mbpp.jsonl \
    --out   outputs/tau_rerun/remask_mbpp_t${THRESH}.jsonl \
    --confidence_threshold ${THRESH} \
    --temperature 0.0 --top_p 1.0 --seed 3407

done
```

**Note on τ=0.5:** expect very few tokens to be masked (confidence < 0.5 is rare for most correct tokens). If the output for τ=0.5 is nearly identical to the AR input, that is the real result — the 13.4% figure in the current table is suspect.

---

## Step 3: Sanitize all outputs

```bash
for THRESH in 0.5 0.7 0.8 0.9 0.93 0.95 0.97 0.99; do

  python -m coder.scripts.postprocess_evalplus \
    --dataset humaneval \
    --samples outputs/tau_rerun/remask_humaneval_t${THRESH}.jsonl

  python -m coder.scripts.postprocess_evalplus \
    --dataset mbpp \
    --samples outputs/tau_rerun/remask_mbpp_t${THRESH}.jsonl

done
```

This produces `*-sanitized.jsonl` files next to each output.

---

## Step 4: Evaluate all sanitized outputs

```bash
for THRESH in 0.5 0.7 0.8 0.9 0.93 0.95 0.97 0.99; do

  python -m coder.scripts.eval_evalplus \
    --backend local \
    --dataset humaneval \
    --samples outputs/tau_rerun/remask_humaneval_t${THRESH}-sanitized.jsonl \
    --summary_out outputs/tau_rerun/remask_humaneval_t${THRESH}_summary.json

  python -m coder.scripts.eval_evalplus \
    --backend local \
    --dataset mbpp \
    --samples outputs/tau_rerun/remask_mbpp_t${THRESH}-sanitized.jsonl \
    --summary_out outputs/tau_rerun/remask_mbpp_t${THRESH}_summary.json

done
```

---

## Step 5: Print summary table

```bash
python3 -c "
import json, os

def read_summary(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    s = d.get('summary', d)
    n = s.get('n_tasks', 0)
    base = s.get('n_base_pass', 0)
    plus = s.get('n_plus_pass', 0)
    return base/n*100 if n else None, plus/n*100 if n else None

print('tau | HumanEval | HumanEval+ | MBPP | MBPP+')
print('----|-----------|------------|------|------')

# AR baseline
r = read_summary('outputs/tau_rerun/ar_humaneval_summary.json')
r2 = read_summary('outputs/tau_rerun/ar_mbpp_summary.json')
if r and r2:
    print(f'AR  | {r[0]:.1f} | {r[1]:.1f} | {r2[0]:.1f} | {r2[1]:.1f}')

for t in ['0.5', '0.7', '0.8', '0.9', '0.93', '0.95', '0.97', '0.99']:
    rh = read_summary(f'outputs/tau_rerun/remask_humaneval_t{t}_summary.json')
    rm = read_summary(f'outputs/tau_rerun/remask_mbpp_t{t}_summary.json')
    he = f'{rh[0]:.1f} | {rh[1]:.1f}' if rh else '--- | ---'
    mb = f'{rm[0]:.1f} | {rm[1]:.1f}' if rm else '--- | ---'
    print(f'{t}  | {he} | {mb}')
"
```

---

## Step 6: Diagnostic — check mask fraction at τ=0.5

After running Step 2, check how many tokens actually got masked at τ=0.5 (to confirm whether "catastrophic" behavior is real or the original data was wrong):

```bash
python3 -c "
import json

path = 'outputs/tau_rerun/remask_humaneval_t0.5.jsonl'
changed = unchanged = 0
for line in open(path):
    r = json.loads(line)
    if r.get('draft_completion','').strip() != r.get('raw_completion','').strip():
        changed += 1
    else:
        unchanged += 1
print(f'tau=0.5: {changed} drafts modified, {unchanged} unchanged (out of {changed+unchanged})')
"
```

---

## Expected outcomes

| Scenario | τ=0.5 result | Interpretation |
|----------|-------------|----------------|
| Genuine failure | << AR baseline | Something breaks at very sparse mask; investigate denoising |
| Data was wrong | ≈ AR baseline | τ=0.5 ≈ "no change"; 13.4% was an artifact of Kodai's run |

---

## Files to update after rerun

Once results are in, update the following:

1. **`NeurIPS26-CoCoder/tables/tab_tau.tex`** — replace all numbers with rerun results; fix the AR baseline in the caption to match the re-evaluated baseline from Step 1.

2. **`NeurIPS26-CoCoder/section/05_experiments.tex`** (lines 62–64) — update the τ=0.5 description if the result changes.

3. Optionally re-evaluate `base_tuteng/deepseek_dream_remask_humaneval_t0.9_timed-sanitized.jsonl` with current EvalPlus to get a self-consistent main-table DeepSeek+Dream number.
