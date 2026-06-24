# Specs Index

Active specs stay in this directory. Completed or superseded specs live in `done/`.

## Active

| Spec | Status |
|---|---|
| `spec_calibration_plot.md` | Pending: calibration plots exist on disk; need to copy to `images/` and add figure to `app:calibration` in appendix. No GPU needed. ~20 min. |
| `spec_oracle_locator_run.md` | Pending: code exists in `src/`; need to run Phase 1–3 (1 GPU, ~2h) then integrate results into `tab_locator` and `06_analysis.tex`. |
| `spec_granularity_ablation.md` | Pending: token/span/line mask granularity ablation still has unchecked completion items. |
| `spec_mistral_codellama.md` | Pending final integration check: result artifacts exist, but this spec has not been marked complete in this docs pass. |
| `spec_research_writing_benchmarks.md` | Pending: research/writing benchmark generation and evaluation runbook. |
| `spec_rewriting_benchmarks.md` | Pending: ASSET/CoEdIT rewriting benchmark generation and evaluation runbook. |
| `spec_seed_coder_instruct_conservative_remask.md` | Active follow-up: pkgv2 reanalysis separates packaging artifacts from true raw-code harm; gate safety is strongest for LLaDA/MBPP, while Dream HumanEval is no-op under current packaging. |
| `spec_table4_expand_ar_baselines.md` | Pending: Table 4c/4d baseline expansion for Llama-3.1 and StarCoder2. |

## Done

See `done/README.md`.
