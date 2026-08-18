# Comparison and Ablation Experiments

## 1. Why the L2 Final Merge Timed Out

The previous run succeeded through L1 extraction and L2 batch merging, then timed out at `l2_paper_merge_final`. This usually means the global merge call is doing heavier reasoning and longer JSON generation than the earlier batch calls. The code has been updated in two ways:

- The default request timeout is now 600 seconds.
- L2 final merge now receives compact batch summaries rather than full batch JSON.

Because earlier L1/L2 batch calls were cached, rerunning the same command should reuse previous results and continue from the final merge stage.

## 2. Conditions

The experiment runner supports five conditions:

- `proposed_full`: the full L1-L2-L3 multimodal streaming pipeline.
- `baseline_oneshot_mineru`: a one-shot baseline using MinerU text and sampled images in a single model call.
- `ablation_text_only`: removes image inputs to test the contribution of cross-modal evidence.
- `ablation_no_l3`: removes the final quality-audit stage to test L3.
- `ablation_large_chunk`: uses much larger chunks to weaken section-aware streaming.

## 3. Run Experiments

Set the API key in the environment:

```powershell
$env:DASHSCOPE_API_KEY = "your_key_here"
```

Run all conditions:

```powershell
python experiments/ablation_runner.py --paper-dir data/paper1
```

Run selected conditions:

```powershell
python experiments/ablation_runner.py --paper-dir data/paper1 --conditions proposed_full,ablation_text_only
```

Dry-run without API calls:

```powershell
python experiments/ablation_runner.py --paper-dir data/paper1 --dry-run
```

## 4. Compute Automatic Metrics

After running experiments:

```powershell
python experiments/metrics.py --result-root experiments/results/paper1
```

This writes:

- `experiments/results/paper1/metrics_summary.json`
- `experiments/results/paper1/metrics_summary.csv`

## 5. Suggested Paper-Level Evaluation

Automatic metrics are useful but not sufficient for a journal paper. Recommended evaluation:

- Automatic schema validity: JSON parse success and required-field coverage.
- Automatic evidence coverage: ratio of extracted items with evidence references.
- Automatic cross-modal usage: ratio of items using figures/tables/equations/images.
- Human granularity score: 1-5 rating for atomicity of research problems and methods.
- Human faithfulness score: 1-5 rating for whether each item is supported by the paper.
- Human usefulness score: 1-5 rating for whether the result helps scientific paper mining.

For each paper, compare `proposed_full` against the one-shot baseline and ablations.
