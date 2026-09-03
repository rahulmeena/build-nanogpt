# EXPERIMENT 2D7 — BOUNDARY ALIGNMENT N/O/G COMPLETE

Baseline CE: `3.028566462714` (perplexity `20.667583591300`)
Overlap-1 CE: `3.028416871670` (perplexity `20.664492137125`)
Gap-1 CE: `3.028763854837` (perplexity `20.671663612167`)

Baseline − Overlap: `+0.000149591044`
95% CI: `[+0.000077991189, +0.000221717781]`

Baseline − Gap: `-0.000197392123`
95% CI: `[-0.000265466045, -0.000128593376]`

Overlap − Gap: `-0.000346983167`
95% CI: `[-0.000422119802, -0.000271681652]`

Perplexity ratios:
Baseline/Overlap: `1.000149602233`
Baseline/Gap: `0.999802627358`
Overlap/Gap: `0.999653077025`

Per-sequence wins:
Baseline vs Overlap: Baseline `963`, Overlap `1085`, ties `0`
Baseline vs Gap: Baseline `1133`, Gap `915`, ties `0`
Overlap vs Gap: Overlap `1189`, Gap `859`, ties `0`

Point-estimate ranking:
1. OVERLAP1_REAL
2. BASELINE_REAL
3. GAP1_REAL

delta_CE:
`0.0001`

Practical/statistical classification:
Baseline vs Overlap: Overlap-1 statistically superior to Baseline, but not beyond delta_CE
Baseline vs Gap: Baseline superior to Gap-1 by more than delta_CE
Overlap vs Gap: Overlap-1 superior to Gap-1 by more than delta_CE

Persistent state:
Baseline: `33,289,728` bytes/sequence
Overlap-1: `33,289,728` bytes/sequence
Gap-1: `33,289,728` bytes/sequence
Overlap − Baseline: `+0` bytes/sequence
Gap − Baseline: `+0` bytes/sequence

Training counters:
Starting global update: 2099
First new update:       2100
Final global update:    2290
Final cumulative targets: 1,200,619,520

Terminal stream equality:
Final loader cursor SHA: `682abcbcc8db8886274ccbb604927683af38d010ecae31b1494987513ae982d3`
Next-global-batch SHA: `a021ce09f7a25b6617632e2a76da1acb0980ac1dda888df5f1c8eb65b3939fbe`
Next-stream SHA: `7918ea7e6f979b8e49fca89c60dd68ace44b768854d34bc96dd751bee07b2567`
MATCH

Final checkpoints:
N: `/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d7_trained_boundary_alignment_nog/N/scientific_cumulative_001200619520.pt`
SHA-256: `57e62a2094693205b520e2986047d46c28d042d4ec34d6e65b2135f474adec20`

O: `/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d7_trained_boundary_alignment_nog/O/scientific_cumulative_001200619520.pt`
SHA-256: `c1354459f03703b31d25b649cd7b64d2aad95a60d1764b3dee3af7c675f59cc6`

G: `/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d7_trained_boundary_alignment_nog/G/scientific_cumulative_001200619520.pt`
SHA-256: `9fb22d0feda59ae2008bf07e6ef418c476d0c9819f1d743ba144ee16d5cd7f2c`

AUDIT:
PASS

GPU STATUS:
STOPPED

## SCIENTIFIC INTERPRETATION

The numerical ordering is OVERLAP1_REAL < BASELINE_REAL < GAP1_REAL. Overlap-1 is statistically better than Baseline, but its confidence interval does not establish an advantage beyond 0.0001 CE; both Overlap-1 and Baseline establish greater-than-margin superiority to Gap-1. Boundary-token continuity matters, while dual-depth duplication has no established practically meaningful gain over Baseline.

## RECOMMENDATION

Recommend BASELINE/N provisionally because no alternative established sufficient superiority.
