# EXPERIMENT 2D8 — OVERLAP WIDTH N/O1/O2 COMPLETE

## FRESH PANEL

Sequences: 4096

Fresh N CE: `3.030318233011`
Fresh O1 CE: `3.030163707525`
Fresh O2 CE: `3.030168359876`

Fresh perplexities:
N: `20.703820180166`
O1: `20.700621159467`
O2: `20.700717466242`

## PAIRWISE FRESH RESULTS

N − O1: `+0.000154525486`
95% CI: `[+0.000103576253, +0.000205237518]`
Perplexity ratio: `1.000154537425`
Classification: O1 superior to N beyond delta_CE

O1 − O2: `-0.000004652351`
95% CI: `[-0.000051136085, +0.000041085229]`
Perplexity ratio: `0.999995347660`
Classification: practical equivalence established

N − O2: `+0.000149873135`
95% CI: `[+0.000099390197, +0.000200113966]`
Perplexity ratio: `1.000149884367`
Classification: O2 statistically superior to N; beyond-margin superiority not established

delta_CE:
0.0001

Per-sequence wins:
N vs O1: N 1891, O1 2205, ties 0
O1 vs O2: O1 2070, O2 2026, ties 0
N vs O2: N 1886, O2 2210, ties 0

Fresh point-estimate ranking:
1. O1
2. O2
3. N

## N-vs-O1 CONFIRMATION

Old reused sealed matched panel:
N − O1: `+0.000149591044`
95% CI: `[+0.000077991189, +0.000221717781]`

Fresh panel:
N − O1: `+0.000154525486`
95% CI: `[+0.000103576253, +0.000205237518]`

Stratified pooled N − O1: `+0.000152880672`
95% CI: `[+0.000110933481, +0.000193975395]`
Classification: O1 superior to N beyond delta_CE

Heterogeneity:
D_fresh − D_old: `+0.000004934442`
95% CI: `[-0.000082897880, +0.000092526309]`

## PERSISTENT STATE

N: `33,289,728` bytes/sequence
O1: `33,289,728` bytes/sequence
O2: `33,289,728` bytes/sequence

O1 − N: `+0` bytes/sequence
O2 − N: `+0` bytes/sequence
O2 − O1: `+0` bytes/sequence

## O2 TRAINING / PROVENANCE

Starting global update: 2099
First new update:       2100
Final global update:    2290

Final cumulative targets:
1,200,619,520

O2 checkpoint:
`/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d8_trained_overlap_width_n_o1_o2/O2/scientific_cumulative_001200619520.pt`
SHA-256: `792b85f99164e8d1a096e2913e9d116f7f544e84d63c5a48c9a20e136cd9b69f`

Terminal loader cursor SHA:
`682abcbcc8db8886274ccbb604927683af38d010ecae31b1494987513ae982d3`
Terminal next-global-batch SHA:
`a021ce09f7a25b6617632e2a76da1acb0980ac1dda888df5f1c8eb65b3939fbe`
Terminal next-stream SHA:
`7918ea7e6f979b8e49fca89c60dd68ace44b768854d34bc96dd751bee07b2567`
MATCH

AUDIT:
PASS

GPU STATUS:
STOPPED

## SCIENTIFIC INTERPRETATION

1. Did fresh data replicate the O1 advantage over N?
Yes. Fresh data replicate O1's statistical superiority over N beyond the 0.0001 CE margin.

2. Is O2 better than O1?
No established improvement; O1 and O2 are practically equivalent at the ±0.0001 CE scale.

3. Is there evidence for an overlap-width dose response?
No established monotonic overlap-width dose response; the fresh point-estimate ordering is O1 < O2 < N.

4. Which geometry should become the preferred architecture?
O1. O1 remains favored over N and wider overlap does not establish an improvement over O1.

5. Is any further overlap-width experiment warranted?
No. Wider overlap is equivalent to or worse than O1, so further width expansion is not currently warranted.
