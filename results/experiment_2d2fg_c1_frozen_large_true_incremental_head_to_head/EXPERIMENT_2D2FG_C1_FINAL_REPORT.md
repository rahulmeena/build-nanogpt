EXPERIMENT 2D2FG-C1 COMPLETE

ABSOLUTE QUALITY CLASSIFICATION:
2D2G ABSOLUTE CE ADVANTAGE STRONGLY CONFIRMED

2D2F REAL CE:
3.091034226748601

2D2G REAL CE:
3.0867643054976432

F_MINUS_G CE:
0.004269921250957598

2D2F B3 RECURRENCE CONFIRMATION:
STRONGLY CONFIRMED

2D2G B3 RECURRENCE CONFIRMATION:
DIRECTIONALLY CONFIRMED

CANONICAL ARCHITECTURE UNDER PREREGISTERED CE/STATE POLICY:
2D2F

## Frozen provenance

- 2D2F checkpoint SHA-256: `a58dd647e7aa70c22b5c1cd49cc708d85576e95c0a0c30fc93b1f0c02eae0ea6`
- 2D2G checkpoint SHA-256: `36ec3fa28741fa8ce999c43e309baa175112b795c079575d2a89525f134c3da0`
- Validation subset SHA-256: `1e78e49423dfc2d8064daaaac36d72b6fca2745822eb1f02b6b99f8415ddb7ca`
- Sequences: `1024`; targets per control: `1048576`.
- Mandatory and optional disjointness passed: `True`.

## Absolute CE and paired sequence result

- F_REAL: `3.091034226748601`
- G_REAL: `3.0867643054976432`
- F minus G: `0.004269921250957598`
- Mean/median/std: `{'count': 1024, 'mean': 0.0042699212509575545, 'median': 0.003967571552632165, 'negative': 190, 'positive': 834, 'sample_standard_deviation': 0.004964310812892337, 'ties': 0}`
- F wins / G wins / ties: `190` / `834` / `0`
- Bootstrap 95% CI: `[0.00397041806403922, 0.004575195470495523]` (20,000 paired sequence resamples; seed 20260828).

## B3 recurrence

- 2D2F gain OFF-REAL: `9.204009314611739e-05`; CI `[1.643824259555109e-05, 0.00016789940023979774]`; paired `{'count': 1024, 'mean': 9.204009314554102e-05, 'median': 8.951215244223931e-05, 'negative': 487, 'positive': 537, 'sample_standard_deviation': 0.0012197081690233065, 'ties': 0}`.
- 2D2F gap SHUFFLED-REAL: `8.009180974921648e-05`; CI `[3.172492915208027e-06, 0.00015714882279736742]`; paired `{'count': 1024, 'mean': 8.009180974945197e-05, 'median': 9.80579425586825e-05, 'negative': 478, 'positive': 546, 'sample_standard_deviation': 0.0012574125472596363, 'ties': 0}`.
- 2D2G gain OFF-REAL: `6.784321865183074e-06`; CI `[-6.698714491316685e-05, 7.992457295652972e-05]`; paired `{'count': 1024, 'mean': 6.784321865337464e-06, 'median': 4.3008379069675584e-05, 'negative': 506, 'positive': 518, 'sample_standard_deviation': 0.0012136703297954293, 'ties': 0}`.
- 2D2G gap SHUFFLED-REAL: `1.739761153807251e-06`; CI `[-6.962111917924888e-05, 7.578674689063815e-05]`; paired `{'count': 1024, 'mean': 1.7397611541502926e-06, 'median': -1.0487969873906877e-05, 'negative': 515, 'positive': 509, 'sample_standard_deviation': 0.001187782765261354, 'ties': 0}`.
- F gain minus G gain: `8.525577128093431e-05`; CI `[-1.8190776154569123e-05, 0.00018930459532761056]`.
- F gap minus G gap: `7.835204859540923e-05`; CI `[-2.5891583745558486e-05, 0.00018056570189069584]`.

## Position-bin comparison

| Positions | F CE | G CE | F-G |
|---|---:|---:|---:|
| 1-31 | 3.89161300986 | 3.89123780192 | 0.000375207941151 |
| 32-63 | 3.32847814693 | 3.32721134671 | 0.00126680022466 |
| 64-127 | 3.20475020167 | 3.20156434564 | 0.0031858560239 |
| 128-255 | 3.09701525031 | 3.0939669606 | 0.00304828971095 |
| 256-511 | 3.06811227003 | 3.06347967376 | 0.00463259627142 |
| 512-767 | 3.02572366603 | 3.02088808723 | 0.00483557879701 |
| 768-1023 | 3.02142762717 | 3.01636049481 | 0.00506713236425 |

## Memory and preregistered policy

- F state: `31718400` bytes; G state: `34765824` bytes.
- F saves `3047424` bytes (`2.90625` MiB; `8.765573915348591`%).
- F CE cost: `0.004269921250957598`; CE cost per MiB saved: `0.0014692202153832594`.
- The preregistered <=0.005 CE/state policy selects **2D2F**.

## Scientific questions Q1-Q25

### Q1

`3.091034226748601`

### Q2

`3.0867643054976432`

### Q3

`0.004269921250957598`

### Q4

`{"difference": 0.0006346368874226016, "historical": 0.0036352843635349963, "new": 0.004269921250957598}`

### Q5

`190`

### Q6

`834`

### Q7

`0.003967571552632165`

### Q8

`{"lower": 0.00397041806403922, "upper": 0.004575195470495523}`

### Q9

`"2D2G ABSOLUTE CE ADVANTAGE STRONGLY CONFIRMED"`

### Q10

`9.204009314611739e-05`

### Q11

`8.009180974921648e-05`

### Q12

`537`

### Q13

`546`

### Q14

`{"gain": {"lower": 1.643824259555109e-05, "upper": 0.00016789940023979774}, "gap": {"lower": 3.172492915208027e-06, "upper": 0.00015714882279736742}}`

### Q15

`6.784321865183074e-06`

### Q16

`1.739761153807251e-06`

### Q17

`518`

### Q18

`509`

### Q19

`{"gain": {"lower": -6.698714491316685e-05, "upper": 7.992457295652972e-05}, "gap": {"lower": -6.962111917924888e-05, "upper": 7.578674689063815e-05}}`

### Q20

`8.525577128093431e-05`

### Q21

`{"lower": -1.8190776154569123e-05, "upper": 0.00018930459532761056}`

### Q22

`"2D2F"`

### Q23

`{"bytes": 3047424, "mib": 2.90625, "percent": 8.765573915348591}`

### Q24

`true`

### Q25

`"2D2J: add B4 W128 plus B9\u2192B4 to the clean frozen 2D2F architecture; retain B1 W2+B12\u2192B1, B2 W32 with no recurrence, and B3 W64+B10\u2192B3."`

## Exactly one recommended next training experiment

2D2J: add B4 W128 plus B9→B4 to the clean frozen 2D2F architecture; retain B1 W2+B12→B1, B2 W32 with no recurrence, and B3 W64+B10→B3.

Do not execute it as part of this confirmation.

## Post-GPU finalization disclosure

The frozen scientific evaluation and terminal scientific audit completed on the pod. The pod image lacked Matplotlib, so the already-written scientific JSON was checksum-copied locally, the pod was stopped, and only plots/report publication were completed locally. No model execution, training, checkpoint, subset, or scientific metric changed.

## Integrity, Git, and runtime

- Zero mutation counters: `{'backward_calls': 0, 'optimizer_steps': 0, 'parameter_updates': 0, 'scheduler_steps': 0, 'training_targets': 0}`
- Cache and physical-state audit passed: `True`
- Implementation regression passed: `True`
- Results commit: `b80af970170d030e62037ec4d4c49069b4676e58`
- Artifact path: `/Users/rahul/Documents/GPT-2 Enhancement/parallel_2d2_master_dev/2d2fg_c1/results/experiment_2d2fg_c1_frozen_large_true_incremental_head_to_head`
- Pod `rvgztsr0azrwyo` was stopped at `2026-08-27T20:35:58Z` after a verified local backup.
- Persistent volume `yhzyb27fb5` is retained.

# EXPERIMENT 2D2FG-C1 COMPLETE
