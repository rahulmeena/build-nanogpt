EXPERIMENT 2D2A PRIMARY CLASSIFICATION:
TOKEN-INDEXED RECURRENT K/V LEARNS POSITIVE UTILITY
EXPERIMENT 2D2A SECONDARY CLASSIFICATION:
SEQUENCE-SPECIFIC RECURRENT K/V

# Experiment 2D2A final report

The B12→B1 token-indexed recurrent K/V pilot completed its exact 96-update, 50,331,648-target budget on one NVIDIA A100-SXM4-80GB. The result below keeps parallel Pass-2 behavior separate from deployment-equivalent true self recurrence.

## Verified setup

- Parameters: 124,475,904 parent + 1 scalar = 124,475,905 total.
- Geometry: B1 local W2; B12 recurrent states at t−3/t−2; B2–B12 W1024.
- Source checkpoint: `/workspace/exp2d1_assets/runs/gpt2_124m_fineweb10b_20260810T141222Z/checkpoints/model_19072.pt` (`924ce6c8392c06ae24ab8f2ffd203787ee0022055c54554bac43bd9a34037871`).
- Parent full CE: 3.0750437753; parent B1-W2 CE: 5.0488550166.
- Runtime: 849.2 s over 96 updates.

## Final losses

| Evaluation | Plain | Real | Shuffled | Recurrent gain | Sequence gap |
|---|---:|---:|---:|---:|---:|
| Parallel Pass-2 | 3.1331379047 | 3.1308107377 | 3.1327184089 | +0.0023271670 | +0.0019076712 |
| True incremental self | 3.0759934814 | 3.0738462681 | 3.0763928307 | +0.0021472133 | +0.0025465626 |

The raw gate ended at +0.0238654446, giving tanh(g_rec)=+0.0238609146. Temporal writer gradients into attached Pass-1 B12 states were finite and nonzero at updates 10, 20, 48, and 96.

Incremental storage passed with one historical B1 K/V entry, at most 1023 historical K/V entries in each B2–B12 cache, and a three-state raw B12 residual ring. This pilot does not claim whole-model KV-cache savings.

Exactly one next experiment is recommended: **EXTEND RECURRENT K/V TO THE MIRRORED HIGH→LOW LAYER PAIRS**. It was not executed.

## Q1–Q26

### Q1

124,475,904

### Q2

Yes. Exactly one scalar g_rec; total 124,475,905.

### Q3

Yes; gate-zero reproduced the B1-W2 oracle within the frozen tolerance.

### Q4

+1.973811241239 CE (B1-W2 5.048855016570 vs full 3.075043775332).

### Q5

It opened on update 1 to raw -2.99999519e-05 (tanh -2.99999519e-05).

### Q6

Yes; pinned gradient norms: [1.1838963473564945e-05, 0.00012170169065939263, 0.00037441562744788826, 0.0006502412143163383].

### Q7

-0.0000906922

### Q8

+0.0004743443

### Q9

+0.0015477375

### Q10

+0.0023271670

### Q11

Yes.

### Q12

20 updates.

### Q13

Final parallel Real-vs-Shuffled wins: 20/20; gap +0.0019076712.

### Q14

t-2 (0.491357 vs 0.508643).

### Q15

No clear split across heads.

### Q16

No; positions 257-1023 averaged +0.0020209892 versus +0.0032816218 for positions 3-256. The largest individual bin was 17-32 (+0.0059174891).

### Q17

Yes; no numerical divergence was observed and all preregistered finite/RMS/loss checks passed.

### Q18

Parallel gain +0.0023271670; true-self gain +0.0021472133.

### Q19

+0.0021472133

### Q20

+0.0025465626

### Q21

One historical B1 K/V entry.

### Q22

Three raw B12 residual states, the minimum pipeline ring.

### Q23

Two recurrent score/value entries per warmed B1 query (equal to the two-entry local branch); model-level runtime is reported in performance.json.

### Q24

No; follow the selected next-experiment rule first.

### Q25

Yes.

### Q26

EXTEND RECURRENT K/V TO THE MIRRORED HIGH→LOW LAYER PAIRS

## Integrity and handoff

- Integrity audit: PASS.
- Final checkpoint: `/workspace/exp2d2a_run/checkpoints/scientific_update_0096.pt` (`24fd2481e220ec504db3a6e912054d0ad502cdb3a6fc497b22dd32ec682e3afb`).
- Implementation Git commit: `26b2493f86e3b465035a51afe8a702e2611fa6f1`.
- Results Git commit: `PENDING RESULTS COMMIT`.
- Artifact directory: `/workspace/build-nanogpt-exp2d2a/results/experiment_2d2a_b12_b1_recurrent_kv`.
- GPU pod is stopped after final Git and artifact synchronization; the network volume and historical checkpoints are preserved.

# EXPERIMENT 2D2A COMPLETE
