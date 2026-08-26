EXPERIMENT 2D2B PRIMARY CLASSIFICATION:
FULL-BANK RECURRENT K/V SCALES POSITIVE UTILITY

FINAL TRUE-SELF FULL-BANK GAIN:
0.003682778211103077

FINAL TRUE-SELF SEQUENCE GAP:
0.004046077382548585

FINAL TRUE-SELF BANK-WIDTH GAIN:
0.0023854059530723504

# Experiment 2D2B — Full-Width B12→B1 Token-Indexed Recurrent K/V Bank

## Result

The final classification is **FULL-BANK RECURRENT K/V SCALES POSITIVE UTILITY**.
The exactly one next experiment is **EXTEND FULL-BANK RECURRENT K/V TO MIRRORED HIGH→LOW LAYER PAIRS**.

## Source and architecture

- Source checkpoint: `/workspace/exp2d2a_run/checkpoints/scientific_update_0096.pt`
- Source SHA-256: `24fd2481e220ec504db3a6e912054d0ad502cdb3a6fc497b22dd32ec682e3afb`
- Parameters: 124,475,905 (zero new versus 2D2A)
- Hardware: 1 × NVIDIA A100-SXM4-80GB
- B1 local bank: `t-1,t` (W=2)
- B1 recurrent bank: `max(0,t-1023)...t-2` (maximum 1022 entries)
- B2–B12: W=1024; overall T=1024
- Recurrent source: B12 post-MLP residual immediately before `ln_f`
- Existing B1 LN/K/V slices, separate recurrent softmax, one `c_proj`, one trained scalar gate

Experiment 2D2A already used attached temporal B12 source states. Temporal writer gradients into Pass-1 B12 states were finite and nonzero during training. 2D2B is the first full recurrent-token-bank experiment, not the first writer-learning experiment.

The disclosed 2D2A evaluation-only BF16 correction is preserved: active-prefix absolute comparisons use the already-preregistered Plain tolerance 1.25; strict FP32 checks are unchanged. No model, checkpoint, training, data, loss, or scientific metric was changed by that correction.

## Training

- Additional updates: 191 / 191
- Additional targets: 100,139,008
- Cumulative 2D2 targets: 150,470,656
- Runtime: 1667.65 seconds (8.73 sec/update)
- Throughput: 60047.88 targets/sec
- Peak allocated/reserved VRAM: 54993.03/62516.00 MiB
- Gate: raw 0.02386544458568096 → 0.07656901329755783; tanh 0.023860914632678032 → 0.07641972601413727
- Optimizer, loader, and Python/NumPy/Torch CPU/CUDA RNG states resumed from 2D2A; warmup was not restarted.
- Mandatory fresh-process restart after local update 96 passed.

## Zero-shot bank expansion

- Plain: 3.1331379047466728
- FullReal: 3.1314429954088157
- FullShuffled: 3.1321171021028933
- Legacy TwoSlotReal: 3.13081073770154
- Full-bank zero-shot gain: 0.0016949093378570979
- Zero-shot sequence gap: 0.0006741066940776363
- Legacy bank-expansion gain: -0.0006322577072754854

## Final parallel validation

- Plain: 3.104398060634412
- FullReal: 3.09949293354166
- FullShuffled: 3.102238688560283
- TwoSlotReal: 3.102510876629276
- Full-bank gain: 0.004905127092751815
- Sequence gap: 0.002745755018622731
- Bank-width gain: 0.0030179430876158264
- Paired batch wins (Plain/Shuffled/TwoSlot): 20/20/20 of 20

## Final true incremental validation

- Plain: 3.0481036438847378
- FullReal: 3.0444208656736347
- FullShuffled: 3.0484669430561833
- TwoSlotReal: 3.046806271626707
- Targets/control: 131,072
- Sequence wins vs Plain/Shuffled/TwoSlot: 113/101/85 of 128

## Attention, gradients, and old-memory ablation

- Final recurrent mass partitions: {'lags_128_511': 0.4632356339761837, 'lags_2_31': 0.1402054201535483, 'lags_32_127': 0.2531486464136836, 'lags_512_1023': 0.14341029945658437}
- Strongest normalized-density bin: 2-7
- Long/short writer-gradient RMS ratio: 56.91631937731549
- Recent-only minus Full loss: -0.0011000649415109365
- Plain minus Old-only loss: 0.004044190359854483

## Cache and storage

- B1 historical same-layer KV: at most 1 entry
- B12 raw recurrent buffer: at most 1023 states
- B2–B12 ordinary historical KV: at most 1023 entries/layer
- BF16 total experimental inference state, B=1: 34.469 MiB
- BF16 total experimental inference state, B=64: 2206.031 MiB

> 2D2B is a mechanism experiment and does not claim whole-model KV savings because B2-B12 remain full-context.

## Scientific questions Q1–Q27

### Q1. Did frozen 2D2A weights benefit zero-shot from widening the bank?

false

### Q2. What was zero-shot full-bank gain?

0.0016949093378570979

### Q3. What was zero-shot bank-width gain versus two-slot?

-0.0006322577072754854

### Q4. How did full-bank recurrent gain evolve?

{"0": 0.0016949093378570979, "143": 0.004081544944801951, "191": 0.004905127092751815, "20": 0.002181393742921589, "48": 0.002273339939256047, "96": 0.0032954346372204846}

### Q5. At what milestone did FullReal beat Plain?

0

### Q6. At what milestone did FullReal beat TwoSlotReal?

96

### Q7. Did sequence-specificity strengthen with bank width?

true

### Q8. What fraction of recurrent mass remained at lags 2-31?

0.1402054201535483

### Q9. What fraction went to lags 32-127?

0.2531486464136836

### Q10. What fraction went to lags 128-511?

0.4632356339761837

### Q11. What fraction went to lags 512-1023?

0.14341029945658437

### Q12. Which lag range had strongest normalized density?

"2-7"

### Q13. Did heads specialize by temporal distance?

true

### Q14. Did masking old memory hurt validation?

false

### Q15. Does OLD_ONLY contain positive utility?

true

### Q16. Does benefit grow with current position?

{"128-255": {"available_recurrent_history": [127, 254], "bank_width_gain": 0.004579019076481932, "full_bank_gain": 0.005914081013179397, "full_real_loss": 3.1200333284257793, "full_shuffled_loss": 3.123508109262912, "plain_loss": 3.1259474094389588, "sequence_gap": 0.0034747808371321277, "two_slot_real_loss": 3.1246123475022616}, "16-31": {"available_recurrent_history": [15, 30], "bank_width_gain": -0.0015823743829970482, "full_bank_gain": 0.009107509872410352, "full_real_loss": 3.5789951870101504, "full_shuffled_loss": 3.5817593803280037, "plain_loss": 3.5881026968825607, "sequence_gap": 0.002764193317852881, "two_slot_real_loss": 3.5774128126271534}, "2-15": {"available_recurrent_history": [1, 14], "bank_width_gain": -0.0009358887560664909, "full_bank_gain": 0.005626631727708276, "full_real_loss": 4.075986236200801, "full_shuffled_loss": 4.076402031071484, "plain_loss": 4.081612867928509, "sequence_gap": 0.0004157948706831695, "two_slot_real_loss": 4.075050347444735}, "256-511": {"available_recurrent_history": [255, 510], "bank_width_gain": 0.006550782494014109, "full_bank_gain": 0.005202295103299547, "full_real_loss": 3.0602922424550343, "full_shuffled_loss": 3.063170183787588, "plain_loss": 3.065494537558334, "sequence_gap": 0.002877941332553738, "two_slot_real_loss": 3.066843024949048}, "32-63": {"available_recurrent_history": [31, 62], "bank_width_gain": -0.0009920385928125774, "full_bank_gain": 0.008378771995194295, "full_real_loss": 3.346979861613363, "full_shuffled_loss": 3.3503289654210677, "plain_loss": 3.355358633608557, "sequence_gap": 0.003349103807704562, "two_slot_real_loss": 3.3459878230205504}, "512-767": {"available_recurrent_history": [511, 766], "bank_width_gain": 0.0028266275672649433, "full_bank_gain": 0.004411290667485456, "full_real_loss": 3.0347560631165837, "full_shuffled_loss": 3.0375484342592247, "plain_loss": 3.0391673537840687, "sequence_gap": 0.0027923711426410674, "two_slot_real_loss": 3.0375826906838483}, "64-127": {"available_recurrent_history": [63, 126], "bank_width_gain": 0.0010519819188630347, "full_bank_gain": 0.006255002661782791, "full_real_loss": 3.2093032869859597, "full_shuffled_loss": 3.212139166910492, "plain_loss": 3.215558289647743, "sequence_gap": 0.0028358799245325195, "two_slot_real_loss": 3.2103552689048227}, "768-1023": {"available_recurrent_history": [767, 1022], "bank_width_gain": 0.00041594191061450174, "full_bank_gain": 0.0035618591391539927, "full_real_loss": 3.0317461840564643, "full_shuffled_loss": 3.033998392415742, "plain_loss": 3.0353080431956188, "sequence_gap": 0.0022522083592775768, "two_slot_real_loss": 3.032162125967079}}

### Q17. Did writer gradients reach hundreds of tokens back?

true

### Q18. How did long-lag gradient compare with short-lag?

56.91631937731549

### Q19. Did the gate grow or shrink?

"grew"

### Q20. Did parallel gain transfer to true incremental inference?

true

### Q21. Final true-self FullReal vs Plain gain?

0.003682778211103077

### Q22. Final true-self sequence gap?

0.004046077382548585

### Q23. Final true-self bank-width gain?

0.0023854059530723504

### Q24. How much recurrent state is stored?

{"B1_B12_raw_BF16_bytes_B1": 1571328, "B1_B12_raw_BF16_bytes_B64": 100564992, "B1_historical_KV": 1, "raw_B12_states": 1023}

### Q25. Does the result justify mirrored links?

true

### Q26. Does it justify dedicated projections?

false

### Q27. What exactly one experiment should run next?

"EXTEND FULL-BANK RECURRENT K/V TO MIRRORED HIGH\u2192LOW LAYER PAIRS"

## Integrity and artifacts

- Final audit passed: True
- Final checkpoint: `/workspace/exp2d2b_run/checkpoints/scientific_update_0191.pt`
- Final checkpoint SHA-256: `8c39f47248e3a5f4dc69f5e8e97c8a1cd1bcdfa91154eba5804c448942075326`
- Implementation commit: `072d9825119a9ca9954747fb3c7cc749ff230e55`
- Results commit: `c8cdc34639758ed4a94ffd8b7a7a0e60230651da`
- Artifact directory: `/workspace/build-nanogpt-exp2d2b/results/experiment_2d2b_full_b12_b1_recurrent_bank`
- GPU pod: `serious_indigo_swordfish` (`e8nd7m6piw5km2`), status `READY_TO_STOP_AFTER_SEAL_COMMIT_PUSH`

# EXPERIMENT 2D2B COMPLETE

Final RunPod stop remains the only lifecycle action after the seal commit is pushed.
