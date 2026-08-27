# Experiment 2D2H — Final Report

## Result

**B2 W32 SECOND RECURRENT LINK IS HARMFUL**

- Recommendation: IMPROVE B2 RECURRENT READOUT BEFORE ADDING B10→B3
- Source: exact final 2D2B `8c39f47248e3a5f4dc69f5e8e97c8a1cd1bcdfa91154eba5804c448942075326`
- Final checkpoint: `6929a4d37a2c5b87bbd35cc4cfe1e1c8cccd302a12110ef31d5b96bdd3371938`
- Parameters: 124,475,905
- B1: physical W2 only; B12 recurrence, gate, ring, and optimizer state absent
- B2: W32 local attention plus B11 recurrent K/V for lags 32…1023
- Initial combined damage: 0.0438600973614
- True incremental Real: 3.0706384049
- True incremental B2-Off: 3.0704821506
- True incremental B2-Shuffled: 3.0706134842
- True recurrent gain: -0.000156254296179
- True sequence gap: -2.49206953598e-05
- Wins vs Off: 110/256
- Wins vs Shuffled: 126/256
- Final tanh(g_rec_B2): -0.00505795702338
- Eight-pass stability: True
- Scientific integrity: True
- Evaluation-only FP32/TF32 audit correction disclosed: True

The final checkpoint is persistent. Update-96 and smoke checkpoints are ephemeral;
their hashes and strict-reopen audits remain in the artifact set. Stop—but do not
delete—the pod only after repository/local artifact synchronization is verified.
