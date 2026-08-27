# Experiment 2D2G Final Report

Primary classification: **POSITIVE UTILITY ESTABLISHED**

## Architecture

Stage A continued exact 2D2B for 191 matched updates. Stage B kept B2 W1024 with no B11 recurrence and added B3 W64 plus B10 recurrence at lags 64–1023.

## True incremental result

- Real: `3.058321604749196`
- B3 off: `3.058428822703619`
- B3 shuffled: `3.058373571699625`
- Gain: `0.00010721795442325543`
- Sequence gap: `5.196695042908317e-05`
- Wins vs off: `140/256`
- Wins vs shuffled: `132/256`

## Integrity

Final audit passed: `True`.
Stage B contains no B11 recurrent gate or raw-state ring.
