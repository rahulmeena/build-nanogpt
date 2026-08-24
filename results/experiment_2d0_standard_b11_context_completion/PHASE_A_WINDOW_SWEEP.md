# Experiment 2D0 Phase-A B11 Window Sweep

All configurations used the same 20 canonical B64 × T1024 validation batches. B1–B10 and B12 retained full 1024-token context. There was no recurrence, optimizer, backward pass, or training.

| W_B11 | Validation loss | Damage vs W1024 | B11 state cosine | B12 state cosine |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 3.0750437753 | +0.0000000000 | 0.9999999917 | 0.9999999916 |
| 896 | 3.0753463744 | +0.0003025990 | 0.9998283632 | 0.9998353501 |
| 768 | 3.0757388638 | +0.0006950885 | 0.9996055236 | 0.9996199206 |
| 512 | 3.0773528399 | +0.0023090646 | 0.9988526994 | 0.9988820830 |

The W1024 validation value regressed to the historical 3.0750441551 value within 3.80e-7. Every window consumed the canonical batch collection with SHA-256 `3331f585ae53e8fa5be3690aeb82345bf43c9270559403f113e717e937f5cdeb`.

The preregistered rule did not select a window:

- W768 damage was below 0.01, so W512 was tested as the specified fallback.
- W512 damage was also below 0.01.
- W896 is not eligible when W768 damage is below 0.01, and its damage was below 0.01 in any case.
- No candidate had damage in the required `[0.01, 0.10]` interval.

Phase B is therefore not authorized. A new, shorter window requires explicit user approval and a new frozen sweep rule.

