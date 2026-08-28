EXPERIMENT 2D3A — 100M COMPLETE

PRIMARY CLASSIFICATION:
MULTI-LINK POSITIVE RECURRENT PYRAMID

2D3A CUMULATIVE TARGETS:
100,139,008

B3 TRUE RECURRENT GAIN:
4.620920874698342e-05

B5 TRUE RECURRENT GAIN:
0.00023513201926261829

B6 TRUE RECURRENT GAIN:
3.919779449601535e-05

B3 TRUE SEQUENCE GAP:
6.010995910576966e-05

B5 TRUE SEQUENCE GAP:
0.00021931987573919898

B6 TRUE SEQUENCE GAP:
0.00010982944737936151

## Source
- Path: `/workspace/exp2d3a_source/stage_a_scientific_update_0191.pt`
- SHA-256: `cb5dd5904779617959b5619982a9dfe69f0c4d705679652f4f99a8285879b5e8`

## Architecture and training
- B1 W2 + B12 recurrence; B2 W1024; B3 W32 + B10 recurrence; B4 W1024; B5 W64 + B8 recurrence; B6 W512 + B7 recurrence; B7-B12 W1024.
- Parameters: 124,475,908
- Runtime: 1807.128s; mean 9.461s/update.

| Destination | Local window | Recurrent source | Recurrent lags |
|---|---:|---|---|
| B1 | 2 | B12 post-MLP residual | 2–1023 |
| B2 | 1024 | — | — |
| B3 | 32 | B10 post-MLP residual | 32–1023 |
| B4 | 1024 | — | — |
| B5 | 64 | B8 post-MLP residual | 64–1023 |
| B6 | 512 | B7 post-MLP residual | 512–1023 |
| B7–B12 | 1024 | — | — |

## Link classifications
- B3: POSITIVE UTILITY
- B5: POSITIVE UTILITY
- B6: SEQUENCE-SPECIFIC BUT NOT ESTABLISHED

## Continuation
- Final checkpoint: `/workspace/exp2d3a_run/checkpoints/scientific_cumulative_000100139008.pt`
- SHA: `8727e86c6f18164f3a8104af3c726290536136d9f8d0efe810dcc29656d33667`
- Exact next-batch SHA: `91fa2cae4e6e52cfddd2b470175ec704f0548b447f02861917ec548736fe18e7`
- The checkpoint is resume-ready; no gate, optimizer, scheduler, loader, warmup, or RNG reset is permitted.

## Operational closure
- Git branch: `experiment-2d3a-alternating-integration-pyramid-100m`
- Final evidence commit before postflight metadata: `d27a6a0c`
- Artifact directory: `results/experiment_2d3a_alternating_integration_pyramid_100m`
- Local backup: `/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d3a_alternating_integration_pyramid_100m/scientific_cumulative_000100139008.pt`
- Independently verified local SHA-256: `8727e86c6f18164f3a8104af3c726290536136d9f8d0efe810dcc29656d33667`
- Persistent network volume retained: `yhzyb27fb5`
- Exact pod scheduled to stop after final Git synchronization: `7i2zyd53ytspwz` (`empirical_tan_panda`); pod will not be deleted.

## Q1–Q45
- **Q1. What exact Stage-A source path was resolved?**
  /workspace/exp2d3a_source/stage_a_scientific_update_0191.pt
- **Q2. Did its SHA equal the required SHA?**
  True
- **Q3. What was inherited B1 gate at update 0?**
  0.12740735709667206
- **Q4. What was source 2D2G-A canonical CE?**
  3.089876675605774
- **Q5. What was initial joint compression damage?**
  0.19281643629074097
- **Q6. What was B3-W32-only damage?**
  0.07822490930557269
- **Q7. What was B5-W64-only damage?**
  0.028761005401611417
- **Q8. What was B6-W512-only damage?**
  0.003239083290100364
- **Q9. Did B3 gate open?**
  True
- **Q10. At what update/sign?**
  update 1 / positive
- **Q11. Did B5 gate open?**
  True
- **Q12. At what update/sign?**
  update 1 / positive
- **Q13. Did B6 gate open?**
  True
- **Q14. At what update/sign?**
  update 1 / positive
- **Q15. Final B1 gate?**
  {'raw': 0.17240437865257263, 'effective': 0.1707163006067276}
- **Q16. Final B3 gate?**
  {'raw': 0.013444459065794945, 'effective': 0.01344364881515503}
- **Q17. Final B5 gate?**
  {'raw': 0.04243546724319458, 'effective': 0.04241001233458519}
- **Q18. Final B6 gate?**
  {'raw': 0.04043339565396309, 'effective': 0.04041137546300888}
- **Q19. B3 gain at update 48?**
  4.0042400359929786e-05
- **Q20. B3 gain at update 96?**
  3.6597251891201665e-06
- **Q21. B3 gain at update 143?**
  4.699230194082915e-05
- **Q22. B3 gain at update 191?**
  -7.486343383433791e-06
- **Q23. B5 gain at update 48?**
  4.438161849984468e-05
- **Q24. B5 gain at update 96?**
  3.14354896544522e-05
- **Q25. B5 gain at update 143?**
  6.357431411752046e-05
- **Q26. B5 gain at update 191?**
  7.80105590822977e-05
- **Q27. B6 gain at update 48?**
  1.8942356109441505e-05
- **Q28. B6 gain at update 96?**
  6.663799285711036e-06
- **Q29. B6 gain at update 143?**
  -1.5974044802469223e-06
- **Q30. B6 gain at update 191?**
  2.647638320940615e-05
- **Q31. Final true B1 gain/gap?**
  [0.007576267839606743, 0.008448259596616658]
- **Q32. Final true B3 gain/gap?**
  [4.620920874698342e-05, 6.010995910576966e-05]
- **Q33. Final true B5 gain/gap?**
  [0.00023513201926261829, 0.00021931987573919898]
- **Q34. Final true B6 gain/gap?**
  [3.919779449601535e-05, 0.00010982944737936151]
- **Q35. True paired wins for each link vs Off?**
  {'b1': 236, 'b3': 134, 'b5': 139, 'b6': 127}
- **Q36. True paired wins for each link vs Shuffled?**
  {'b1': 210, 'b3': 140, 'b5': 147, 'b6': 145}
- **Q37. Combined new-link true gain?**
  0.0003269422815974643
- **Q38. Combined new-link sequence gap?**
  0.00024995758083434794
- **Q39. Did long-lag writer gradients reach all eligible bins?**
  True
- **Q40. Did B6 provide positive representation utility?**
  True
- **Q41. What is exact theoretical BF16 inference state?**
  33288192
- **Q42. What is exact saving vs Standard?**
  4423680
- **Q43. Did 8-pass self-composition remain stable?**
  True
- **Q44. What checkpoint should be used for 250M continuation?**
  /workspace/exp2d3a_run/checkpoints/scientific_cumulative_000100139008.pt
- **Q45. Is it proven resume-ready?**
  True

FUTURE 2D3A MATURATION TARGETS:

- 250M: cumulative updates 477; cumulative targets 250,085,376
- 500M: cumulative updates 954; cumulative targets 500,170,752
- 1B: cumulative updates 1908; cumulative targets 1,000,341,504

NO FURTHER TRAINING WAS RUN AFTER 100M.

# EXPERIMENT 2D3A 100M COMPLETE
