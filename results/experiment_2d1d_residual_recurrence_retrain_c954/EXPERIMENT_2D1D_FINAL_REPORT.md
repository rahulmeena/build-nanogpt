# Experiment 2D1D — End-to-End Residual Recurrence Retraining from C954

## Classification

**Primary:** RESIDUAL RECURRENCE APPROACHES NEUTRALITY

**Secondary:** SEQUENCE-SPECIFIC BUT NON-USEFUL RECURRENCE

The exact C954 checkpoint `22abc6de4e49e27504b4d0e66ca0d2e3396ed6d76d7ee18e0e11cfb1eb3192c0` was resumed at global update 954. Its base Transformer parameters and Adam state were preserved, while only `W_u`, `W_g`, and their optimizer state were reset to exact zero/fresh state. The result path used fixed `X = E + 0.03125F` for 477 updates (250,085,376 targets), ending at global update 1431.

**OPTIMIZER CONSISTENTLY PUSHES AGAINST W_U CAP**

## Validation trajectory

| Local update | Global update | Additional targets | Geometry | Plain CE | Real CE | Gain (Plain-Real) | Sequence gap | Real wins vs plain |
|---:|---:|---:|:---|---:|---:|---:|---:|---:|
| 0 | 954 | 0 | SOURCE-B | 3.0735813706 | 3.0735813706 | +0.0000000000 | +0.0000000000 | 0/20 |
| 20 | 974 | 10,485,760 | B-R | 3.0736800735 | 3.0748336232 | -0.0011535496 | +0.0046231747 | 1/20 |
| 48 | 1002 | 25,165,824 | B-R | 3.0710802725 | 3.0707878881 | +0.0002923844 | +0.0090867180 | 13/20 |
| 96 | 1050 | 50,331,648 | B-R | 3.0705618535 | 3.0706750145 | -0.0001131610 | +0.0124487523 | 8/20 |
| 191 | 1145 | 100,139,008 | C-R | 3.0749409631 | 3.0749614281 | -0.0000204650 | +0.0097906056 | 9/20 |
| 286 | 1240 | 149,946,368 | C-R | 3.0732906505 | 3.0739537570 | -0.0006631066 | +0.0086197029 | 6/20 |
| 477 | 1431 | 250,085,376 | C-R | 3.0723323574 | 3.0735785652 | -0.0012462078 | +0.0087749279 | 0/20 |

The immediate B12→C12 prestep shock at local 96 changed plain CE by `+0.0503261305`, real CE by `+0.0405582364`, and recurrent gain by `+0.0097678941` without an optimizer update.

## Scale and stability

Final effective `alphaF/E` was `0.3512583284` and `X/E` was `1.0549245029`. The maximum training `X` RMS was `0.0382811725` against the hard threshold `0.3550996296`. Final 32-pass classification: **BOUNDED OSCILLATORY**.

Projection by stage: `{"B-R": {"fraction_projected": 0.125, "maximum_post_sigma": 1.0262387990951538, "maximum_raw_sigma": 1.0404261350631714, "mean_projection_scale": 0.9987015562137395, "minimum_projection_scale": 0.9862942063603578, "optimizer_consistently_pushes_against_cap": false, "projected_updates": 12, "updates": 96}, "C-R": {"fraction_projected": 1.0, "maximum_post_sigma": 1.0262418985366821, "maximum_raw_sigma": 1.0561660528182983, "mean_projection_scale": 0.979972973301218, "minimum_projection_scale": 0.9715987405884198, "optimizer_consistently_pushes_against_cap": true, "projected_updates": 381, "updates": 381}}`.

## Decision

Exactly one recommended next experiment: **RETHINK RECURRENT STATE / READOUT REPRESENTATION**.

## Scientific questions

- **Q1:** Yes. All zero-identity checks passed: {'F_exact_zero': True, 'W_g_zero': True, 'W_u_zero': True, 'X_equals_E': True, 'alphaF_exact_zero': True, 'logits_exact': True, 'loss_exact': True, 'passed': True, 'plain_loss': 3.7945785522460938, 'real_loss': 3.7945785522460938, 'top_states_exact': True}.
- **Q2:** Yes. W_u had a nonzero finite gradient at disposable step 1 and result update 1; first result gradient report: {'finite': True, 'nonzero': True, 'norm': 0.015507043339312077, 'tensors': 1}.
- **Q3:** First nonzero W_g gradient was local update 2.
- **Q4:** Threshold crossings: {'0.01': {'additional_targets': 1048576, 'alphaF_over_E': 0.027044669002029717, 'global_update': 956, 'local_update': 2}, '0.05': {'additional_targets': 2621440, 'alphaF_over_E': 0.058766530423272526, 'global_update': 959, 'local_update': 5}, '0.1': {'additional_targets': 9437184, 'alphaF_over_E': 0.10094613313992304, 'global_update': 972, 'local_update': 18}, '0.25': {'additional_targets': 39845888, 'alphaF_over_E': 0.25314905867620296, 'global_update': 1030, 'local_update': 76}}; final alphaF/E=0.3512583284.
- **Q5:** Yes; first projection was local update 85.
- **Q6:** Maximum training X RMS=0.0382811725; final 32-pass classification=BOUNDED OSCILLATORY.
- **Q7:** At 10M/local20, recurrent gain=-0.0011535496.
- **Q8:** At 25M/local48, recurrent gain=+0.0002923844.
- **Q9:** At 50M/local96 under B12, recurrent gain=-0.0001131610.
- **Q10:** Immediate C12 shock: plain +0.0503261305, real +0.0405582364, gain +0.0097678941.
- **Q11:** At 100M/local191, recurrent gain=-0.0000204650.
- **Q12:** At 150M/local286, recurrent gain=-0.0006631066.
- **Q13:** At 250M/local477, recurrent gain=-0.0012462078.
- **Q14:** Earliest measured positive gain: 48.
- **Q15:** Earliest measured positive milestone: 48.
- **Q16:** Final real-vs-shuffled: {'losses': 0, 'mean_paired_delta': -0.008774936199188232, 'per_batch_differences': [-0.01074361801147461, -0.011263608932495117, -0.007168769836425781, -0.008930444717407227, -0.008589744567871094, -0.009594440460205078, -0.00924229621887207, -0.008831977844238281, -0.010246038436889648, -0.009512186050415039, -0.0075185298919677734, -0.0076029300689697266, -0.008738517761230469, -0.008564472198486328, -0.0067784786224365234, -0.008985280990600586, -0.007860660552978516, -0.009778261184692383, -0.00829625129699707, -0.007252216339111328], 'ties': 0, 'wins': 20}.
- **Q17:** Final real-vs-plain: {'losses': 20, 'mean_paired_delta': 0.001246201992034912, 'per_batch_differences': [0.0013327598571777344, 0.0008702278137207031, 0.0012938976287841797, 0.00012493133544921875, 0.0019481182098388672, 6.67572021484375e-06, 0.002290964126586914, 0.0009660720825195312, 0.0009584426879882812, 0.0004634857177734375, 0.0017852783203125, 0.0014355182647705078, 0.0014767646789550781, 0.0005784034729003906, 0.0020945072174072266, 0.0011391639709472656, 0.0004787445068359375, 0.0013575553894042969, 0.0014653205871582031, 0.002857208251953125], 'ties': 0, 'wins': 0}.
- **Q18:** Best final position bin 897-1023 (+0.0018614535); worst 129-256 (-0.0051328533).
- **Q19:** Late-context preference is supported by the best final bin; see position_bin_metrics.json.
- **Q20:** Plain CE changed by -0.0012490132 from source B12 to final C12 (geometry also changed).
- **Q21:** Projection summaries: {'B-R': {'updates': 96, 'projected_updates': 12, 'fraction_projected': 0.125, 'mean_projection_scale': 0.9987015562137395, 'minimum_projection_scale': 0.9862942063603578, 'maximum_raw_sigma': 1.0404261350631714, 'maximum_post_sigma': 1.0262387990951538, 'optimizer_consistently_pushes_against_cap': False}, 'C-R': {'updates': 381, 'projected_updates': 381, 'fraction_projected': 1.0, 'mean_projection_scale': 0.979972973301218, 'minimum_projection_scale': 0.9715987405884198, 'maximum_raw_sigma': 1.0561660528182983, 'maximum_post_sigma': 1.0262418985366821, 'optimizer_consistently_pushes_against_cap': True}}.
- **Q22:** Final W_u singular spectrum is stored exactly in branch_growth.json; summary: {'condition_number': 41126.080119338956, 'frobenius_norm': 1.3193581104278564, 'largest_10_singular_values': [1.0262324810028076, 0.21050749719142914, 0.14454025030136108, 0.1348557025194168, 0.12753833830356598, 0.12543494999408722, 0.12009630352258682, 0.1182553842663765, 0.114250548183918, 0.11211453378200531], 'max_column_norm': 0.06712575256824493, 'max_row_norm': 0.1912502646446228, 'mean_column_norm': 0.046809762716293335, 'mean_row_norm': 0.04187610000371933, 'mean_singular_value': 0.01923942379653454, 'minimum_singular_value': 2.4953325919341296e-05, 'shape': [768, 768], 'smallest_10_singular_values_ascending': [2.4953325919341296e-05, 3.674905019579455e-05, 4.176439324510284e-05, 4.962465027347207e-05, 6.53844399494119e-05, 8.384757529711351e-05, 8.903591515263543e-05, 9.449802018934861e-05, 0.00013704820594284683, 0.00014584395103156567], 'spectral_norm': 1.0262324810028076}.
- **Q23:** W_g first received gradient at local 2; final diagnostics: {'condition_number': 72062.25439108214, 'frobenius_norm': 11.175174713134766, 'largest_10_singular_values': [7.052809715270996, 4.998951435089111, 3.1174252033233643, 2.6567509174346924, 2.09525465965271, 1.628283143043518, 1.5047821998596191, 1.2644777297973633, 1.0928982496261597, 1.0314394235610962], 'max_column_norm': 0.8157077431678772, 'max_row_norm': 1.3761473894119263, 'mean_column_norm': 0.39243027567863464, 'mean_row_norm': 0.3687061667442322, 'mean_singular_value': 0.1428888887166977, 'minimum_singular_value': 9.787106682779267e-05, 'shape': [768, 768], 'smallest_10_singular_values_ascending': [9.787106682779267e-05, 0.00019447598606348038, 0.0003174888843204826, 0.0005541276768781245, 0.0006693262839689851, 0.0007325902115553617, 0.0008648699149489403, 0.0011203249450773, 0.0013826722279191017, 0.0014703404158353806], 'spectral_norm': 7.052809715270996}.
- **Q24:** Sequence specificity emerged before useful recurrence: True; earliest measured sequence gap>0=20, gain>0=48.
- **Q25:** The preregistered 250M classification is RESIDUAL RECURRENCE APPROACHES NEUTRALITY; final gain=-0.0012462078, gap=+0.0087749279, alphaF/E=0.3512583284.
- **Q26:** RETHINK RECURRENT STATE / READOUT REPRESENTATION

## Integrity

Scientific audit passed: `True`. Final checkpoint SHA-256: `a295c4cbf45763a84cb01c4545a86b42d17166056a6d35a170f10251a92524c4`.

# EXPERIMENT 2D1D COMPLETE
