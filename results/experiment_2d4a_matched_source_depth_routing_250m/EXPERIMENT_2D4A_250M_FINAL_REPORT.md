EXPERIMENT 2D4A MATCHED 250M CONTINUATION COMPLETE

PRIMARY CLASSIFICATION:
SOURCE-DEPTH ROUTING DIRECTIONALLY POSITIVE BUT NOT ESTABLISHED

Fixed 250M real CE:
3.094319574277

Routed 250M real CE:
3.094273515310

Fixed−Routed:
0.000046058967
95% CI:
[-0.000021712, 0.000114579]

Routed Route-Off CE:
3.094316187485

Route-Off−Routed:
0.000042672175
95% CI:
[-0.000026520, 0.000111169]

Routed Uniform CE:
3.094281590251

Uniform−Routed:
0.000008074941
95% CI:
[-0.000048006, 0.000063666]

Fresh evaluation: 2097152 targets per condition; 2048 paired sequences

Training: 477 total local updates and 250,085,376 total 2D4A targets per arm; 286 continuation updates and 149,946,368 continuation targets per arm

Fixed final SHA-256: 2a0f9344bc2a03e42bc3c2fb96c2e1f309f4108eaaadfa0f3e95d44e66ddecfc
Routed final SHA-256: 44e4d049f10e69757398b4aa99d4645c5114288efc26bd47e9d9f7c9306d788e

Persistent inference-state delta: 0 bytes

## 100M-to-250M interpretation

- fixed_minus_routed: grew (0.000028874543 → 0.000046058967)
- route_off_minus_routed: grew (0.000029068038 → 0.000042672175)
- uniform_minus_routed: shrunk (0.000026245677 → 0.000008074941)

## Q1–Q15

Q1. `{"at_100m": 2.887454251898191e-05, "at_250m": 4.60589669476067e-05, "change": 1.718442442862479e-05, "direction": "grew"}`
Q2. `{"at_100m": 2.9068038065281328e-05, "at_250m": 4.267217516561151e-05, "change": 1.360413710033018e-05, "direction": "grew"}`
Q3. `{"at_100m": 2.624567749672001e-05, "at_250m": 8.074941417153926e-06, "change": -1.8170736079566084e-05, "direction": "shrunk"}`
Q4. `{"b1": 0.9942113161087036, "b3": 0.999635636806488, "b5": 0.9992350935935974, "b6": 0.9991087317466736}`
Q5. `{"b1": true, "b3": true, "b5": true, "b6": true}`
Q6. `{"b3": {"route_gain": 0.00010377024849228533, "route_off_ce": 3.044077858695256}, "b6": {"route_gain": 9.791876235309971e-05, "route_off_ce": 3.0440720072091167}}`
Q7. `{"b1": true, "b5": true}`
Q8. `{"all_recurrent_memory_shuffled_ce": 3.05903586066253, "coherent_row_derangement": true, "routed_real_ce": 3.0439740884467636, "shuffled_minus_real": 0.015061772215766212}`
Q9. `{"delta_bytes": 0, "expected_each_bytes": 33288192, "fixed_bytes": 33288192, "optimizer_state_bytes": 49232, "optimizer_state_tensors": 36, "routed_bytes": 33288192, "router_parameter_bf16_bytes": 12296, "router_parameter_fp32_bytes": 24592, "router_parameters": 6148}`
Q10. `{"fixed_minus_routed": 4.60589669476067e-05, "runtime_overhead": 0.8264003039210193}`
Q11. `"SOURCE-DEPTH ROUTING DIRECTIONALLY POSITIVE BUT NOT ESTABLISHED"`
Q12. `{"backups": true, "causality": true, "fixed_286_updates": true, "fixed_cache": true, "fixed_checkpoint": true, "fixed_final_counter": true, "fixed_restart_334": true, "fixed_stability": true, "large_disjointness": true, "matched_continuation": true, "persistent_state_zero_delta": true, "preflight": true, "routed_286_updates": true, "routed_cache": true, "routed_checkpoint": true, "routed_final_counter": true, "routed_restart_334": true, "routed_stability": true}`
Q13. `{"candidate_writer_gradients": {"b1": [{"finite": true, "gradient_rms": 2.0244783627276774e-06, "nonzero": true, "source_block": 2}, {"finite": true, "gradient_rms": 1.7501519096185802e-06, "nonzero": true, "source_block": 3}, {"finite": true, "gradient_rms": 1.4719347518621362e-06, "nonzero": true, "source_block": 4}, {"finite": true, "gradient_rms": 1.3557450984080788e-06, "nonzero": true, "source_block": 5}, {"finite": true, "gradient_rms": 1.1910258308489574e-06, "nonzero": true, "source_block": 6}, {"finite": true, "gradient_rms": 1.0374060366302729e-06, "nonzero": true, "source_block": 7}, {"finite": true, "gradient_rms": 8.749605626690027e-07, "nonzero": true, "source_block": 8}, {"finite": true, "gradient_rms": 6.777060548301961e-07, "nonzero": true, "source_block": 9}, {"finite": true, "gradient_rms": 5.633244199998444e-07, "nonzero": true, "source_block": 10}, {"finite": true, "gradient_rms": 4.5806834236827854e-07, "nonzero": true, "source_block": 11}, {"finite": true, "gradient_rms": 3.316953325338545e-07, "nonzero": true, "source_block": 12}], "b3": [{"finite": true, "gradient_rms": 3.062171174406103e-07, "nonzero": true, "source_block": 4}, {"finite": true, "gradient_rms": 2.8369191795718507e-07, "nonzero": true, "source_block": 5}, {"finite": true, "gradient_rms": 2.446287510338152e-07, "nonzero": true, "source_block": 6}, {"finite": true, "gradient_rms": 2.2628506712862873e-07, "nonzero": true, "source_block": 7}, {"finite": true, "gradient_rms": 2.0204241479859775e-07, "nonzero": true, "source_block": 8}, {"finite": true, "gradient_rms": 1.60324717057847e-07, "nonzero": true, "source_block": 9}, {"finite": true, "gradient_rms": 1.1970324464982696e-07, "nonzero": true, "source_block": 10}, {"finite": true, "gradient_rms": 2.5614044218968957e-09, "nonzero": true, "source_block": 11}, {"finite": true, "gradient_rms": 1.2641339086627568e-09, "nonzero": true, "source_block": 12}], "b5": [{"finite": true, "gradient_rms": 3.496245994938363e-07, "nonzero": true, "source_block": 6}, {"finite": true, "gradient_rms": 3.41779099244377e-07, "nonzero": true, "source_block": 7}, {"finite": true, "gradient_rms": 3.0342997092702717e-07, "nonzero": true, "source_block": 8}, {"finite": true, "gradient_rms": 8.650163252355014e-09, "nonzero": true, "source_block": 9}, {"finite": true, "gradient_rms": 6.280374442724224e-09, "nonzero": true, "source_block": 10}, {"finite": true, "gradient_rms": 4.033334555231249e-09, "nonzero": true, "source_block": 11}, {"finite": true, "gradient_rms": 2.054679093532741e-09, "nonzero": true, "source_block": 12}], "b6": [{"finite": true, "gradient_rms": 1.0225594593293863e-07, "nonzero": true, "source_block": 7}, {"finite": true, "gradient_rms": 2.2028816548669283e-09, "nonzero": true, "source_block": 8}, {"finite": true, "gradient_rms": 1.6388861379112996e-09, "nonzero": true, "source_block": 9}, {"finite": true, "gradient_rms": 1.2278039696056453e-09, "nonzero": true, "source_block": 10}, {"finite": true, "gradient_rms": 7.951035940756412e-10, "nonzero": true, "source_block": 11}, {"finite": true, "gradient_rms": 3.9618294755960903e-10, "nonzero": true, "source_block": 12}]}, "destinations": {"b1": {"baseline_block": 12, "baseline_memory_rms": 2.744506359100342, "candidate_blocks": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "gamma": 0.06589433550834656, "largest_effective_block": 12, "mean_beta": [0.10558778047561646, 0.10336501896381378, 0.09460929036140442, 0.09463236480951309, 0.09326666593551636, 0.0907464474439621, 0.08933046460151672, 0.0907229632139206, 0.08994623273611069, 0.08749907463788986, 0.06029367819428444], "mean_effective_coefficients": [0.006957637146115303, 0.006811169441789389, 0.006234216503798962, 0.006235736422240734, 0.00614574458450079, 0.0059796771965920925, 0.005886371713131666, 0.005978129804134369, 0.005926947109401226, 0.005765693262219429, 0.9380786418914795], "mean_entropy": 2.38401460647583, "mean_normalized_entropy": 0.9942113161087036, "most_weighted_beta_block": 2, "norm_scale_max": 1.0077974796295166, "norm_scale_mean": 1.0003058910369873, "norm_scale_min": 0.9998295903205872, "norm_scale_rms_displacement": 0.0006658381898887455, "query_norm": 0.0851452499628067, "raw_gate": 0.06598995625972748, "routed_baseline_cosine": 0.9995114207267761, "routed_memory_rms": 2.6094863414764404}, "b3": {"baseline_block": 10, "baseline_memory_rms": 2.0583083629608154, "candidate_blocks": [4, 5, 6, 7, 8, 9, 10, 11, 12], "gamma": 0.09254033118486404, "largest_effective_block": 10, "mean_beta": [0.11634813249111176, 0.11546917259693146, 0.11399316042661667, 0.1124703586101532, 0.1104070171713829, 0.10859276354312897, 0.10745473951101303, 0.1061399057507515, 0.10912474989891052], "mean_effective_coefficients": [0.0107668936252594, 0.010685555636882782, 0.010548965074121952, 0.01040804386138916, 0.010217102244496346, 0.010049210861325264, 0.9174035787582397, 0.009822222404181957, 0.010098440572619438], "mean_entropy": 2.1964240074157715, "mean_normalized_entropy": 0.999635636806488, "most_weighted_beta_block": 4, "norm_scale_max": 1.000509262084961, "norm_scale_mean": 1.000009298324585, "norm_scale_min": 0.999933660030365, "norm_scale_rms_displacement": 4.40810872532893e-05, "query_norm": 0.02492823824286461, "raw_gate": 0.09280585497617722, "routed_baseline_cosine": 0.999727725982666, "routed_memory_rms": 1.9905567169189453}, "b5": {"baseline_block": 8, "baseline_memory_rms": 1.5023856163024902, "candidate_blocks": [6, 7, 8, 9, 10, 11, 12], "gamma": 0.04624107480049133, "largest_effective_block": 8, "mean_beta": [0.15430431067943573, 0.1490691602230072, 0.1436157524585724, 0.13873887062072754, 0.13571888208389282, 0.13363689184188843, 0.14491614699363708], "mean_effective_coefficients": [0.007135196588933468, 0.0068931179121136665, 0.960399866104126, 0.006415434647351503, 0.006275787018239498, 0.006179513409733772, 0.00670107826590538], "mean_entropy": 1.944421648979187, "mean_normalized_entropy": 0.9992350935935974, "most_weighted_beta_block": 6, "norm_scale_max": 1.002008318901062, "norm_scale_mean": 1.0000214576721191, "norm_scale_min": 0.9998862743377686, "norm_scale_rms_displacement": 0.00010137555364053696, "query_norm": 0.032382041215896606, "raw_gate": 0.04627407342195511, "routed_baseline_cosine": 0.9996826648712158, "routed_memory_rms": 1.4976751804351807}, "b6": {"baseline_block": 7, "baseline_memory_rms": 1.351029872894287, "candidate_blocks": [7, 8, 9, 10, 11, 12], "gamma": 0.020715128630399704, "largest_effective_block": 7, "mean_beta": [0.17111995816230774, 0.1687394678592682, 0.16787685453891754, 0.16830028593540192, 0.16919958591461182, 0.1547638475894928], "mean_effective_coefficients": [0.9828296899795532, 0.0034954596776515245, 0.003477590624243021, 0.0034863620530813932, 0.003504991065710783, 0.0032059531658887863], "mean_entropy": 1.7901625633239746, "mean_normalized_entropy": 0.9991087317466736, "most_weighted_beta_block": 7, "norm_scale_max": 1.0007134675979614, "norm_scale_mean": 1.0000109672546387, "norm_scale_min": 0.9999943971633911, "norm_scale_rms_displacement": 5.4459964303532615e-05, "query_norm": 0.04237928241491318, "raw_gate": 0.02071809209883213, "routed_baseline_cosine": 0.9998243451118469, "routed_memory_rms": 1.350363850593567}}, "finite": true, "gradients": {"existing_recurrent_gates": {"b1": {"connected": true, "finite": true, "nonzero": true, "norm": 0.0009729234152473509}, "b3": {"connected": true, "finite": true, "nonzero": true, "norm": 0.011473000049591064}, "b5": {"connected": true, "finite": true, "nonzero": true, "norm": 0.010296259075403214}, "b6": {"connected": true, "finite": true, "nonzero": true, "norm": 0.0013330793008208275}}, "routers": {"b1_gate": {"connected": true, "finite": true, "nonzero": true, "norm": 0.0019511185819283128}, "b1_norm": {"connected": true, "finite": true, "nonzero": true, "norm": 2.194763055740623e-06}, "b1_query": {"connected": true, "finite": true, "nonzero": true, "norm": 0.0008771576103754342}, "b3_gate": {"connected": true, "finite": true, "nonzero": true, "norm": 8.235353448071692e-07}, "b3_norm": {"connected": true, "finite": true, "nonzero": true, "norm": 3.4809988846973283e-07}, "b3_query": {"connected": true, "finite": true, "nonzero": true, "norm": 0.00031127280090004206}, "b5_gate": {"connected": true, "finite": true, "nonzero": true, "norm": 0.00034397278795950115}, "b5_norm": {"connected": true, "finite": true, "nonzero": true, "norm": 5.221705237090646e-07}, "b5_query": {"connected": true, "finite": true, "nonzero": true, "norm": 0.00032744152122177184}, "b6_gate": {"connected": true, "finite": true, "nonzero": true, "norm": 5.8186319620290305e-06}, "b6_norm": {"connected": true, "finite": true, "nonzero": true, "norm": 9.420193691767054e-08}, "b6_query": {"connected": true, "finite": true, "nonzero": true, "norm": 3.94987546314951e-05}}}, "loss": 3.4135231971740723, "validation_batch": {"combined_sha256": "c401684ed344e839c2fcb8a6113516101afa087e6ce7843713edfcb20b9cad83", "input_sha256": "6ae2081ddf8cf3ec502df81555395679c56b11437082b02e845699230167a9e6", "target_sha256": "046be4762b7b69fcd698f35cebd3ade020fcaca13e5a621f65c673365bcdef25"}}`
Q14. `{"150m": {"fixed_minus_routed": -0.0001345987408178928, "route_off_minus_routed": -1.4997087443902046e-05, "uniform_minus_routed": 6.837151945937592e-06}, "200m": {"fixed_minus_routed": -0.00014226667347116972, "route_off_minus_routed": -1.8284024398518284e-05, "uniform_minus_routed": -1.9662130282149803e-05}, "250m": null}`
Q15. `{"fixed_minus_routed": {"lower_2_5": -2.171243304528666e-05, "mean": 4.60589669476067e-05, "resamples": 50000, "seed": 20260829, "upper_97_5": 0.00011457920295187707}, "route_off_minus_routed": {"lower_2_5": -2.65198029759834e-05, "mean": 4.267217516561151e-05, "resamples": 50000, "seed": 20260829, "upper_97_5": 0.00011116878723479939}, "sequences_favoring_fixed_vs_routed": 1011, "sequences_favoring_routed_vs_fixed": 1037, "ties": 0, "uniform_minus_routed": {"lower_2_5": -4.8005934405282965e-05, "mean": 8.074941417153926e-06, "resamples": 50000, "seed": 20260829, "upper_97_5": 6.366626823040437e-05}}`

NO TRAINING BEYOND 250,085,376 2D4A TARGETS PER ARM WAS RUN.

# EXPERIMENT 2D4A MATCHED 250M CONTINUATION COMPLETE
