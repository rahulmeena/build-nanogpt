# Experiment 2D11: COMPLETED

Completed 19,072 updates, 9,999,220,736 logical targets and 20,310,917,120 H CE pass-target evaluations. The mean CE pass count is 2.03125.

Cumulative billed pod time: 15.5260 hours; aggregate GPU time 62.1039 hours. Quoted total pod price $6.36/hour; estimated compute $98.75, excluding storage.

- completed_10b: **PASS**
- cuda_preflight_passed: **PASS**
- cumulative_budget_respected: **PASS**
- independent_exports_verified: **PASS**
- provider_stop_verified: **PASS**
- scientific_update_coverage: **PASS**
- terminal_cursor: **PASS**

A poor early H trajectory may reflect the complete H-from-scratch recipe, including the preregistered 50/50 initial branch mixture; this experiment does not prove that every possible H initialization would fail.

The frozen fresh base uses the original seed 1337/source initialization order and equals the mapped H base exactly. macOS ARM and Linux x86 seeded tensor generation differed in the disposable probe. Scientific training loads the immutable locally frozen tensors on every rank. No historical step-zero checkpoint is available, so historical bitwise initialization equality is not claimed.

These are single training trajectories comparing complete architecture-plus-training recipes. H uses additional multipass training compute. Four-GPU wall time versus historical one-GPU G does not establish architecture speedup.

Prespecified rules: at u=1000 stop if H CE minus historical G CE >=0.50 (skip HellaSwag in that case). At u=2000 stop if g500>=0.15 AND g1>=0.15 AND g1>=0.75*g500 AND HellaSwag advantage<=0.005. These compute-budget heuristics are not hypothesis tests.

Saved G original parallel B64 monitoring evaluator reproduction: 3.0750441551208496. Historical parallel and new incremental execution can differ numerically; intermediate historical aggregates have no paired confidence intervals.

Approximately 500M review:

```json
{
  "aggregate_gpu_hours": 6.311608218616909,
  "billed_seconds": 5680.4473967552185,
  "ce_pass_equivalent_updates": 2031,
  "ce_pass_targets": 1064828928,
  "completed_updates": 1000,
  "g_ce": 4.13529109954834,
  "g_hellaswag": 0.2550288787094204,
  "h_ce": 4.052053325275326,
  "h_hellaswag": 0.2599083847839076,
  "historical_comparison_has_paired_ci": false,
  "inputs": {
    "g500": -0.08323777427301415,
    "severe_deficit_threshold": 0.5
  },
  "interpretation": "A poor early H trajectory may reflect the complete H-from-scratch recipe, including the preregistered 50/50 initial branch mixture; this experiment does not prove that every possible H initialization would fail.",
  "logical_targets": 524288000,
  "matched_g_ce_pass_targets": 524288000,
  "mean_ce_passes_per_logical_target": 2.031,
  "outcome": "CONTINUE",
  "resources": {
    "evaluation_seconds": 667.8429408073425,
    "training_seconds": 2556.652936697006
  },
  "revised_projection": {
    "ceiling_seconds": 86400,
    "checkpoint_export_seconds": 1181.4394278292093,
    "checkpoint_export_worker_seconds": 34320.9557462609,
    "conservative_allowance_seconds": 10440.415155429924,
    "elapsed_billed_seconds": 5680.44740319252,
    "export_bandwidth_safety_fraction": 0.5,
    "export_queue_events": [
      {
        "export_finished_seconds": 2196.1191247467423,
        "file_ready_seconds": 1275.577004790306,
        "update": 1500
      },
      {
        "export_finished_seconds": 3781.1826177601242,
        "file_ready_seconds": 2860.640497803688,
        "update": 2000
      },
      {
        "export_finished_seconds": 5058.055527306023,
        "file_ready_seconds": 4137.5134073495865,
        "update": 2500
      },
      {
        "export_finished_seconds": 6434.997647500458,
        "file_ready_seconds": 5514.455527544022,
        "update": 3000
      },
      {
        "export_finished_seconds": 7711.870557046356,
        "file_ready_seconds": 6791.32843708992,
        "update": 3500
      },
      {
        "export_finished_seconds": 9090.108581996383,
        "file_ready_seconds": 8169.5664620399475,
        "update": 4000
      },
      {
        "export_finished_seconds": 10365.685586786689,
        "file_ready_seconds": 9445.143466830254,
        "update": 4500
      },
      {
        "export_finished_seconds": 11950.74907980007,
        "file_ready_seconds": 11030.206959843636,
        "update": 5000
      },
      {
        "export_finished_seconds": 13226.326084590377,
        "file_ready_seconds": 12305.783964633942,
        "update": 5500
      },
      {
        "export_finished_seconds": 14604.564109540404,
        "file_ready_seconds": 13684.02198958397,
        "update": 6000
      },
      {
        "export_finished_seconds": 15881.437019086303,
        "file_ready_seconds": 14960.894899129868,
        "update": 6500
      },
      {
        "export_finished_seconds": 17258.37913928074,
        "file_ready_seconds": 16337.837019324303,
        "update": 7000
      },
      {
        "export_finished_seconds": 18535.25204882664,
        "file_ready_seconds": 17614.7099288702,
        "update": 7500
      },
      {
        "export_finished_seconds": 19913.490073776666,
        "file_ready_seconds": 18992.94795382023,
        "update": 8000
      },
      {
        "export_finished_seconds": 21189.06707856697,
        "file_ready_seconds": 20268.524958610535,
        "update": 8500
      },
      {
        "export_finished_seconds": 22567.305103517,
        "file_ready_seconds": 21646.762983560562,
        "update": 9000
      },
      {
        "export_finished_seconds": 23842.882108307305,
        "file_ready_seconds": 22922.33998835087,
        "update": 9500
      },
      {
        "export_finished_seconds": 25427.945601320687,
        "file_ready_seconds": 24507.40348136425,
        "update": 10000
      },
      {
        "export_finished_seconds": 26704.818510866586,
        "file_ready_seconds": 25784.27639091015,
        "update": 10500
      },
      {
        "export_finished_seconds": 28081.76063106102,
        "file_ready_seconds": 27161.218511104584,
        "update": 11000
      },
      {
        "export_finished_seconds": 29358.63354060692,
        "file_ready_seconds": 28438.091420650482,
        "update": 11500
      },
      {
        "export_finished_seconds": 30736.871565556947,
        "file_ready_seconds": 29816.32944560051,
        "update": 12000
      },
      {
        "export_finished_seconds": 32012.448570347253,
        "file_ready_seconds": 31091.906450390816,
        "update": 12500
      },
      {
        "export_finished_seconds": 33390.68659529728,
        "file_ready_seconds": 32470.144475340843,
        "update": 13000
      },
      {
        "export_finished_seconds": 34666.26360008759,
        "file_ready_seconds": 33745.72148013115,
        "update": 13500
      },
      {
        "export_finished_seconds": 36044.501625037614,
        "file_ready_seconds": 35123.95950508118,
        "update": 14000
      },
      {
        "export_finished_seconds": 37321.37453458351,
        "file_ready_seconds": 36400.832414627075,
        "update": 14500
      },
      {
        "export_finished_seconds": 38905.1421228413,
        "file_ready_seconds": 37984.600002884865,
        "update": 15000
      },
      {
        "export_finished_seconds": 40182.0150323872,
        "file_ready_seconds": 39261.47291243076,
        "update": 15500
      },
      {
        "export_finished_seconds": 41560.25305733723,
        "file_ready_seconds": 40639.71093738079,
        "update": 16000
      },
      {
        "export_finished_seconds": 42835.830062127534,
        "file_ready_seconds": 41915.2879421711,
        "update": 16500
      },
      {
        "export_finished_seconds": 44214.06808707756,
        "file_ready_seconds": 43293.525967121124,
        "update": 17000
      },
      {
        "export_finished_seconds": 45489.64509186787,
        "file_ready_seconds": 44569.10297191143,
        "update": 17500
      },
      {
        "export_finished_seconds": 46867.883116817895,
        "file_ready_seconds": 45947.34099686146,
        "update": 18000
      },
      {
        "export_finished_seconds": 48144.75602636379,
        "file_ready_seconds": 47224.213906407356,
        "update": 18500
      },
      {
        "export_finished_seconds": 49521.69814655823,
        "file_ready_seconds": 48601.15602660179,
        "update": 19000
      },
      {
        "export_finished_seconds": 50702.07577714962,
        "file_ready_seconds": 49781.533657193184,
        "update": 19072
      }
    ],
    "export_tail_seconds": 920.5421199564371,
    "final_evaluation_seconds": 681.3795039653778,
    "fits": true,
    "g_four_gpu_time_is_projection": true,
    "hellaswag_seconds": 1034.1273403167725,
    "intermediate_exports_overlap_gpu_work": true,
    "monitor_ce_seconds": 1925.9371926784515,
    "projected_g_four_gpu_training_seconds": 16467.144149780273,
    "projected_remaining_gpu_hours": 67.93610103619949,
    "remaining_allowance_less_two_hour_contingency": 73519.55259680748,
    "remaining_projected_seconds": 61142.490932579545,
    "training_seconds": 45879.19231235981,
    "writes_conservatively_counted_as_blocking": true
  },
  "stop": false
}
```

Approximately 1B review:

```json
{
  "aggregate_gpu_hours": 9.387544194327461,
  "billed_seconds": 8448.789774894714,
  "ce_pass_equivalent_updates": 4062,
  "ce_pass_targets": 2129657856,
  "completed_updates": 2000,
  "g_ce": 3.6628191471099854,
  "g_hellaswag": 0.26070503883688506,
  "h_ce": 3.6106201912402804,
  "h_hellaswag": 0.26767576180043817,
  "historical_comparison_has_paired_ci": false,
  "inputs": {
    "a1": 0.006970722963553111,
    "g1": -0.05219895586970491,
    "g500": -0.08323777427301415
  },
  "interpretation": "A poor early H trajectory may reflect the complete H-from-scratch recipe, including the preregistered 50/50 initial branch mixture; this experiment does not prove that every possible H initialization would fail.",
  "logical_targets": 1048576000,
  "matched_g_ce_pass_targets": 1048576000,
  "mean_ce_passes_per_logical_target": 2.031,
  "outcome": "CONTINUE",
  "resources": {
    "evaluation_seconds": 867.9471979141235,
    "training_seconds": 5115.120496273041
  },
  "revised_projection": {
    "ceiling_seconds": 86400,
    "checkpoint_export_seconds": 1167.3368706468973,
    "checkpoint_export_worker_seconds": 32465.76894916572,
    "conservative_allowance_seconds": 9868.287055869187,
    "elapsed_billed_seconds": 8448.789782762527,
    "export_bandwidth_safety_fraction": 0.5,
    "export_queue_events": [
      {
        "export_finished_seconds": 2197.4150295023346,
        "file_ready_seconds": 1276.8729095458984,
        "update": 2500
      },
      {
        "export_finished_seconds": 3574.3571496967697,
        "file_ready_seconds": 2653.8150297403336,
        "update": 3000
      },
      {
        "export_finished_seconds": 4851.230059242668,
        "file_ready_seconds": 3930.687939286232,
        "update": 3500
      },
      {
        "export_finished_seconds": 6229.468084192696,
        "file_ready_seconds": 5308.9259642362595,
        "update": 4000
      },
      {
        "export_finished_seconds": 7505.045088983002,
        "file_ready_seconds": 6584.502969026566,
        "update": 4500
      },
      {
        "export_finished_seconds": 9090.108581996383,
        "file_ready_seconds": 8169.5664620399475,
        "update": 5000
      },
      {
        "export_finished_seconds": 10365.685586786689,
        "file_ready_seconds": 9445.143466830254,
        "update": 5500
      },
      {
        "export_finished_seconds": 11743.923611736716,
        "file_ready_seconds": 10823.381491780281,
        "update": 6000
      },
      {
        "export_finished_seconds": 13020.796521282615,
        "file_ready_seconds": 12100.25440132618,
        "update": 6500
      },
      {
        "export_finished_seconds": 14397.73864147705,
        "file_ready_seconds": 13477.196521520615,
        "update": 7000
      },
      {
        "export_finished_seconds": 15674.611551022948,
        "file_ready_seconds": 14754.069431066513,
        "update": 7500
      },
      {
        "export_finished_seconds": 17052.849575972978,
        "file_ready_seconds": 16132.30745601654,
        "update": 8000
      },
      {
        "export_finished_seconds": 18328.426580763284,
        "file_ready_seconds": 17407.884460806847,
        "update": 8500
      },
      {
        "export_finished_seconds": 19706.66460571331,
        "file_ready_seconds": 18786.122485756874,
        "update": 9000
      },
      {
        "export_finished_seconds": 20982.241610503617,
        "file_ready_seconds": 20061.69949054718,
        "update": 9500
      },
      {
        "export_finished_seconds": 22567.305103517,
        "file_ready_seconds": 21646.762983560562,
        "update": 10000
      },
      {
        "export_finished_seconds": 23844.178013062898,
        "file_ready_seconds": 22923.63589310646,
        "update": 10500
      },
      {
        "export_finished_seconds": 25221.120133257333,
        "file_ready_seconds": 24300.578013300896,
        "update": 11000
      },
      {
        "export_finished_seconds": 26497.99304280323,
        "file_ready_seconds": 25577.450922846794,
        "update": 11500
      },
      {
        "export_finished_seconds": 27876.23106775326,
        "file_ready_seconds": 26955.68894779682,
        "update": 12000
      },
      {
        "export_finished_seconds": 29151.808072543565,
        "file_ready_seconds": 28231.265952587128,
        "update": 12500
      },
      {
        "export_finished_seconds": 30530.046097493592,
        "file_ready_seconds": 29609.503977537155,
        "update": 13000
      },
      {
        "export_finished_seconds": 31805.6231022839,
        "file_ready_seconds": 30885.08098232746,
        "update": 13500
      },
      {
        "export_finished_seconds": 33183.861127233926,
        "file_ready_seconds": 32263.31900727749,
        "update": 14000
      },
      {
        "export_finished_seconds": 34460.734036779824,
        "file_ready_seconds": 33540.19191682339,
        "update": 14500
      },
      {
        "export_finished_seconds": 36044.501625037614,
        "file_ready_seconds": 35123.95950508118,
        "update": 15000
      },
      {
        "export_finished_seconds": 37321.37453458351,
        "file_ready_seconds": 36400.832414627075,
        "update": 15500
      },
      {
        "export_finished_seconds": 38699.61255953354,
        "file_ready_seconds": 37779.0704395771,
        "update": 16000
      },
      {
        "export_finished_seconds": 39975.189564323846,
        "file_ready_seconds": 39054.64744436741,
        "update": 16500
      },
      {
        "export_finished_seconds": 41353.42758927387,
        "file_ready_seconds": 40432.885469317436,
        "update": 17000
      },
      {
        "export_finished_seconds": 42629.00459406418,
        "file_ready_seconds": 41708.46247410774,
        "update": 17500
      },
      {
        "export_finished_seconds": 44007.24261901421,
        "file_ready_seconds": 43086.70049905777,
        "update": 18000
      },
      {
        "export_finished_seconds": 45284.115528560105,
        "file_ready_seconds": 44363.57340860367,
        "update": 18500
      },
      {
        "export_finished_seconds": 46661.05764875454,
        "file_ready_seconds": 45740.5155287981,
        "update": 19000
      },
      {
        "export_finished_seconds": 47841.43527934593,
        "file_ready_seconds": 46920.893159389496,
        "update": 19072
      }
    ],
    "export_tail_seconds": 920.5421199564371,
    "final_evaluation_seconds": 681.3795039653778,
    "fits": true,
    "g_four_gpu_time_is_projection": true,
    "hellaswag_seconds": 827.301872253418,
    "intermediate_exports_overlap_gpu_work": true,
    "monitor_ce_seconds": 1824.5720772743225,
    "projected_g_four_gpu_training_seconds": 16467.144149780273,
    "projected_remaining_gpu_hours": 64.12191370579458,
    "remaining_allowance_less_two_hour_contingency": 70751.21021723747,
    "remaining_projected_seconds": 57709.72233521512,
    "training_seconds": 43340.84495520592,
    "writes_conservatively_counted_as_blocking": true
  },
  "stop": false
}
```

Measured preflight and full-work projection:

```json
{
  "ceiling_seconds": 86400,
  "checkpoint_export_seconds": 1202.5932636026773,
  "checkpoint_export_worker_seconds": 37103.73594190368,
  "conservative_allowance_seconds": 11136.410696644867,
  "elapsed_billed_seconds": 2427.124544620514,
  "export_bandwidth_safety_fraction": 0.5,
  "export_queue_events": [
    {
      "export_finished_seconds": 1235.7839820150755,
      "file_ready_seconds": 315.2418620586395,
      "update": 0
    },
    {
      "export_finished_seconds": 2714.09121761364,
      "file_ready_seconds": 1793.5490976572037,
      "update": 500
    },
    {
      "export_finished_seconds": 4400.519826031151,
      "file_ready_seconds": 3479.9777060747147,
      "update": 1000
    },
    {
      "export_finished_seconds": 5676.096830821457,
      "file_ready_seconds": 4755.554710865021,
      "update": 1500
    },
    {
      "export_finished_seconds": 7261.160323834839,
      "file_ready_seconds": 6340.618203878403,
      "update": 2000
    },
    {
      "export_finished_seconds": 8538.033233380736,
      "file_ready_seconds": 7617.491113424301,
      "update": 2500
    },
    {
      "export_finished_seconds": 9914.975353575172,
      "file_ready_seconds": 8994.433233618736,
      "update": 3000
    },
    {
      "export_finished_seconds": 11191.84826312107,
      "file_ready_seconds": 10271.306143164635,
      "update": 3500
    },
    {
      "export_finished_seconds": 12570.086288071097,
      "file_ready_seconds": 11649.544168114662,
      "update": 4000
    },
    {
      "export_finished_seconds": 13845.663292861404,
      "file_ready_seconds": 12925.121172904968,
      "update": 4500
    },
    {
      "export_finished_seconds": 15430.726785874785,
      "file_ready_seconds": 14510.18466591835,
      "update": 5000
    },
    {
      "export_finished_seconds": 16706.303790665093,
      "file_ready_seconds": 15785.761670708656,
      "update": 5500
    },
    {
      "export_finished_seconds": 18084.54181561512,
      "file_ready_seconds": 17163.999695658684,
      "update": 6000
    },
    {
      "export_finished_seconds": 19361.41472516102,
      "file_ready_seconds": 18440.872605204582,
      "update": 6500
    },
    {
      "export_finished_seconds": 20738.356845355454,
      "file_ready_seconds": 19817.814725399017,
      "update": 7000
    },
    {
      "export_finished_seconds": 22015.229754901353,
      "file_ready_seconds": 21094.687634944916,
      "update": 7500
    },
    {
      "export_finished_seconds": 23393.46777985138,
      "file_ready_seconds": 22472.925659894943,
      "update": 8000
    },
    {
      "export_finished_seconds": 24669.044784641686,
      "file_ready_seconds": 23748.50266468525,
      "update": 8500
    },
    {
      "export_finished_seconds": 26047.282809591714,
      "file_ready_seconds": 25126.740689635277,
      "update": 9000
    },
    {
      "export_finished_seconds": 27322.85981438202,
      "file_ready_seconds": 26402.317694425583,
      "update": 9500
    },
    {
      "export_finished_seconds": 28907.923307395402,
      "file_ready_seconds": 27987.381187438965,
      "update": 10000
    },
    {
      "export_finished_seconds": 30184.7962169413,
      "file_ready_seconds": 29264.254096984863,
      "update": 10500
    },
    {
      "export_finished_seconds": 31561.738337135735,
      "file_ready_seconds": 30641.1962171793,
      "update": 11000
    },
    {
      "export_finished_seconds": 32838.611246681634,
      "file_ready_seconds": 31918.069126725197,
      "update": 11500
    },
    {
      "export_finished_seconds": 34216.84927163166,
      "file_ready_seconds": 33296.307151675224,
      "update": 12000
    },
    {
      "export_finished_seconds": 35492.42627642197,
      "file_ready_seconds": 34571.88415646553,
      "update": 12500
    },
    {
      "export_finished_seconds": 36870.664301371995,
      "file_ready_seconds": 35950.12218141556,
      "update": 13000
    },
    {
      "export_finished_seconds": 38146.2413061623,
      "file_ready_seconds": 37225.699186205864,
      "update": 13500
    },
    {
      "export_finished_seconds": 39524.47933111233,
      "file_ready_seconds": 38603.93721115589,
      "update": 14000
    },
    {
      "export_finished_seconds": 40801.35224065823,
      "file_ready_seconds": 39880.81012070179,
      "update": 14500
    },
    {
      "export_finished_seconds": 42385.11982891602,
      "file_ready_seconds": 41464.57770895958,
      "update": 15000
    },
    {
      "export_finished_seconds": 43661.992738461915,
      "file_ready_seconds": 42741.45061850548,
      "update": 15500
    },
    {
      "export_finished_seconds": 45040.23076341194,
      "file_ready_seconds": 44119.688643455505,
      "update": 16000
    },
    {
      "export_finished_seconds": 46315.80776820225,
      "file_ready_seconds": 45395.26564824581,
      "update": 16500
    },
    {
      "export_finished_seconds": 47694.045793152276,
      "file_ready_seconds": 46773.50367319584,
      "update": 17000
    },
    {
      "export_finished_seconds": 48969.62279794258,
      "file_ready_seconds": 48049.080677986145,
      "update": 17500
    },
    {
      "export_finished_seconds": 50347.86082289261,
      "file_ready_seconds": 49427.31870293617,
      "update": 18000
    },
    {
      "export_finished_seconds": 51624.73373243851,
      "file_ready_seconds": 50704.19161248207,
      "update": 18500
    },
    {
      "export_finished_seconds": 53001.67585263294,
      "file_ready_seconds": 52081.133732676506,
      "update": 19000
    },
    {
      "export_finished_seconds": 54182.053483224336,
      "file_ready_seconds": 53261.5113632679,
      "update": 19072
    }
  ],
  "export_tail_seconds": 920.5421199564371,
  "final_evaluation_seconds": 681.3795039653778,
  "fits": true,
  "g_four_gpu_time_is_projection": true,
  "hellaswag_seconds": 1447.7782764434814,
  "intermediate_exports_overlap_gpu_work": true,
  "monitor_ce_seconds": 2432.7627696990967,
  "projected_g_four_gpu_training_seconds": 16467.144149780273,
  "projected_remaining_gpu_hours": 72.57607131096579,
  "remaining_allowance_less_two_hour_contingency": 76772.87545537949,
  "remaining_projected_seconds": 65318.4641798692,
  "training_seconds": 48417.5396695137,
  "writes_conservatively_counted_as_blocking": true
}
```

Final paired contrasts, positive favors H:

```json
{
  "ce": {
    "estimate": 0.0236656668398704,
    "ci95": [
      0.022851540471203843,
      0.024474788396451088
    ],
    "ci97_5": [
      0.022745424544782445,
      0.024583506290689932
    ],
    "classification": "benefit",
    "unit_count": 4096,
    "resamples": 50000,
    "seed": 20260918,
    "numpy_version": "2.5.2",
    "percentile_method": "linear",
    "positive_favors": "H",
    "g_ce": 3.0386137885072637,
    "h_ce": 3.0149481216673935,
    "relative_perplexity_change_h_over_g": -0.02338783098552144,
    "h_sequence_wins": 3377,
    "ties": 0,
    "historical_small_effect_reference": 0.0001,
    "adjusted_lower_exceeds_reference": true
  },
  "hellaswag": {
    "estimate": 0.006074487153953396,
    "ci95": [
      0.00019916351324437363,
      0.011850229038040231
    ],
    "ci97_5": [
      -0.0005974905397331209,
      0.012746464847639912
    ],
    "classification": "unresolved",
    "unit_count": 10042,
    "resamples": 50000,
    "seed": 20260919,
    "numpy_version": "2.5.2",
    "percentile_method": "linear",
    "positive_favors": "H",
    "percentage_points": 0.6074487153953396,
    "paired_correctness": {
      "G0_H0": 6523,
      "G0_H1": 475,
      "G1_H0": 414,
      "G1_H1": 2630
    },
    "g_accuracy": 0.3031268671579367,
    "h_accuracy": 0.30920135431189005
  },
  "joint_claim_intervals": "Bonferroni 97.5% marginal",
  "seed_replication": false
}
```

Adjusted 97.5% marginal intervals govern endpoint classifications; 95% intervals are descriptive. The CE margin 0.0001 is a small-effect reference, not a threshold for worthwhile compute. An unresolved or harmful endpoint prevents a blanket overall-superiority claim.

See CSV/JSON and PNG/PDF artifacts in this directory. Full checkpoint files remain outside Git; manifests identify the local archive and independent persistent-volume copies. Transfer worker duration overlaps useful GPU work and is not additive billed time. Residual setup/save/failure/idle time is labeled when it cannot be separated reliably.
