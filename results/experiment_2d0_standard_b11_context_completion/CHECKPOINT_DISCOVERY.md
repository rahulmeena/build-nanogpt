# Experiment 2D0 Checkpoint Discovery

The preferred reproduced Karpathy-style Standard GPT-2 checkpoint was found and verified before Phase A.

- Historical checkpoint: `/workspace/build-nanogpt/runs/gpt2_124m_fineweb10b_20260810T141222Z/checkpoints/model_19072.pt`
- Historical source pod: `golden_tomato_cat`
- Persistent 2D0 copy: `/workspace/exp2d0_assets/runs/gpt2_124m_fineweb10b_20260810T141222Z/checkpoints/model_19072.pt`
- Result-bearing local-scratch copy: `/root/exp2d0_assets/model_19072.pt`
- SHA-256: `924ce6c8392c06ae24ab8f2ffd203787ee0022055c54554bac43bd9a34037871`
- Size: 497,958,271 bytes
- Historical step: 19,072
- Processed training tokens: 9,999,745,024
- Historical validation loss: 3.0750441551208496
- Architecture: Standard GPT-2, 12 blocks, width 768, 12 heads, context 1024, padded vocabulary 50,304
- Parameters: 124,475,904
- State-dict entries: 149
- Full AttnRes active modules: 0
- Full AttnRes trainable parameters: 0
- B11 mapping: `transformer.h[10]`
- B12 mapping: `transformer.h[11]`

The canonical validation shard and two disjoint training shards were also copied to the new pod and verified by SHA-256. Phase A used only the validation shard. No training shard was consumed because the preregistered window-selection rule stopped the experiment before Phase B.

