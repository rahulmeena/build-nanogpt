# 2D5C BF16 persistent inference-state accounting

| Component | Fixed logical bytes | C logical bytes | Logical reduction | Fixed physical bytes | C physical bytes | Physical reduction |
|---|---:|---:|---:|---:|---:|---:|
| B10_same_layer_local_kv | 3,142,656 | 3,142,656 | 0 | 3,142,656 | 3,142,656 | 0 |
| B11_same_layer_local_kv | 3,142,656 | 3,142,656 | 0 | 3,142,656 | 3,142,656 | 0 |
| B12_same_layer_local_kv | 3,142,656 | 3,142,656 | 0 | 3,142,656 | 3,142,656 | 0 |
| B1_same_layer_local_kv | 3,072 | 3,072 | 0 | 3,072 | 3,072 | 0 |
| B2_same_layer_local_kv | 3,142,656 | 3,142,656 | 0 | 3,142,656 | 3,142,656 | 0 |
| B3_same_layer_local_kv | 95,232 | 3,072 | 92,160 | 95,232 | 3,072 | 92,160 |
| B4_same_layer_local_kv | 3,142,656 | 3,142,656 | 0 | 3,142,656 | 3,142,656 | 0 |
| B5_same_layer_local_kv | 193,536 | 3,072 | 190,464 | 193,536 | 3,072 | 190,464 |
| B6_same_layer_local_kv | 1,569,792 | 1,569,792 | 0 | 1,569,792 | 1,569,792 | 0 |
| B7_same_layer_local_kv | 3,142,656 | 3,142,656 | 0 | 3,142,656 | 3,142,656 | 0 |
| B8_same_layer_local_kv | 3,142,656 | 3,142,656 | 0 | 3,142,656 | 3,142,656 | 0 |
| B9_same_layer_local_kv | 3,142,656 | 3,142,656 | 0 | 3,142,656 | 3,142,656 | 0 |
| recurrent_ring_h10 | 1,571,328 | 1,571,328 | 0 | 1,571,328 | 1,571,328 | 0 |
| recurrent_ring_h12 | 1,571,328 | 1,571,328 | 0 | 1,571,328 | 1,571,328 | 0 |
| recurrent_ring_h7 | 1,571,328 | 1,571,328 | 0 | 1,571,328 | 1,571,328 | 0 |
| recurrent_ring_h8 | 1,571,328 | 1,571,328 | 0 | 1,571,328 | 1,571,328 | 0 |

Logical reduction: **282,624 bytes (276 KiB)**.
Measured physical reduction: **282,624 bytes (276 KiB)**.
