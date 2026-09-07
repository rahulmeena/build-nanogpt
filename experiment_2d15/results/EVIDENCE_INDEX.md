# Experiment 2D15 evidence index

[Final report](FINAL_REPORT.md) · [Full paired results](ANALYSIS.json) · [Resource ledger](RESOURCE_LEDGER.json)

The verified [evidence bundle](FINAL_RAW_AND_EXECUTION_EVIDENCE.tar.gz) contains 1,015 files (20.86 MB compressed): all 660 raw batch outputs and 33 complete summaries, both complete 5000-update logs, GPU/preflight checks and failed attempts, checkpoint manifests/reopening audits, retention/export/handoff records, storage archival receipts, final analysis and stop evidence. The [bundle manifest](FINAL_EVIDENCE_BUNDLE_MANIFEST.json) records every member SHA-256 and byte size; all members were reopened and verified after compression.

Large checkpoint tensors remain in `/Users/rahul/Documents/GPT-2 Enhancement/runpod-checkpoint-archive/experiment_2d15/lr_nf4_20260907` and persistent `/workspace/exp2d15/lr_nf4_20260907` storage. [Independent export verification](FINAL_EXPORT_VERIFICATION.json) proves matching local/persistent bytes for u0/u1000/u2000/u5000 of both arms plus all raw evaluation outputs. Historical H comparators remain separately preserved.

The [optimizer implementation audit](OPTIMIZER_IMPLEMENTATION_AUDIT.json) records actual dispatch. The immutable frozen optimizer declaration predates that completed GPU audit. [Both-arm stream verification](BOTH_FULL_STREAMS_VERIFIED.json), [combined completion state](COMBINED_STATE.json) and [verified provider stop](STOP_VERIFICATION.json) establish execution closeout.

[Historical Mac archive index](HISTORICAL_ARCHIVE_INDEX.md) records the 46.19 GB reclaimed during active training. Completed 2D13/2D14 artifacts are preserved. Earlier interim/partial reports and operational fixes remain in Git history.
