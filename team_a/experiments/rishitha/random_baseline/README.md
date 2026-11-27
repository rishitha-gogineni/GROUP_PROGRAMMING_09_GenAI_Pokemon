# Day 2 local verification - 21 November 2026

Run: `20261121T224116Z_1e244d62`.

Ten callback-based individual battles and two battles from a fresh archive extraction passed, with zero recorded invalid responses or engine rejections. Each worker also passed five contract cases: initial deck loading, single choice, multiple choice, zero options and optional selection.

- `environment.json`: actual Colab runtime and input hashes.
- `package_manifest.json`: identity and contents of the tested archive.
- `local_checks/`: ten-battle results, full report and worker exit status.
- `archive_checks/`: two-battle archive check and worker status.
- `day2_status.json` and `baseline_submission_record.csv`: unchanged snapshots created before the manual Kaggle upload.
- [Subsequent Kaggle screenshots and rating history](../../../submissions/rishitha/A-BASE/).

The pre-upload snapshots still say not_submitted; this is expected. The separate official-evidence folder shows the later Complete status. Local tests are individual random-versus-random battles, not a playing-strength benchmark or a full hosted match-harness test. Engine randomness is not verified as seeded. No simulator was rerun during repository integration.
