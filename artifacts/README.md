# Versioned hardware evidence

This directory stores curated, reviewable hardware-run evidence by project version. Generated local outputs remain
under ignored `runs/`; only completed acceptance records with explicit provenance, integrity hashes, workload scope
and limitations are promoted here.

Each version directory should contain:

- an acceptance summary that separates correctness, capacity and performance claims;
- machine-readable JSON reports and the relevant console logs;
- environment and topology evidence;
- a portable `SHA256SUMS.txt`;
- the Git commit and exact workload needed to interpret the measurements.

Large archives, model weights, checkpoints and Git bundles are intentionally kept outside Git history.
