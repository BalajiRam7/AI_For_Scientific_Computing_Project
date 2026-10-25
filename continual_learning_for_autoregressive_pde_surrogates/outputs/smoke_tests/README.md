# CPU smoke tests

The `advdiff1d` smoke test is intentionally separate from benchmark runs.

Reduced settings:

- two stream experiences instead of the full continual-learning stream;
- 16 snapshots per simulation instead of 100;
- context length 3 instead of 10;
- rollout and test rollout of 2 instead of 50;
- one epoch and batch size 32;
- one seed, CPU, and no pretrained checkpoint.

The runner compares all six implemented methods: `naive`, `replay`,
`replay_ts`, `gem`, `a_gem`, and `ewc`. Results are written below
`outputs/smoke_tests/advdiff1d_v2/<algorithm>/` and summarized in
`outputs/smoke_tests/advdiff1d_v2/summary.csv`. They should not be reported as
final benchmark results.
