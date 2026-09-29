# Runtime notes (measured)

Machine: RTX 4060 Laptop GPU (8 GB), 24-thread CPU, WSL2 Ubuntu 24.04 with a 7.6 GB memory cap
(no `.wslconfig`), Windows host with 15.7 GB RAM. TensorFlow 2.20 (GPU via WSL2).

Full run: sample 1,400,000 rows → train 857,995; CNN-LSTM 25
epochs.

## Per stage (from `outputs/metrics/stage_runtime.json`)

| Stage | Wall time | Peak RSS (process) | Min. system RAM available |
|---|---|---|---|
| prepare | 0.3 min (17 s) | 2.05 GB | 4.64 GB |
| train | 16.9 min (1016 s) | 2.50 GB | 3.74 GB |
| baselines | 17.9 min (1071 s) | 2.88 GB | 5.14 GB |
| cv | 3.9 min (232 s) | 2.19 GB | 5.65 GB |
| compare | 0.0 min (1 s) | 0.95 GB | 6.65 GB |
| edge | 0.3 min (16 s) | 1.72 GB | 5.89 GB |

"Peak RSS" is sampled every 50 ms by a background thread (`runtime.py`). The prepare stage reused
the raw-CSV sample cache (`data/original/cache/*.pkl`) if present; a cold read of the 2.3 GB CSVs adds
about 35-55 s and ~0.3 GB peak.

## Epoch time

* CNN-LSTM (GPU, 3,352 steps/epoch): 25 epochs, median 42 s,
  range 40-50 s (first epoch includes graph tracing).
* Baselines CNN / LSTM / MLP together: 66 epochs, median 10 s
  (per-model totals in the README cost table: LSTM-only is the slowest).

**CPU-only (measured on this laptop's CPU, GPU hidden):** 21.4 ms/step at batch 256 → ≈1.2 min per full epoch (3,352 steps), i.e. ≈0.5 h for 25 epochs.

## Other environments (NOT measured here - estimates)

* **Colab free (T4 GPU, ~12.7 GB RAM):** this model is tiny (≈80k parameters) so the step time is
  dominated by per-batch overhead rather than GPU FLOPs; expect roughly the same order as here
  (≈1-2 min per epoch, ~30-60 min for the whole pipeline). RAM is sufficient for the 1.4M-row sample
  with the memory-mapped cache.
* **CPU-only:** measured above on this laptop (24 threads). Because the model has only ~80k
  parameters, the GPU advantage is small here; a slower CPU (e.g. Colab's 2 vCPUs) will be several
  times slower than this measurement - start with `--sample 500000 --epochs 10` there.

## Memory: root cause of the earlier crash and the fixes

The first full run died in epoch 2 with no traceback and no Linux OOM entry: it was a background
shell started by the coding agent, which was reaped when the **Windows host** ran low on memory
(WSL VM ≈ Python 2.6 GB + ~2.5 GB page cache from reading the CSVs, plus Windows apps). The old code
path would also have peaked at ≈4.5 GB (Random Forest with 24 threads on all 858k rows while the
DataFrame, all models and raw arrays stayed alive). Fixes: detached launch (`tools/detach.sh`,
`setsid -f`), per-chunk float32 parsing and sampling, `.npy` cache read memory-mapped by every later
stage, batch feeding from the memory-mapped arrays (`ArraySequence`), Random Forest on a 300k
stratified subsample with `n_jobs=8`, CV on a 150k cached subset with `clear_session()` + `gc` per
fold, and per-epoch resumable checkpoints.

Optional `C:\Users\ayush\.wslconfig` (not applied automatically):

```ini
[wsl2]
memory=11GB          # ~70% of 15.7 GB
swap=16GB
[experimental]
autoMemoryReclaim=gradual   # return WSL page cache to Windows
```
Apply with `wsl --shutdown` from PowerShell, then reopen WSL.
