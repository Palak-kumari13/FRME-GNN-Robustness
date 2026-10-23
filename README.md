# FRME-GNN — Stage 3

Stage 3 adds structural-perturbation consistency training (A4) and cumulative drift tests to the Stage 2 project. It is a standalone repository: A0, A1, A2, A3 and A4 are included.

## Experiments

| Run | Model | Training |
| --- | --- | --- |
| A0 | Full-edge GCN | Cross-entropy |
| A1 | Masked GCN | Cross-entropy |
| A2 | Masked GCN | Cross-entropy + frozen A0 KL + sparse mask |
| A3 | Fair masked GCN | A2 + soft demographic parity + soft equal opportunity + mask-exposure gap |
| A4 | Robust fair masked GCN | A3 + prediction consistency + shared-edge mask consistency |

No sensitive-direction projector, optimized adversarial attack, or geometric distillation is implemented yet. The cumulative drift test uses sequential random edge edits. Do **not** call it a trained adversarial attacker or a robustness certificate.

## Data and reproducibility

Use **Bail, Credit and German**, matching the original preprocessing and graph construction used for NIFTY Vanilla, NIFTY SSF, FairGNN (both run configurations) and EDITS. Datasets and seed-specific original baseline splits are **not included**.

Place `data/bail.npz`, `data/credit.npz`, `data/german.npz` with:

- `x`: `(N,F)` finite float feature matrix
- `edge_index`: `(2,E)` or `(E,2)` zero-based endpoints
- `y`: `(N,)` binary labels (0/1)
- `sensitive`: `(N,)` binary protected attributes (0/1)
- `train_idx`, `val_idx`, `test_idx`: disjoint node ID arrays

For existing per-seed splits, use `data/bail_seed0.npz`, etc. These take priority. The model treats edges as **undirected**, deduplicates them, and handles self-loops internally. All models operate on the **same input graph and splits**, using training indices for loss and validation indices for checkpoint selection. Because these are transductive graph tasks, message passing can include unlabeled test nodes; no test labels enter the loss.

`prepare_splits.py` can create **fixed stratified 60/20/20** splits from `<dataset>_raw.npz` if original splits are missing. It stratifies by joint `(label, sensitive group)` where feasible and otherwise by label. This preserves observed group/class proportions; it does not artificially rebalance the test distribution. **Do not compare newly split runs numerically to previous baselines unless you rerun those baselines using the same splits.** For original comparisons, retain seeds 0–4.

## Install and run

```bash
pip install -r requirements.txt
python train.py --dataset bail --data-dir data --output results/bail
python train.py --dataset credit --data-dir data --output results/credit
python train.py --dataset german --data-dir data --output results/german
python summarize.py results/bail/stage3_results.csv --output results/bail/summary.csv
```

For a new split, run `python prepare_splits.py --dataset bail --data-dir data --split-seed 42` first, then rerun **all** comparison methods on that split. Use different `--output` directories when changing hyperparameters; results are never silently overwritten. Training defaults to seeds 0–4.

Useful knobs:

```bash
python train.py --dataset bail --data-dir data --output results/bail_rob_off --lambda-rob 0 --lambda-mask-rob 0
python train.py --dataset bail --data-dir data --output results/bail_rob_on --lambda-rob 1 --lambda-mask-rob 0.1 --attack-budget 5 --drift-steps 5
```

## Measurements

- **Prediction:** test F1, ROC-AUC; demographic parity and equal opportunity gaps; positive prediction rate.
- **Explanation proxy:** mean mask value, fraction above 0.5, between-group mean incident-edge exposure gap.
- **Stage 3 robustness:** test prediction disagreement from clean predictions after cumulative edits; KL divergence from clean class probabilities; mean absolute mask-score change on *common edges*; perturbed test F1. Smaller disagreement/KL/mask drift is preferable **if clean utility is retained**.
- **Stochasticity:** mean ± **sample** SD over the five seeds (`summarize.py`).

All model variants see the same reproducible random-edit sequences per dataset and seed. The training-time robustness regularizer compares *only train-node probabilities* against their clean reference and compares masks on common edges. Held-out robustness evaluation scores test nodes. Attacks edit up to `--attack-budget` undirected edges per step; the cumulative graph may differ substantially after `--drift-steps` steps. These edits **do not guarantee prediction preservation** and are **not optimized worst-case attacks**.

The code records mask consistency, **not correctness**. No ground-truth explanation masks are supplied by Bail/Credit/German. Fidelity here is KL to the A0 teacher on train nodes, not proof of explanatory correctness.

## Baseline comparison requirements

Keep datasets, features, edge preprocessing, test indices and seeds identical to the earlier baseline experiments. Report NIFTY Vanilla, NIFTY SSF, FairGNN (both run sets) and EDITS alongside A0–A4. Do not compare the old `robustness` column with `drift_pred_change_final`: the definitions are not verified to match. For any direct attack comparison, rerun the earlier methods under this same edge-edit generator and report matched metrics.

## What changed from Stage 2

- `robustness.py` adds reproducible random edge edits, shared-edge mask drift and sequential drift evaluation.
- `train.py` adds A4 consistency losses and applies the same perturbation evaluation to A0–A4.
- `summarize.py` includes attack metrics.
- Fairness mask exposure is still only a proxy; group-aware geometric projections, motif correctness and teacher distillation of geometric directions remain future work.

### Stage 2 to Stage 3 implementation detail

The masked edge scorer now uses one-hop neighborhood context before scoring edges. The old Stage 2 edge scorer depended only on its endpoint feature vectors, so altering other edges could not change the score on a surviving edge. A contextual scorer is needed for a nontrivial shared-edge drift experiment. Consequently **A1–A3 are re-trained** when running this repository; do not mix earlier Stage 2 checkpoints or report its runs as strictly architecture-identical. All A0–A4 within this repository share the same scorer and protocol.
