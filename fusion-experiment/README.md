# Fusion Experiment: Prediction-Based MDS + GDS

This experiment analyzes whether better opinion quality improves the fused
trust decision (Section VI-E of the paper). It fuses the SL opinions of two
detection systems with Subjective Logic (SL) fusion operators and measures
how well the fused opinion detects malicious messages:

- **`ai_mbd`**: opinions for the prediction-based Misbehavior Detection System
  (MDS) on VeReMi NextGen
- **`gnss`**: opinions for the GNSS Spoofing Detection System (GDS) on TEXBAT

For every message, each detection system's opinion is a binomial SL opinion
`(b, d, u)` about the proposition *"the message is benign"*, with base rate
`a = 0.5`:

| Symbol | Meaning |
|---|---|
| `b` | belief that the message is benign |
| `d` | belief that the message is malicious |
| `u` | uncertainty (`b + d + u = 1`) |
| `p = b + a·u` | projected probability that the message is benign |

Two quantification approaches (**variants**) are compared:

- `mlp`: opinions of the multi-objective multilayer perceptron (MLP) approach
- `analytic_f1`: opinions of the single-objective analytic approach (baseline,
  optimized for F1 only)

## Method

### Pairing

Pairs are built per GDS attack type (`ds3`, `ds4`, `ds7`, `ds8`):

1. Draw `N_SAMPLES = 1,000,000` gnss messages of that attack type with replacement,
   of which `ATTACK_RATE = 20 %` are malicious (`y_true == 0`) and 80 % benign.
2. Pair each gnss message with an ai_mbd message of the same label:
   - benign gnss message → random benign ai_mbd message (any attack type)
   - malicious gnss message → random malicious ai_mbd message of type
     `constantPositionOffset`, `randomPositionOffset` or `suddenStop`
3. The ground truth of a pair is the label of its gnss message.

The pairing is drawn once per seed from the `mlp` files and reused by row position for
the `analytic_f1` files. Within a seed, differences between variants and operators
therefore come only from the quantification method and the fusion operator, not from
sampling. The script checks at startup that the files of both variants are row-aligned
and aborts otherwise. The whole experiment is repeated for three seeds
(`42, 43, 44`) to estimate sampling variability.

### Fusion operators

| Operator | Description |
|---|---|
| `cbf` | Cumulative Belief Fusion |
| `abf` | Averaging Belief Fusion (dependent sources) |
| `bcf` | Belief Constraint Fusion (Dempster's rule), Jøsang (2016) Eq. 12.20 |
| `ccf` | Consensus & Compromise Fusion, Jøsang (2016) Ch. 12.6 |
| `mult` | Binomial multiplication (logical AND), Jøsang (2016) Eq. 7.1 |

### Decision rule and metrics

A pair is classified as malicious iff the projected probability of the fused opinion is
below 0.5. With *malicious* as the positive class, the script reports:

- **F1 per gnss attack type**
- **Macro-F1**: mean of the per-attack-type F1 scores
- **Overall F1**: F1 over all pairs of all attack types pooled together

## Results

Mean ± standard deviation over 3 seeds (1,000,000 pairs per attack type, 20 % attack rate):

| Operator | Macro-F1 `mlp` | Macro-F1 `analytic_f1` | Overall F1 `mlp` | Overall F1 `analytic_f1` |
|---|---|---|---|---|
| `bcf`  | 0.8704 ± 0.0003 | **0.7983 ± 0.0006** | 0.8711 ± 0.0003 | **0.7998 ± 0.0006** |
| `abf`  | **0.8705 ± 0.0003** | 0.7965 ± 0.0007 | **0.8712 ± 0.0003** | 0.7980 ± 0.0006 |
| `cbf`  | **0.8705 ± 0.0003** | 0.7965 ± 0.0007 | **0.8712 ± 0.0003** | 0.7980 ± 0.0006 |
| `ccf`  | 0.8704 ± 0.0003 | 0.7771 ± 0.0007 | **0.8712 ± 0.0003** | 0.7788 ± 0.0007 |
| `mult` | 0.6742 ± 0.0002 | 0.4454 ± 0.0001 | 0.6718 ± 0.0002 | 0.4452 ± 0.0001 |

Key observations:

- The `mlp` variant outperforms `analytic_f1` for every operator
  (about +0.07 F1 for `bcf`/`abf`/`cbf`, +0.09 for `ccf`, +0.23 for `mult`).
- With `mlp` opinions, `cbf`, `abf`, `bcf` and `ccf` perform almost identically
  (F1 ≈ 0.871). With `analytic_f1` opinions the operators differ more; `bcf` is best
  and `ccf` falls behind.
- Binomial multiplication (`mult`) is clearly unsuitable for this task.
- Standard deviations across seeds are ≤ 0.0007, so the differences above are not
  sampling noise.

## Usage

Requirements: Python ≥ 3.8, `numpy`, `pandas`.

```bash
pip install -r requirements.txt
python3 sl_fusion_eval.py
```

By default the script reads its input from a `data/` directory next to the script:

```
data/
├── ai_mbd_mlp.csv
├── gnss_mlp.csv
├── ai_mbd_analytic_f1.csv
└── gnss_analytic_f1.csv
```

Other locations can be set in `INPUT_PATHS` at the top of the script. Each file has the
columns:

```
attack, y_true, y_pred, correct, p, b, d, u
```

where `y_true == 1` means benign and `y_true == 0` means malicious. The ai_mbd files (and
likewise the gnss files) of both variants must contain the same messages in the same order.

These files are the per-sample opinion CSVs written by the final test of the
quantification sweeps (`final_test/opinions_per_sample_<method>.csv`, same columns):

| Input file | Source sweep |
|---|---|
| `ai_mbd_mlp.csv` | `quantification-approach/prediction-based_mds/multi-objective_mlp/` |
| `ai_mbd_analytic_f1.csv` | `quantification-approach/prediction-based_mds/single-objective_analytic/` |
| `gnss_mlp.csv` | `quantification-approach/gds/multi-objective_mlp/` |
| `gnss_analytic_f1.csv` | `quantification-approach/gds/single-objective_analytic/` |

### Output

Written to the script's directory:

| File | Content |
|---|---|
| `sl_fusion_results_v5_per_seed.csv` | per seed, variant and operator: F1 per attack type, macro-F1, overall F1 |
| `sl_fusion_results_mlp_v5.csv` | mean/std over seeds for the `mlp` variant |
| `sl_fusion_results_analytic_f1_v5.csv` | mean/std over seeds for the `analytic_f1` variant |
| `sl_fusion_results_summary_v5.csv` | side-by-side comparison of both variants |

## Reference

A. Jøsang, *Subjective Logic: A Formalism for Reasoning Under Uncertainty*. Springer, 2016.
