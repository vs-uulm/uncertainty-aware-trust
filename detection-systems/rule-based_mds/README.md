# Rule-Based MDS

Rule-based Misbehavior Detection System (MDS) for V2X/VANET message data. Applies
a set of plausibility and consistency checks to sender/receiver vehicle messages
(position, speed, acceleration, heading, intersection, sudden appearance) and
labels each message as benign or attacker, producing per-message predictions plus
aggregated precision/recall/F1 metrics. Used to generate labeled detection output
for downstream ML pipelines.

## Provenance

- The detection approach (plausibility/consistency checks and thresholds)
  implemented here follows:
  J. Kamel, A. Kaiser, I. Ben Jemaa, P. Cincilla, and P. Urien,
  "CaTch: A Confidence Range Tolerant Misbehavior Detection Approach,"
  2019 IEEE Wireless Communications and Networking Conference (WCNC),
  Marrakesh, Morocco, 2019.
  <https://hal.science/hal-02126960v1/document>
- The input data (V2X message scenarios with injected attacks) originates
  from the **VeReMi NextGen** dataset: <https://veremi-dataset.github.io/veremi-nextgen>.

## Files

- `main.py` — CLI entry point; discovers scenario folders, processes each
  vehicle JSON (packed in per-scenario `.zip` archives) in parallel, and
  writes per-message predictions plus an aggregated metrics JSON.
- `data_processing.py` — orchestrates the checks over a scenario's message
  DataFrame, computes confusion-matrix metrics.
- `catch_checks.py` — the individual plausibility/consistency checks (range,
  position, speed, heading, intersection, sudden appearance).
- `mdm_lib.py` — geometry/time helper functions used by the checks.
- `data_structures.py` — `Message`/`VehicleData`/`Coord` dataclasses, the
  `Parameters` dataclass (check thresholds), and JSON<->DataFrame mapping.

## Input format

`--input_folder` is expected to contain subdirectories named
`<scenario>_<density>_<attack>` (e.g. `InTAS_urban_high_positionMirroring`),
each holding one or more `.zip` archives. Each archive contains one JSON file
per vehicle with the message fields consumed by `Mapper.row_to_message`
(`sender_*`/`receiver_*` position, speed, acceleration, heading, plus
`rcvTime`, `sendTime`, `sender_id`, `attacker`, ...). Folders without at least
3 underscore-separated name parts, and anything containing `ground_truth`, are
skipped.

## Usage

```bash
pip install -r requirements.txt

python main.py \
    --input_folder /path/to/scenarios \
    --output_folder ./results \
    --workers 8
```

- `--output_folder`: where per-message JSON output and the aggregated metrics
  JSON are written (default: `./results`).
- `--workers`: number of parallel worker processes (default: CPU count).

### Check thresholds (`Parameters`)

Thresholds have built-in defaults (see `data_structures.py`). Without
`--train 1` or `--parameter`, these defaults are used and any `--mpr`/
`--msar`/... flags are ignored. There are two ways to override them:

- `--train 1` together with the individual CLI flags `--mpr`, `--msar`,
  `--mpdn`, `--mps`, `--mpa`, `--mpd`, `--mhc`, `--mdi`, `--mtd`, `--pht`,
  `--mmru`, `--mmrd`, `--msat`, `--mnrs` (see `Parameters` in
  `data_structures.py` for what each stands for). Intended for driving this
  script from an external hyperparameter search over the thresholds.
- `--parameter path/to/params.json`, containing a top-level `"parameters"`
  object with the same short keys (`mpr`, `msar`, `mpdn`, `mps`, `mpa`,
  `mpd`, `mhc`, `mdi`, `mtd`, `pht`, `mmru`, `mmrd`, `msat`, `mnrs`).

## Output

For each processed vehicle JSON, a mirrored JSON file is written under
`--output_folder` containing, per message: `sender_id`, `sender_alias`,
`messageID`, `attacker` (ground truth), and the individual check results
(`check.range_plausibility_check`, `check.position_consistency_check`, ...).
After processing, `<output_folder>/<input_folder_name>_predicted.json` holds
the aggregated confusion-matrix metrics (`tp`, `tn`, `fp`, `fn`, `accuracy`,
`precision`, `recall`, `f1`) across all processed scenarios.
