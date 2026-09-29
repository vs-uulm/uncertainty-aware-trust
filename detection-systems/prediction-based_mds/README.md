# Prediction-Based MDS

Prediction-based Misbehavior Detection System (MDS) for V2X messages. A pretrained
context-window MLP predicts a vehicle's next expected message (position, speed,
acceleration, heading, distance to road edge, receive time) from its recent
message history; the absolute error between prediction and the actually
received message is used as a per-message anomaly feature. `generate_feature_output_abs.py`
runs this model over labeled VeReMi NextGen scenarios and writes one JSON file
per vehicle with these six error features (`relative_position_error`,
`sender_speed_error`, `sender_acceleration_error`,
`distance_to_road_edge_error`, `receiver_time_error`, `sender_heading_error`)
plus the ground-truth label. These errors are the trust evidence used by the
trust quantification approaches in `quantification-approach/prediction-based_mds/`.

## Provenance

- The input data (V2X message scenarios with injected attacks) is the
  **VeReMi NextGen** dataset: <https://veremi-dataset.github.io/veremi-nextgen>.

## Files

- `generate_feature_output_abs.py` — entry point; loads the pretrained model,
  runs it over Train/Validation/Test scenarios, and writes the per-vehicle
  feature-output JSON files.
- `test_train_helper.py` — `get_model()`, used to construct the model
  architecture before loading the checkpoint's weights.
- `NN_classes/MLP_PRED_REL_POS.py` — the model architecture
  (`MLP_PRED_REL_POS`, a `pytorch_lightning` module).
- `NN_datasets/AttackType.py` — enum mapping attack names to the ids used in
  `ATTACK_TYPES`/scenario folder names.
- `NN_datasets/VeReMiNextGen.py` — loads the raw per-vehicle VeReMi NextGen
  JSON messages for a given scenario/density/attack/split, from either a
  directory or a `.zip` archive.
- `NN_datasets/feature_preparation_next_gen.py` — turns raw message history
  into fixed-length context windows (model input) and normalized target
  values (model output) used by both training and this script.
- `NN_datasets/VeReMiNextGenPredictionDatasetRelPos.py` — `torch` `Dataset`
  wrapper combining the two above for a given (scenario, density, attack,
  split) selection.
- `model/MLP_PRED_NO_RSSI_VeReMiNextGenNew_0_feature_output.pth`
  — the pretrained model checkpoint used by `generate_feature_output_abs.py`.

## Setup

```bash
pip install -r requirements.txt
```

## Input data layout

Run from a directory containing a `dataset/` folder laid out as:

```
dataset/InTAS_<scenario>_<density>_<attackName>/<Train|Validation|Test>/InTAS_<scenario>_<density>_<attackName>.zip
```

e.g. `dataset/InTAS_urban_2_reversedHeading/Train/InTAS_urban_2_reversedHeading.zip`.
Each `.zip` contains one JSON file per vehicle with its message history
(sender/receiver position, speed, acceleration, heading, timestamps, attacker
label, ...). `<attackName>` must match one of the names in
`NN_datasets/AttackType.py`.

## Usage

```bash
python generate_feature_output_abs.py
```

There are no CLI arguments; adjust the constants at the top of
`generate_feature_output_abs.py` to select what to process:

- `SCENARIOS` — e.g. `["urban", "highway"]`.
- `TRAFFIC_DENSITIES` — e.g. `[2, 7]`.
- `ATTACK_TYPES` — attack ids (see `AttackType`) to include, e.g.
  `[9, 11, 3, 4]` = `reversedHeading`, `suddenStop`, `dataReplay`, `dosAttack`.
- `CONTEXT_LENGTH` — number of preceding messages fed to the model (must
  match the checkpoint's training configuration; default `15`).
- `MODEL_PATH` — path to the pretrained checkpoint.

The script processes Train, Validation and Test splits in turn.

## Output

For each vehicle, `grouped_feature_outputs/<scenario>_<density>_<attackName>/<Train|Validation|Test>/<receiver>.json`
holds a JSON array of per-message entries:

```json
{
  "messageId": ...,
  "attacker": 0,
  "relative_position_error": 0.0123,
  "sender_speed_error": 0.0045,
  "sender_acceleration_error": 0.0089,
  "distance_to_road_edge_error": 0.0012,
  "receiver_time_error": 0.0301,
  "sender_heading_error": 0.0067
}
```
