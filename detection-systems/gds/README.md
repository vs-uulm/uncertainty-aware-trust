# GDS (GNSS Spoofing Detection System)

GNSS Spoofing Detection System (GDS) on TEXBAT. Builds per-window features from GNSS
receiver tracking logs (C/N0, code/carrier lock, early-minus-late/early-plus-
late correlator ratios, code error, Doppler) and fits a strict one-class
detector against real spoofing attacks. Distances are also exported per
feature group as inputs for a downstream Subjective Logic (SL) quantification
layer.

## Provenance

- The input data are recordings from the **TEXBAT** (Texas Spoofing Test
  Battery) dataset: <https://radionavlab.ae.utexas.edu/texbat/>.

## Files

- `detector.py` — shared data loading (`load_scenario`) and feature
  windowing (`windowize`): parses per-channel tracking CSVs into fixed-length
  time windows with C/N0, lock, E/P/L and Doppler statistics, and assigns the
  clean/spoofed ground-truth label from each scenario's attack onset time.
- `detector_oneclass.py` — the actual detector: a per-feature-group
  Mahalanobis model is fit only on the first 60% of `cleanStatic`, its
  decision threshold is chosen only from the next 20% of `cleanStatic`
  (clean-only, no attack labels used anywhere in fitting), then evaluated on
  the remaining clean data plus all attack scenarios. Also produces a
  chronological train/val/test split across *all* scenarios for the
  downstream SL layer.
- `gnss_sdr_texbat_ds3.conf` — GNSS-SDR receiver configuration used to
  process the TEXBAT recordings and dump the per-channel tracking data (see
  below).
- `csv_generation.py` — converts the GNSS-SDR tracking dumps (`.mat`) into
  the per-channel CSVs read by `detector.py`.

## Setup

```bash
pip install -r requirements.txt
```

## Generating the input CSVs

`detector.py`'s `load_scenario()` reads per-channel tracking files
`trk_ch_*.csv` with `pandas.read_csv`. They contain the tracking observables
that the open-source software receiver **GNSS-SDR** (<https://gnss-sdr.org>)
records while it processes a TEXBAT raw IF recording. They are produced in
two steps:

```
TEXBAT <scenario>.bin
  └─ gnss-sdr (gnss_sdr_texbat_ds3.conf) ─→ outputs/<scenario>/tracking/trk_ch_<N>.{dat,mat}
       └─ csv_generation.py ─────────────→ csv_outputs/<scenario>/tracking/trk_ch_<N>.csv
            └─ detector_oneclass.py
```

### 1. Run GNSS-SDR with the provided configuration

`gnss_sdr_texbat_ds3.conf` is the configuration for TEXBAT scenario 3
(Static Matched-Power Time Push). It decodes the TEXBAT signal format
(complex baseband IQ at L1, 25 Msps, 16-bit interleaved I/Q → `ishort`) and
tracks GPS L1 C/A on 10 channels with a DLL/PLL tracking loop
(PLL 30 Hz, DLL 2 Hz, early–late spacing 0.5 chips). The relevant setting is
`Tracking_1C.dump=true`, which writes the per-channel tracking data.

Before running it, edit the lines marked `; <-- EDIT`:

| Setting | Meaning |
|---|---|
| `SignalSource.filename` | Path to the TEXBAT raw recording of the scenario (e.g. `ds3.bin`) |
| `Tracking_1C.dump_filename` | Prefix of the per-channel tracking dumps, default `./outputs/ds3/tracking/trk_ch_` |
| `Observables.dump_filename`, `PVT.dump_filename`, `PVT.nmea_dump_filename` | Output paths of the observables/PVT dumps (not used by the detector) |

For the other scenarios, copy the file, set `SignalSource.filename` and
replace `ds3` in the dump paths with the scenario name (`cleanStatic`, `ds4`,
`ds7`, `ds8`). 

```bash
mkdir -p outputs/ds3/tracking outputs/ds3/observables
gnss-sdr --config_file=gnss_sdr_texbat_ds3.conf
```

For every tracking channel `N`, GNSS-SDR writes a binary dump
`trk_ch_<N>.dat` and, when the receiver stops, the same data as a MATLAB
file `trk_ch_<N>.mat`.

### 2. Convert the tracking dumps to CSV

```bash
python csv_generation.py --input-dir outputs --output-dir csv_outputs
```

`csv_generation.py` reads every `outputs/<scenario>/tracking/*.mat` file and
writes one CSV per channel to

```
csv_outputs/<scenario>/tracking/trk_ch_<N>.csv
```

Each tracking variable becomes one column; the auxiliary fields `aux1`/`aux2`
are dropped. `--scenarios cleanStatic ds3 ...` converts only the listed
scenarios. The `.mat` files are MATLAB v7.3 (HDF5), which is why `h5py` is
required.

Each channel CSV holds whatever PRN that receiver slot tracked (the PRN is a
column, not the filename); confirmed header from an actual run:

```
CN0_SNV_dB_Hz,PRN,PRN_start_sample_count,Prompt_I,Prompt_Q,TOW_ms,WN,
abs_E,abs_L,abs_P,abs_VE,abs_VL,acc_carrier_phase_rad,carr_error_filt_hz,
carr_error_hz,carrier_doppler_hz,carrier_doppler_rate_hz,carrier_lock_test,
code_error_chips,code_error_filt_chips,code_freq_chips,code_freq_rate_chips
```

`load_scenario` requires `PRN`, `CN0_SNV_dB_Hz` and `PRN_start_sample_count`
(time is derived as `PRN_start_sample_count / 25e6`); `windowize` additionally
needs `abs_E`, `abs_L`, `abs_P`, `carrier_lock_test`, `code_error_chips` and
`carrier_doppler_hz`.

## Usage

```bash
python3 detector_oneclass.py
```

`detector_oneclass.py`'s `__main__` block assumes the five scenario
directories above are present under `csv_outputs/`; edit the `directories`
dict at the bottom of the script to point elsewhere or use a different
scenario subset. `detector.py` is not run directly — it's imported by
`detector_oneclass.py`.

## Output

- `oneclass_<scenario>.csv` — per-scenario windows with per-group
  Mahalanobis distances (`d_power`, `d_shape`, `d_track`, `d_dyn`), combined
  `d_total`, `p_spoof`, and the threshold decision `pred`.
- `oneclass_all.csv` — all scenarios combined (`cleanStatic` contributes only
  its held-out test portion, so no fitting data leaks in).
- `oneclass_sl_train.csv` / `oneclass_sl_val.csv` / `oneclass_sl_test.csv` —
  chronological 60/20/20 split (per scenario, stratified by label) of all
  scenarios, for training/evaluating the downstream SL layer.
