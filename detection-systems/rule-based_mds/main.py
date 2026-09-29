import argparse
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from data_structures import Parameters
import data_processing


def has_attack_folder(folder_name: str) -> bool:
    """
    Accept only directories like:
    InTAS_<scenario>_<density>_<attack>
    Ignore benign-only directories like:
    InTAS_urban_2
    """
    return len(folder_name.split("_")) >= 4


def evaluate_predictions(scenario_stats):
    total_tp = sum(s.get('tp', 0) for s in scenario_stats)
    total_tn = sum(s.get('tn', 0) for s in scenario_stats)
    total_fp = sum(s.get('fp', 0) for s in scenario_stats)
    total_fn = sum(s.get('fn', 0) for s in scenario_stats)

    total_messages = total_tp + total_tn + total_fp + total_fn

    aggregated_metrics = {
        'tp': total_tp,
        'tn': total_tn,
        'fp': total_fp,
        'fn': total_fn,
        'accuracy': (total_tp + total_tn) / total_messages if total_messages > 0 else 0,
        'precision': total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
        'recall': total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
    }
    aggregated_metrics['f1'] = (
        2 * aggregated_metrics['precision'] * aggregated_metrics['recall'] /
        (aggregated_metrics['precision'] + aggregated_metrics['recall'])
        if (aggregated_metrics['precision'] + aggregated_metrics['recall']) > 0 else 0
    )
    return aggregated_metrics


def process_vehicle_json_from_zip(
    zip_file: str,
    json_member: str,
    params_dict: Dict[str, Any],
    input_root: str,
    output_root: str
):
    try:
        zip_path = Path(zip_file)
        input_root = Path(input_root)
        output_root = Path(output_root)
        params = Parameters(**params_dict)

        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(json_member) as f:
                data = json.load(f)

        if not data:
            return {
                'status': 'ok',
                'metrics': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0},
                'source': f"{zip_path.name}:{json_member}"
            }

        df = pd.json_normalize(data, sep="_")
        results = data_processing.process_dataframe(df, params)

        relative_zip_path = zip_path.relative_to(input_root)
        output_dir = output_root / relative_zip_path.parent
        output_file = output_dir / Path(json_member).name

        data_processing.save_messages(results, output_file)
        metrics = data_processing.calculate_metrics(results)

        return {
            'status': 'ok',
            'metrics': metrics,
            'source': f"{zip_path.name}:{json_member}"
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'source': f"{Path(zip_file).name}:{json_member}"
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", help="Path to the input files", required=True)
    parser.add_argument("--output_folder", required=False, default="./results",
                        help="Path for the result outputs (default: ./results)")
    parser.add_argument("--train", type=float)
    parser.add_argument("--parameter", required=False, default=None)
    parser.add_argument('--mpr', required=False, type=float)
    parser.add_argument('--msar', required=False, type=float)
    parser.add_argument('--mpdn', required=False, type=float)
    parser.add_argument('--mps', required=False, type=float)
    parser.add_argument('--mpa', required=False, type=float)
    parser.add_argument('--mpd', required=False, type=float)
    parser.add_argument('--mhc', required=False, type=float)
    parser.add_argument('--mdi', required=False, type=float)
    parser.add_argument('--mtd', required=False, type=float)
    parser.add_argument('--pht', required=False, type=float)
    parser.add_argument('--mmru', required=False, type=float)
    parser.add_argument('--mmrd', required=False, type=float)
    parser.add_argument('--msat', required=False, type=float)
    parser.add_argument('--mnrs', required=False, type=float)
    parser.add_argument('--workers', required=False, type=int, default=os.cpu_count() or 4,
                        help="Number of parallel processes (default: CPU count)")
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output_root = Path(args.output_folder)
    output_root.mkdir(parents=True, exist_ok=True)

    scenario_stats = []

    if args.train == 1:
        params = Parameters(
            MAX_PLAUSIBLE_RANGE=args.mpr,
            MAX_SA_RANGE=args.msar,
            MAX_PLAUSIBLE_DIST_NEGATIVE=args.mpdn,
            MAX_PLAUSIBLE_SPEED=args.mps,
            MAX_PLAUSIBLE_ACCEL=args.mpa,
            MAX_PLAUSIBLE_DECEL=args.mpd,
            MAX_HEADING_CHANGE=args.mhc,
            MAX_DELTA_INTERSECTION=args.mdi,
            MAX_TIME_DELTA=args.mtd,
            POS_HEADING_TIME=args.pht,
            MAX_MGT_RNG_UP=args.mmru,
            MAX_MGT_RNG_DOWN=args.mmrd,
            MAX_SA_TIME=args.msat,
            MAX_NON_ROUTE_SPEED=args.mnrs
        )
    elif args.parameter is not None:
        with open(args.parameter, 'r') as f:
            data = json.load(f)

        p = data['parameters']

        params = Parameters(
            MAX_PLAUSIBLE_RANGE=p['mpr'],
            MAX_SA_RANGE=p['msar'],
            MAX_PLAUSIBLE_DIST_NEGATIVE=p['mpdn'],
            MAX_PLAUSIBLE_SPEED=p['mps'],
            MAX_PLAUSIBLE_ACCEL=p['mpa'],
            MAX_PLAUSIBLE_DECEL=p['mpd'],
            MAX_HEADING_CHANGE=p['mhc'],
            MAX_DELTA_INTERSECTION=p['mdi'],
            MAX_TIME_DELTA=p['mtd'],
            POS_HEADING_TIME=p['pht'],
            MAX_MGT_RNG_UP=p['mmru'],
            MAX_MGT_RNG_DOWN=p['mmrd'],
            MAX_SA_TIME=p['msat'],
            MAX_NON_ROUTE_SPEED=p['mnrs']
        )
    else:
        params = Parameters()

    params_dict = vars(params)
    max_workers = args.workers
    count = 0

    print(f"Starting processing with {max_workers} workers...", file=sys.stderr)

    tasks = []

    for scenario_dir in input_folder.iterdir():

        if not scenario_dir.is_dir():
            continue

        if not has_attack_folder(scenario_dir.name):
            continue

        if "ground_truth" in scenario_dir.name.lower():
            continue

        for zip_file in scenario_dir.rglob("*.zip"):
            with zipfile.ZipFile(zip_file, "r") as zf:
                json_members = [
                    name for name in zf.namelist()
                    if name.lower().endswith(".json")
                    and "ground_truth" not in Path(name).name.lower()
                ]

            for json_member in json_members:
                tasks.append((zip_file, json_member))

    total_files = len(tasks)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for zip_file, json_member in tasks:
            future = executor.submit(
                process_vehicle_json_from_zip,
                str(zip_file),
                json_member,
                params_dict,
                str(input_folder),
                str(output_root)
            )
            futures[future] = f"{zip_file.relative_to(input_folder)}::{json_member}"

        for future in as_completed(futures):
            source = futures[future]
            try:
                res = future.result()
                count += 1
                if res.get('status') == 'ok':
                    scenario_stats.append(res.get('metrics', {}))
                    print(f"Processed vehicle file {count}/{total_files}: {source}", file=sys.stderr)
                else:
                    print(f"[ERROR] {source}: {res.get('error')}", file=sys.stderr)
            except Exception as e:
                print(f"[ERROR] processing {source}: {e}", file=sys.stderr)

    aggregated_metrics = evaluate_predictions(scenario_stats)
    print(aggregated_metrics['f1'])


    results_dir = output_root
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / f"{input_folder.name}_predicted.json"
    print(f"Saved in {output_file}")
    with open(output_file, 'w') as f:
        json.dump(aggregated_metrics, f, indent=4)


if __name__ == "__main__":
    main()