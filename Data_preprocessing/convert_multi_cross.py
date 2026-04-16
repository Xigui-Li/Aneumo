"""
Convert 100 cases for cross-geometry generalization experiment.
10 base geometries x 10 deforms (deform 5~14) each.

Train/test split: 80/20 per base geometry (deform 5~12 train, 13~14 test).

Usage:
    python convert_100cases_cross.py [--num-workers 16] [--skip-existing]
"""
import argparse
import multiprocessing as mp
from pathlib import Path
import time

from dataprocess import process_single_case_worker, ProcessResult
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# 10 base geos x 10 deforms (deform 5~14)
SELECTED_CASE_IDS = [
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087,
    2194, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203,
    3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287,
    4289, 4290, 4291, 4292, 4293, 4294, 4295, 4296, 4297, 4298,
    5304, 5305, 5306, 5307, 5308, 5309, 5310, 5311, 5312, 5313,
    6348, 6349, 6350, 6351, 6352, 6353, 6354, 6355, 6356, 6357,
    7360, 7361, 7362, 7363, 7364, 7365, 7366, 7367, 7368, 7369,
    8449, 8450, 8451, 8452, 8453, 8454, 8455, 8456, 8457, 8458,
    9510, 9511, 9512, 9513, 9514, 9515, 9516, 9517, 9518, 9519,
]

VTK_BASE = Path("./vtk_data")  # Update to your VTK data directory
OUTPUT_DIR = Path("./h5_multi_cross")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for cid in SELECTED_CASE_IDS:
        case_path = VTK_BASE / str(cid)
        output_file = OUTPUT_DIR / f"case_{cid}.h5"

        if not case_path.exists():
            print(f"WARNING: {case_path} does not exist, skipping")
            continue
        if args.skip_existing and output_file.exists():
            print(f"SKIP: case_{cid}.h5 already exists")
            continue

        tasks.append((f"case_{cid}", case_path, OUTPUT_DIR, 1e-6))

    print(f"{'='*60}")
    print(f"Converting {len(tasks)} cases for cross-geometry experiment")
    print(f"VTK source: {VTK_BASE}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Workers: {args.num_workers}")
    print(f"{'='*60}")

    start = time.time()
    success = 0
    fail = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_single_case_worker, t): t[0]
            for t in tasks
        }
        with tqdm(total=len(tasks), desc="Converting", unit="case") as pbar:
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    if result.success:
                        success += 1
                        pbar.write(
                            f"OK {result.case_name}: T={result.num_timesteps}, "
                            f"N={result.num_nodes}, wall={result.num_edges}, "
                            f"{result.file_size_mb:.1f}MB"
                        )
                    else:
                        fail += 1
                        pbar.write(f"FAIL {result.case_name}: {result.error_msg}")
                except Exception as e:
                    fail += 1
                    pbar.write(f"FAIL {name}: {e}")
                pbar.update(1)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min. Success={success}, Fail={fail}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
