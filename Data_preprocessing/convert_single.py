"""
Convert a single case from VTK to H5.

Usage:
    python convert_one.py <case_id> <output_dir>
"""
import sys
from pathlib import Path
from dataprocess import process_single_case_worker

VTK_BASE = Path("./vtk_data")  # Update to your VTK data directory

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <case_id> <output_dir>")
        sys.exit(1)

    cid = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    case_path = VTK_BASE / str(cid)
    if not case_path.exists():
        print(f"ERROR: {case_path} does not exist")
        sys.exit(1)

    result = process_single_case_worker((f"case_{cid}", case_path, output_dir, 1e-6))

    if result.success:
        print(f"OK {result.case_name}: T={result.num_timesteps}, "
              f"N={result.num_nodes}, E={result.num_edges}, "
              f"{result.file_size_mb:.1f}MB")
    else:
        print(f"FAIL {result.case_name}: {result.error_msg}")
        sys.exit(1)
