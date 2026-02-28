#!/usr/bin/env python3
"""
Usage examples:

1) Default behavior (scan ./abc_dataset for .obj):
   python3 build_object_paths.py

2) Custom dataset root:
   python3 build_object_paths.py --dataset-root ./my_dataset

3) Multiple extensions:
   python3 build_object_paths.py --extensions obj stl glb

4) Custom output files:
   python3 build_object_paths.py \
     --output-pkl ./example_material/example_object_path.pkl \
     --output-txt ./example_material/example_object_path.txt

5) Limit amount of paths:
   python3 build_object_paths.py --limit 100

Arguments:
- --dataset-root
  Root directory to scan recursively for 3D files.
  Default: ./abc_dataset

- --extensions
  List of extensions to include (without dot), e.g. obj stl glb.
  Default: obj

- --output-pkl
  Output path for pickle list of file paths (used by render scripts).
  Default: ./example_material/example_object_path.pkl

- --output-txt
  Output path for human-readable text list (one path per line).
  Default: ./example_material/example_object_path.txt

- --limit
  Max number of paths to save (0 means no limit).
  Default: 0
"""

import argparse
import glob
import pickle
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build object path index files (.pkl and .txt) from a dataset directory."
    )
    parser.add_argument(
        "--dataset-root",
        default="./abc_dataset",
        help="Root directory to scan recursively (default: ./abc_dataset)",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=["obj"],
        help="File extensions to include without dot (default: obj)",
    )
    parser.add_argument(
        "--output-pkl",
        default="./example_material/example_object_path.pkl",
        help="Output .pkl path (default: ./example_material/example_object_path.pkl)",
    )
    parser.add_argument(
        "--output-txt",
        default="./example_material/example_object_path.txt",
        help="Output .txt path (default: ./example_material/example_object_path.txt)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of paths to store (0 = no limit)",
    )
    return parser.parse_args()


def normalize_ext(ext: str) -> str:
    return ext[1:] if ext.startswith(".") else ext


def collect_paths(dataset_root: str, extensions: list[str]) -> list[str]:
    paths = []
    for ext in extensions:
        pattern = f"{dataset_root}/**/*.{normalize_ext(ext)}"
        paths.extend(glob.glob(pattern, recursive=True))
    return sorted(set(paths))


def main():
    args = parse_args()
    paths = collect_paths(args.dataset_root, args.extensions)

    if args.limit > 0:
        paths = paths[: args.limit]

    if not paths:
        raise SystemExit(f"No files found in '{args.dataset_root}' for extensions: {args.extensions}")

    output_pkl = Path(args.output_pkl)
    output_txt = Path(args.output_txt)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    with output_pkl.open("wb") as f:
        pickle.dump(paths, f)

    with output_txt.open("w", encoding="utf-8") as f:
        for path in paths:
            f.write(path + "\n")

    print(f"Saved {len(paths)} paths")
    print(f"PKL: {output_pkl}")
    print(f"TXT: {output_txt}")
    print("First 5 paths:")
    for path in paths[:5]:
        print(path)


if __name__ == "__main__":
    main()
