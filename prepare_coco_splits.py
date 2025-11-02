"""Split a COCO annotation file into train/val/test id lists."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split COCO annotation file")
    parser.add_argument("src", type=Path, help="Path to the source COCO JSON")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/ruler/annotations"),
        help="Directory to store split id files",
    )
    parser.add_argument("--val", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def split_ids(
    ids: List[int], val_ratio: float, test_ratio: float, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    random.Random(seed).shuffle(ids)
    n_total = len(ids)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test
    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    return train_ids, val_ids, test_ids


def write_ids(path: Path, ids: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for image_id in ids:
            fp.write(f"{image_id}\n")


def main() -> None:
    args = parse_args()
    assert args.src.exists(), f"Annotation file not found: {args.src}"

    with args.src.open("r", encoding="utf-8") as fp:
        coco = json.load(fp)

    image_ids = [image["id"] for image in coco.get("images", [])]
    if not image_ids:
        raise RuntimeError("No images found in annotation file")

    train_ids, val_ids, test_ids = split_ids(image_ids, args.val, args.test, args.seed)

    write_ids(args.out_dir / "split_train.txt", train_ids)
    write_ids(args.out_dir / "split_val.txt", val_ids)
    write_ids(args.out_dir / "split_test.txt", test_ids)

    print("Saved splits to:")
    print(args.out_dir / "split_train.txt")
    print(args.out_dir / "split_val.txt")
    print(args.out_dir / "split_test.txt")


if __name__ == "__main__":
    main()
