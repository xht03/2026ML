#!/usr/bin/env python3
"""生成 TFDS/RLDS 数据集。

用法:
  python scripts/build_rlds.py \
    --raw data/bridge_subset/raw \
    --tfds-dir data/bridge_subset/tfds
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 允许从仓库根目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bridge_subset_dataset import BridgeSubset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/bridge_subset/raw")
    parser.add_argument("--tfds-dir", default="data/bridge_subset/tfds")
    parser.add_argument("--manifest", default="selected_episodes.json")
    args = parser.parse_args()

    raw = Path(args.raw).resolve()
    tfds_dir = Path(args.tfds_dir).resolve()
    if not (raw / args.manifest).exists():
        raise SystemExit(f"缺少 {raw}/{args.manifest}，请先生成清单")
    tfds_dir.mkdir(parents=True, exist_ok=True)
    os.environ["BRIDGE_RAW_DIR"] = str(raw)
    os.environ["BRIDGE_MANIFEST"] = args.manifest

    builder = BridgeSubset(data_dir=str(tfds_dir))
    builder.download_and_prepare()
    print(builder.info)
    print(f"数据集已生成: {builder.data_dir}")


if __name__ == "__main__":
    main()
