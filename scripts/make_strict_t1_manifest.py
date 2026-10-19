#!/usr/bin/env python3
"""生成“严格 T1”训练清单：只保留真正把勺子放到毛巾/布上的轨迹。

其他类别（eggplant / carrot_plate / stack_cube）原样保留，用于 Final 同时服务 T2。
用法：
  python scripts/make_strict_t1_manifest.py \
    --raw data/bridge_subset/raw \
    --output selected_episodes_strict_t1.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def classify_spoon(task: str) -> str:
    tl = task.lower()
    # 目标行为：put/place/pick ... spoon ... (on|onto|in) ... towel/cloth
    if re.search(r"(put|place|pick).*spoon.*(on|onto|in).*(towel|cloth)", tl):
        return "target_like"
    # 更宽松但仍然是把勺子与毛巾/布关联
    if re.search(r"(put|place|pick).*spoon.*(towel|cloth)", tl):
        return "target_like"
    # 移动布/毛巾等干扰任务
    if re.search(r"(move|slide).*(cloth|towel).*spoon|(move|slide).*spoon.*(cloth|towel)", tl):
        return "cloth_moving"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/bridge_subset/raw")
    parser.add_argument("--output", default="selected_episodes_strict_t1.json")
    args = parser.parse_args()

    raw = Path(args.raw)
    episodes = json.loads((raw / "selected_episodes.json").read_text())
    strict, cloth, other = [], [], []
    kept_other_categories = []
    for ep in episodes:
        task = ep["tasks"][0] if ep.get("tasks") else ""
        if ep["category"] != "spoon_towel":
            kept_other_categories.append(ep)
            continue
        kind = classify_spoon(task)
        if kind == "target_like":
            strict.append(ep)
        elif kind == "cloth_moving":
            cloth.append(ep)
        else:
            other.append(ep)

    out = strict + kept_other_categories
    print(
        f"spoon target_like={len(strict)}, cloth_moving={len(cloth)}, other={len(other)}; "
        f"other categories kept={len(kept_other_categories)}; total={len(out)}"
    )
    (raw / args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"written: {raw / args.output}")


if __name__ == "__main__":
    main()
