#!/usr/bin/env python3
"""从 HF 镜像下载 BridgeData V2 的任务相关子集。

只下载每条轨迹的：
  - data/chunk-XXX/episode_XXXXXX.parquet   (action / state / 元数据)
  - videos/chunk-XXX/observation.images.image_0/episode_XXXXXX.mp4  (主视角)

用法:
  python scripts/download_bridge_subset.py --output data/bridge_subset/raw
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO_ID = "IPEC-COMMUNITY/bridge_orig_lerobot"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
BASE_URL = f"{HF_ENDPOINT}/datasets/{REPO_ID}/resolve/main"

# 只保留与课程 T1/T2 相关、以及少量辅助任务。
# 每条轨迹按第一个命中的类别归类，避免重复下载。
CATEGORY_PATTERNS = {
    "spoon_towel": re.compile(
        r"(put|place|pick|move).*spoon.*(towel|cloth)|"
        r"(put|place|pick|move).*(towel|cloth).*spoon",
        re.I,
    ),
    "eggplant": re.compile(r"eggplant", re.I),
    "carrot_plate": re.compile(r"carrot.*plate|plate.*carrot", re.I),
    "stack_cube": re.compile(
        r"(stack|on top of|top of).*(green|yellow).*(cube|block)|"
        r"(green|yellow).*(cube|block).*(stack|on top|top of)",
        re.I,
    ),
}


def fetch(url: str, dest: Path, retries: int = 4) -> None:
    """流式下载，支持断点续传；已完整下载的文件直接跳过。"""
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            headers = {}
            mode = "wb"
            if tmp.exists() and tmp.stat().st_size > 0:
                headers["Range"] = f"bytes={tmp.stat().st_size}-"
                mode = "ab"
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            tmp.rename(dest)
            return
        except Exception as e:  # noqa: BLE001
            if attempt == retries:
                raise
            # hf-mirror 对高并发会返回 429，退避时间需要更长
            status = getattr(getattr(e, "response", None), "status_code", None)
            wait = 20 * attempt if status == 429 else 2 * attempt
            print(
                f"  [retry {attempt}/{retries}] {dest.name}: {e} (sleep {wait}s)",
                flush=True,
            )
            time.sleep(wait)


def load_metadata(output: Path) -> list[dict]:
    meta_dir = output / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = meta_dir / "episodes.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"
    fetch(f"{BASE_URL}/meta/episodes.jsonl", episodes_path)
    fetch(f"{BASE_URL}/meta/tasks.jsonl", tasks_path)
    with open(episodes_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def classify(episode: dict) -> str | None:
    for task in episode.get("tasks", []):
        for category, pattern in CATEGORY_PATTERNS.items():
            if pattern.search(task):
                return category
    return None


def select_episodes(
    episodes: list[dict], categories: list[str], max_per_category: int
) -> list[dict]:
    selected: list[dict] = []
    counts: collections.Counter = collections.Counter()
    for ep in episodes:
        category = classify(ep)
        if category is None or category not in categories:
            continue
        if counts[category] >= max_per_category:
            continue
        counts[category] += 1
        selected.append(
            {
                "episode_index": int(ep["episode_index"]),
                "length": int(ep.get("length", 0)),
                "tasks": ep.get("tasks", []),
                "category": category,
            }
        )
    print("按类别选取的轨迹数：")
    for category in categories:
        print(f"  {category}: {counts[category]}")
    return selected


def file_paths(episode_index: int) -> tuple[str, str]:
    chunk = episode_index // 1000
    parquet = f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    video = (
        f"videos/chunk-{chunk:03d}/observation.images.image_0/"
        f"episode_{episode_index:06d}.mp4"
    )
    return parquet, video


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/bridge_subset/raw")
    parser.add_argument(
        "--categories",
        default="spoon_towel,eggplant,carrot_plate,stack_cube",
        help="逗号分隔的类别名",
    )
    parser.add_argument("--max-per-category", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    episodes = load_metadata(output)
    print(f"元数据轨迹总数: {len(episodes)}")

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    unknown = set(categories) - set(CATEGORY_PATTERNS)
    if unknown:
        raise SystemExit(f"未知类别: {unknown}")

    selected = select_episodes(episodes, categories, args.max_per_category)
    manifest = output / "selected_episodes.json"
    manifest.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"清单已保存: {manifest}（共 {len(selected)} 条轨迹）")
    if args.dry_run:
        return 0

    jobs: list[tuple[str, Path]] = []
    for ep in selected:
        parquet, video = file_paths(ep["episode_index"])
        jobs.append((parquet, output / parquet))
        jobs.append((video, output / video))

    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch, f"{BASE_URL}/{rel}", dest): rel for rel, dest in jobs
        }
        for i, future in enumerate(as_completed(futures), 1):
            rel = futures[future]
            try:
                future.result()
            except Exception as e:  # noqa: BLE001
                failures.append((rel, str(e)))
                print(f"  [FAIL] {rel}: {e}", flush=True)
            if i % 50 == 0 or i == len(jobs):
                print(f"  进度 {i}/{len(jobs)}", flush=True)

    if failures:
        print(f"\n有 {len(failures)} 个文件下载失败，可重新运行本脚本续传。")
        return 1

    total = sum(p.stat().st_size for _, p in jobs if p.exists())
    print(f"完成：{len(jobs)} 个文件，共 {total / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
