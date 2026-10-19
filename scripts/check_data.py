#!/usr/bin/env python3
"""检查 Bridge 子集：图像、指令、动作/状态格式与夹爪约定。

用法:
  python scripts/check_data.py --raw data/bridge_subset/raw --num 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_video(path: Path) -> list[np.ndarray]:
    import av

    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/bridge_subset/raw")
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--save-dir", default="results/dev")
    args = parser.parse_args()

    import pandas as pd

    raw = Path(args.raw)
    episodes = json.loads((raw / "selected_episodes.json").read_text())
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"total episodes: {len(episodes)}")
    for ep in episodes[: args.num]:
        idx = int(ep["episode_index"])
        chunk = idx // 1000
        video_path = (
            raw
            / f"videos/chunk-{chunk:03d}/observation.images.image_0/"
            f"episode_{idx:06d}.mp4"
        )
        pq_path = raw / f"data/chunk-{chunk:03d}/episode_{idx:06d}.parquet"
        frames = read_video(video_path)
        df = pd.read_parquet(pq_path)
        actions = np.asarray(df["action"].tolist(), dtype=np.float32)
        states = np.asarray(df["observation.state"].tolist(), dtype=np.float32)
        instruction = ep["tasks"][0] if ep["tasks"] else ""

        print("-" * 70)
        print(f"episode {idx} | category={ep['category']} | instruction={instruction!r}")
        print(f"  frames={len(frames)} parquet_rows={len(df)}")
        print(
            f"  image: shape={frames[0].shape} dtype={frames[0].dtype} "
            f"mean={frames[0].mean():.1f}"
        )
        print(f"  action: shape={actions.shape} min={actions.min(axis=0)}")
        print(f"          max={actions.max(axis=0)}")
        print(f"  state:  shape={states.shape} min={states.min(axis=0)}")
        print(f"          max={states.max(axis=0)}")
        grip = actions[:, -1]
        print(
            f"  gripper action: open(>0.95)={np.mean(grip > 0.95):.2%}, "
            f"closed(<0.05)={np.mean(grip < 0.05):.2%}, "
            f"mid={np.mean((grip >= 0.05) & (grip <= 0.95)):.2%}"
        )
        if len(frames) != len(df):
            print(f"  WARNING: frame/row mismatch {len(frames)} vs {len(df)}")

        if ep is episodes[0]:
            np.save(save_dir / "sample_frame.npy", frames[0])
            pd.DataFrame(actions).to_csv(save_dir / "sample_actions.csv", index=False)
            print(f"  saved sample frame/actions to {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
