"""把下载的 BridgeData 子集写成 TFDS/RLDS 数据集。

字段与 Octo 的 bridge_dataset 对齐：
  observation.image_0  (256,256,3) uint8
  observation.state    (8,) float32   [x,y,z,roll,pitch,yaw,pad,gripper]
  action               (7,) float32   [x,y,z,roll,pitch,yaw,gripper]
  language_instruction  string
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _decode_video(path: str) -> list[np.ndarray]:
    import av

    frames: list[np.ndarray] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def _read_episode(raw: Path, episode_index: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    import pandas as pd

    chunk = episode_index // 1000
    video_path = (
        raw
        / f"videos/chunk-{chunk:03d}/observation.images.image_0/"
        f"episode_{episode_index:06d}.mp4"
    )
    parquet_path = raw / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"

    frames = _decode_video(str(video_path))
    df = pd.read_parquet(parquet_path)
    actions = np.asarray(df["action"].tolist(), dtype=np.float32)
    states = np.asarray(df["observation.state"].tolist(), dtype=np.float32)
    return frames, actions, states


class BridgeSubset(tfds.core.GeneratorBasedBuilder):
    """BridgeData V2 任务相关子集。"""

    VERSION = tfds.core.Version("0.1.0")
    RELEASE_NOTES = {"0.1.0": "Initial release."}

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "BridgeData V2 subset used for the course project. "
                "Only the primary camera (image_0) is stored."
            ),
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image_0": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(8,), dtype=np.float32
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,), dtype=np.float32
                            ),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        raw = Path(os.environ.get("BRIDGE_RAW_DIR", "data/bridge_subset/raw"))
        manifest = os.environ.get("BRIDGE_MANIFEST", "selected_episodes.json")
        with open(raw / manifest, encoding="utf-8") as f:
            selected = json.load(f)

        # 按完整轨迹划分：每 20 条留 1 条做验证集（约 5%）。
        # 该规则与数据内容无关，避免按帧泄漏。
        train_eps = [ep for ep in selected if ep["episode_index"] % 20 != 0]
        val_eps = [ep for ep in selected if ep["episode_index"] % 20 == 0]
        if not val_eps:
            val_eps = [train_eps.pop()]

        # TFDS 4.9.4 的新 API：dict[split_name, generator]（不再用 SplitGenerator）
        return {
            "train": self._generate_examples(train_eps, str(raw)),
            "val": self._generate_examples(val_eps, str(raw)),
        }

    def _generate_examples(self, episodes: list[dict], raw: str):
        raw_path = Path(raw)
        for ep in episodes:
            episode_index = int(ep["episode_index"])
            frames, actions, states = _read_episode(raw_path, episode_index)
            n = min(len(frames), len(actions), len(states))
            if n < 2:
                continue
            instruction = ep["tasks"][0] if ep["tasks"] else ""
            steps = []
            for t in range(n):
                steps.append(
                    {
                        "observation": {
                            "image_0": frames[t],
                            "state": states[t],
                        },
                        "action": actions[t],
                        "reward": np.float32(0.0),
                        "discount": np.float32(0.0 if t == n - 1 else 1.0),
                        "is_first": t == 0,
                        "is_last": t == n - 1,
                        "is_terminal": t == n - 1,
                        "language_instruction": instruction,
                    }
                )
            yield episode_index, {"steps": steps}
