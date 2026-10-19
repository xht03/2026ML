#!/usr/bin/env python3
"""在 SimplerEnv WidowX 上评测 OpenVLA（Base / Final）。

单独一个进程运行；初始状态来自 configs/eval_states.json。
用法:
  CUDA_VISIBLE_DEVICES=0 venv_openvla/bin/python scripts/evaluate_openvla.py \
    --model-label base --checkpoint downloads/openvla-7b --split formal
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
import traceback
from pathlib import Path

# mediapy 通过 PATH 查找 ffmpeg；venv_openvla/bin 里有链接好的二进制
os.environ["PATH"] = (
    str(Path(__file__).resolve().parents[1] / "venv_openvla" / "bin")
    + os.pathsep
    + os.environ.get("PATH", "")
)

os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("VK_DRIVER_FILES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ["DISPLAY"] = ""

import numpy as np  # noqa: E402

from simpler_env.policies.openvla.openvla_model import OpenVLAInference  # noqa: E402
from simpler_env.utils.env.env_builder import (  # noqa: E402
    build_maniskill2_env,
    get_robot_control_mode,
)
from simpler_env.utils.env.observation_utils import (  # noqa: E402
    get_image_from_maniskill2_obs_dict,
)
from simpler_env.utils.visualization import write_video  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SIMPLER_ROOT = REPO_ROOT / "third_party" / "SimplerEnv"


def run_episode(policy, task_cfg, episode_cfg, video_dir: Path):
    control_mode = get_robot_control_mode(task_cfg["robot"], "openvla")
    env = build_maniskill2_env(
        task_cfg["env_name"],
        obs_mode="rgbd",
        robot=task_cfg["robot"],
        sim_freq=task_cfg["sim_freq"],
        control_mode=control_mode,
        control_freq=task_cfg["control_freq"],
        max_episode_steps=task_cfg["max_episode_steps"],
        scene_name=task_cfg["scene_name"],
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=str(SIMPLER_ROOT / task_cfg["rgb_overlay_path"]),
    )
    try:
        random.seed(episode_cfg["env_seed"])
        np.random.seed(episode_cfg["env_seed"])
        obs, _ = env.reset(
            seed=episode_cfg["env_seed"],
            options={
                "robot_init_options": {
                    "init_xy": np.array(
                        [task_cfg["robot_init"]["x"], task_cfg["robot_init"]["y"]]
                    ),
                    "init_rot_quat": np.array(
                        task_cfg["robot_init"]["quat_center"], dtype=np.float64
                    ),
                },
                "obj_init_options": {
                    "episode_id": int(episode_cfg["obj_episode_id"])
                },
            },
        )
        instruction = env.get_language_instruction()
        policy.reset(instruction)

        image = get_image_from_maniskill2_obs_dict(env, obs)
        images = [image]
        success = False
        steps = 0
        done = truncated = False
        while not (done or truncated):
            _, action = policy.step(image, instruction)
            obs, reward, done, truncated, info = env.step(
                np.concatenate(
                    [action["world_vector"], action["rot_axangle"], action["gripper"]]
                )
            )
            # 严格使用环境原生 success，不用 done/terminated 代替
            success = bool(info["success"])
            steps += 1
            image = get_image_from_maniskill2_obs_dict(env, obs)
            images.append(image)
            if steps >= task_cfg["max_episode_steps"]:
                break

        video_path = (
            video_dir
            / task_cfg["short_name"]
            / f"episode_{episode_cfg['episode']:02d}_{'success' if success else 'failure'}.mp4"
        )
        write_video(str(video_path), images, fps=5)
        return success, steps, str(video_path), ""
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-label", required=True, choices=["base", "final"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter-path", default=None,
                        help="LoRA adapter 目录；Base 评测时不传")
    parser.add_argument("--norm-stats", default=None,
                        help="Final 评测时传入训练 run_dir 下的 dataset_statistics.json")
    parser.add_argument("--unnorm-key", default="bridge_orig")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--invert-gripper", action="store_true")
    parser.add_argument("--no-preresize", action="store_true")
    parser.add_argument("--blur-sigma", type=float, default=0.0)
    parser.add_argument("--split", default="formal", choices=["formal", "dev"])
    parser.add_argument(
        "--tasks",
        default="widowx_spoon_on_towel,widowx_put_eggplant_in_basket",
    )
    parser.add_argument("--config", default="configs/eval_states.json")
    parser.add_argument("--episodes-csv", default="results/episodes_openvla.csv")
    parser.add_argument("--summary-csv", default="results/results_openvla.csv")
    parser.add_argument("--video-dir", default="results/videos_openvla")
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    sim = cfg["simpler_env"]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    video_dir = Path(args.video_dir) / args.model_label
    episodes_csv = Path(args.episodes_csv)
    episodes_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not episodes_csv.exists()

    print(f"[{args.model_label}] loading OpenVLA {args.checkpoint} ...")
    policy = OpenVLAInference(
        saved_model_path=args.checkpoint,
        policy_setup="widowx_bridge",
        unnorm_key=args.unnorm_key,
        action_scale=args.action_scale,
        adapter_path=args.adapter_path,
        norm_stats_path=args.norm_stats,
        invert_gripper=args.invert_gripper,
        pre_resize=not args.no_preresize,
        blur_sigma=args.blur_sigma,
    )

    summary = []
    with open(episodes_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "model",
                    "task",
                    "episode",
                    "obj_episode_id",
                    "env_seed",
                    "model_seed",
                    "success",
                    "steps",
                    "instruction",
                    "error",
                    "video",
                ]
            )

        for task_name in tasks:
            task_cfg = dict(cfg["tasks"][task_name])
            task_cfg.update(
                {
                    "short_name": task_name.replace("widowx_", ""),
                    "control_freq": sim["control_freq"],
                    "sim_freq": sim["sim_freq"],
                }
            )
            episodes = task_cfg[f"{args.split}_episodes"]
            if args.max_episodes:
                episodes = episodes[: args.max_episodes]

            successes = 0
            for ep in episodes:
                t0 = time.time()
                success, steps, video, error = False, 0, "", ""
                try:
                    success, steps, video, error = run_episode(
                        policy, task_cfg, ep, video_dir
                    )
                except Exception:  # noqa: BLE001
                    error = traceback.format_exc().replace("\n", " | ")
                    print(f"[ERROR] {task_name} ep {ep['episode']}: {error}")
                successes += int(success)
                writer.writerow(
                    [
                        args.model_label,
                        task_name,
                        ep["episode"],
                        ep["obj_episode_id"],
                        ep["env_seed"],
                        ep["model_seed"],
                        int(success),
                        steps,
                        task_cfg["instruction"],
                        error,
                        video,
                    ]
                )
                f.flush()
                print(
                    f"[{args.model_label}] {task_name} ep {ep['episode']:02d} "
                    f"success={success} steps={steps} ({time.time()-t0:.1f}s)"
                )
            summary.append((task_name, successes, len(episodes)))

    print("\n=== summary ===")
    for task_name, successes, n in summary:
        print(f"{task_name}: {successes}/{n} = {100*successes/n:.1f}%")
    if len(summary) == 2:
        score = 100 * sum(s / n for _, s, n in summary) / len(summary)
        print(f"B-SimplerScore: {score:.1f}")

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_summary_header = not summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_summary_header:
            writer.writerow(["model", "task", "successes", "episodes", "success_rate"])
        for task_name, successes, n in summary:
            writer.writerow(
                [args.model_label, task_name, successes, n, f"{successes/n:.4f}"]
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
