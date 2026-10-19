# 实验工作日志

详细计划与验收标准见 [`PLAN.md`](PLAN.md)。本文件只记录环境版本和操作时间线。

## 固定版本

| 组件 | 版本 / commit | 来源 |
|---|---|---|
| OpenVLA 模型 | `openvla/openvla-7b`，15 GB（3 个 safetensors 分片） | HuggingFace（hf-mirror） |
| OpenVLA 代码 | `c8f03f48af692657d3060c19588038c7220e9af9` | https://github.com/openvla/openvla |
| SimplerEnv | `eddb5690749a1e34267807c0f7b5eede323918d7` | https://github.com/simpler-env/SimplerEnv |
| ManiSkill2_real2sim | `ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682` | SimplerEnv 子模块 |
| OpenVLA SimplerEnv 适配 | `DelinQu/SimplerEnv-OpenVLA` 的 `openvla_model.py`（改为 sdpa） | 社区适配 |
| 训练数据 | BridgeData V2 子集，1584 条轨迹，362 MB | HF `IPEC-COMMUNITY/bridge_orig_lerobot` |
| RLDS | train 1506 / val 78，1.4 GB | 本地 `data/bridge_subset/tfds` |
| PyTorch | 2.7.1+cu128（已在 RTX 5090 上验证 bf16 矩阵乘） | PyTorch cu128 index |
| Python | 3.10.12（`venv_openvla`） | 系统 Python |

## 环境事实

| 项目 | 值 |
|---|---|
| GPU | 3 × RTX 5090，32 GB；2026-10-19 实验室清理后全部空闲 |
| 驱动 / CUDA | 570.153.02 / CUDA 12.8 |
| Vulkan | 必须设置 `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json` |
| 网络 | GitHub 经代理；HF 走 `hf-mirror.com` |
| 当前显存策略 | 只用 GPU 0；GPU 1/2 留空 |

## 时间线

- 2026-10-19：完成 Octo 环境与数据链路的可行性验证；后因 GPU 释放 + 课程要求，决定改用 OpenVLA-7B。
- 2026-10-19：删除 Octo 专用环境、源码、权重和脚本，仓库只保留 OpenVLA 路线。
- 2026-10-19：下载并校验 OpenVLA-7B 权重（分片大小与官方一致）。
- 2026-10-19：创建 `venv_openvla`，安装 torch 2.7.1+cu128 并通过 5090 计算测试。
- 2026-10-19：生成 Bridge 子集与 RLDS（train 1506 / val 78）。
- 2026-10-19：Base 单 episode 冒烟成功（T1 开发 id 10，60 步 / 14.4 秒，视频正常）。
- 2026-10-19：5 步 LoRA 训练冒烟成功；修正 Final 统计量嵌套格式。
- 2026-10-19 15:24–16:12：正式 LoRA 训练 3000 个优化步（batch 2 × 累积 4），48 分钟；
  loss 约 11.87 → 3.0–3.3；adapter 484 MB，保存在 `checkpoints/openvla_lora/`。
- 2026-10-19 16:14–16:33：正式评测完成。
  - sharp（无模糊、修正 prompt）：Base/Final 均 0/10，归档在 `results/archive/sharp_baseline/`。
  - 域适应：在开发状态 id 10–12 上选择 sigma=1.5 高斯模糊（Base dev 2/6、Final dev 1/6）；
    正式状态 0–9 未参与选择。
  - 最终正式结果：Base T1 2/10、T2 2/10、B-SimplerScore 20.0；
    Final T1 1/10、T2 5/10、B-SimplerScore 30.0；Final − Base = +10.0 个百分点。
  - 共 40 个 episode；最终视频 40 个（约 9 MB）；逐 episode CSV 与汇总 CSV 已生成。
- 2026-10-19：README、video_links.md、checkpoint_link.txt 已写。
- 2026-10-19：报告生成完成（python-docx → LibreOffice）。
- 2026-10-19：发现社区 OpenVLA 适配器漏掉训练 prompt 模板
  （`In: What action should the robot take to ...?\nOut:`）；修正后重跑 40 个正式 episode。
  修正前后 Base/Final 均为 0/10；旧裸 prompt 结果归档在 `results/archive/bare_prompt/`。
- 2026-10-19：真实 BridgeData 帧诊断：OpenVLA 预测动作与 GT 几乎一致，夹爪开/闭也正确，
  证明 prompt/动作解码/反归一化/图像预处理正确；仿真 0 成功归因于域差异与 LoRA 规模。
- 2026-10-19：带图像增强（image_aug=True）的第二次 LoRA 训练完成；加模糊后 dev 1/6，
  与原始 Final 持平，未采用；原始 Final adapter 保留。
- 2026-10-19：T1 数据质量对照 B：spoon_towel 400 条中仅 188 条为真正“勺子放到毛巾/布”，
  179 条为移动布/毛巾等干扰；生成严格清单（188 + 其他类别 1184 = 1372 条），
  重建 RLDS 并按同配置训练，用开发状态比较 T1。
- 2026-10-19：严格 T1 adapter 结果（blur）：dev T1 1/3（与混合 Final 持平）、T2 2/3
  （混合 Final 0/3）；正式 T1 **0/10**（混合 Final 1/10）。T1 未改善，保留当前混合 Final
  （正式 30.0）；严格 adapter 作为数据质量消融。
- 2026-10-19：启动 8000 步训练（混合数据、同配置），并新增 `scripts/eval_val_loss.py`
  用于在 held-out val split 上比较 Base / 3000 / 8000 步的 action-token loss。
- 2026-10-19：8000 步训练完成。val loss：Base 11.16、3000 步 3.17、8000 步 **3.67**；
  8000 步训练 loss 更低（约 2.6）但验证 loss 升高；开发集 T1 1/3、T2 0/3，与 3000 步相同。
  结论：继续加步数过拟合，保留 3000 步 Final；8000 步 adapter 作为消融。
- 待办：公开仓库、adapter 上传、组员分工（见 SUBMISSION.md）。
