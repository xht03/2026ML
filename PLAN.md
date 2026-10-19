# B 类项目计划：OpenVLA-7B + SimplerEnv WidowX

> 课程大作业，B 类（工程实践型）。目标：完成一次真实的 VLA 轻量微调与 SimplerEnv 闭环评测，并提交可复现材料。

## 1. 当前决策

| 项目 | 选择 | 说明 |
|---|---|---|
| 基础模型 | **OpenVLA-7B**（`openvla/openvla-7b`） | PPT 中列出的 VLA；在 BridgeData V2（WidowX）上预训练；官方支持 LoRA 微调 |
| 微调方式 | LoRA（rank 32，官方 `vla-scripts/finetune.py`） | 只更新低秩适配器，32 GB 显存可跑 |
| 训练数据 | BridgeData V2 任务相关子集（HF 镜像 `IPEC-COMMUNITY/bridge_orig_lerobot`） | 公开 WidowX 数据；1584 条轨迹；已转换为 RLDS |
| 评测环境 | SimplerEnv WidowX Visual Matching | T1 `widowx_spoon_on_towel`、T2 `widowx_put_eggplant_in_basket` |
| 评测协议 | Base / Final 各 10 次/任务，共 40 episode | 预注册 object id 0–9；开发用 10–23 |
| 对照 | Base（原始 OpenVLA-7B）vs Final（LoRA 微调后） | B 类不要求 Final 一定更好，重点是真实完整链路 |

**为什么不用 Octo**：Octo 是 2023–2024 年的 93M 轻量通用策略，和 PPT 中 RT-2/OpenVLA/π0 等十亿参数级 VLA 不在一个量级。现在 GPU 已空出完整 32 GB，OpenVLA-7B 的 LoRA 微调可以跑，因此切换到 OpenVLA。Octo 相关环境、源码、权重和专用脚本全部删除，仓库不保留死代码。

## 2. 已完成（2026-10-19）

- [x] GPU 清理：三张 RTX 5090 各 32,121 MiB 空闲。
- [x] 下载并校验 OpenVLA-7B 权重：3 个 safetensors 分片大小与官方完全一致，共 15 GB。
- [x] 创建独立环境 `venv_openvla`，安装 `torch 2.7.1+cu128`，在 5090 上跑通 bf16 矩阵乘。
- [x] 下载 BridgeData V2 子集：1584 条轨迹（3168 个文件，362 MB），图像/动作/夹爪格式已校验。
- [x] 生成 RLDS：train 1506 条、val 78 条，1.4 GB；字段为 `observation.image_0`、`observation.state`、`action`、`language_instruction`。
- [x] OpenVLA SimplerEnv 适配器已复制到 `third_party/SimplerEnv/simpler_env/policies/openvla/`，并把 `flash_attention_2` 改为 `sdpa`（5090 不兼容旧 flash-attn）。
- [x] 在 OpenVLA 代码中注册 `bridge_subset` 数据集配置与 transform（不误删第一帧）。
- [x] 安装 OpenVLA 依赖：`transformers 4.40.1`、`tokenizers 0.19.1`、`timm 0.9.10`、`peft 0.11.1`、`sapien 2.2.2`、TensorFlow 2.15 / TFDS 4.9.4 / dlimp。
- [x] OpenVLA-7B 在 RTX 5090 上加载成功（bf16，显存约 15.1 GB），`predict_action` 输出 7 维动作。
- [x] Base 单 episode 冒烟：T1 开发状态 obj id 10，跑满 60 步，14.4 秒，视频已生成；success=False，仅用于验证接口，不解读模型能力。

## 3. 待完成

### 阶段 A：环境收尾
- [x] 安装 OpenVLA 推理/微调依赖：`transformers==4.40.1`、`tokenizers==0.19.1`、`timm==0.9.10`、`peft==0.11.1`、`draccus`、`einops`、`sentencepiece==0.1.99` 等。
- [x] 安装仿真依赖：`sapien==2.2.2`、`gymnasium`、ManiSkill2_real2sim、SimplerEnv。
- [x] 安装数据依赖：`tensorflow==2.15.0`、`tensorflow_datasets==4.9.4`、`dlimp`（OpenVLA 指定 fork）。
- [x] 验证 `OpenVLAInference` 能在 5090 上加载模型并输出 7 维动作。
- **验收**：全部通过；注意 protobuf 必须锁 3.20.3，wandb 锁 0.17.9。

### 阶段 B：Base 单 episode 冒烟
- [x] 用 OpenVLA Base 跑 T1 的一个开发 episode（object id 10）。
- [x] 检查：机器人确实移动、动作尺度合理、环境原生 `info["success"]` 被记录、视频可播放。
- [x] 测量显存：模型加载约 15.1 GB，32 GB 卡余量充足。
- **结果**：60 步 / 14.4 秒 / success=False；视频与抽帧已人工检查，动作接口正常。
- **验收**：通过；Base 失败不代表接口错误。

### 阶段 C：LoRA 微调
- [x] 基于官方 `vla-scripts/finetune.py` 改写 `scripts/train_openvla.py`：
  - 数据：`--data_root_dir data/bridge_subset/tfds --dataset_name bridge_subset`
  - LoRA rank 32、target_modules=all-linear；可训练参数 110,828,288（1.45%）。
  - batch 2 × 梯度累积 4 = 等效 batch 8；固定 3000 个优化步；lr 5e-4；image_aug=False。
  - 关闭 wandb，loss 写入 `results/train/train_log.txt`。
- [x] 10 步 / 20 步 smoke：无 OOM，吞吐约 4 batch/s（batch 2）。
- [x] 正式训练完成：48 分钟，loss 约 11.87 → 3.0–3.3，action accuracy 约 0.14 → 0.20–0.25。
- [x] 保存 LoRA adapter（484 MB）、processor、`dataset_statistics.json`。
- **产物**：
  - adapter：`checkpoints/openvla_lora/adapters/openvla-7b+bridge_subset+b8+lr-0.0005+lora-r32+dropout-0.0--lora-r32-3000`
  - run 目录：`checkpoints/openvla_lora/runs/...--lora-r32-3000`
- **说明**：Final 使用训练子集统计量做反归一化；Base 使用预训练 `bridge_orig` 统计量。

### 阶段 D：正式评测
- [x] 固定 `configs/eval_states.json`：T1/T2 各 10 个 object id 0–9，robot init、env seed、model seed 全部记录。
- [x] Base 与 Final 分别用独立进程运行 `scripts/evaluate_openvla.py`。
- [x] 输出：
  - `results/episodes.csv` / `results/episodes.csv`（合并为 `results/episodes.csv`）
  - `results/results.csv`：成功数/成功率/B-SimplerScore
- [x] 40 个 episode 的视频全部保存（约 8.6 MB）。
- **结果**：
  - 协议：修正 prompt + Base/Final 统一高斯模糊 sigma=1.5（dev id 10–12 选择；formal 0–9 未参与）。
  - Base：T1 2/10，T2 2/10，B-SimplerScore 20.0
  - Final：T1 1/10，T2 5/10，B-SimplerScore 30.0
  - Final − Base = +10.0 个百分点（T1 −10，T2 +30）
  - 消融：sharp（无模糊）Base/Final 均 0/10，归档在 `results/archive/sharp_baseline/`。
- **验收**：40 个 episode 全部有记录；Base/Final 初始状态完全一致；无人工干预、无筛除。

### 阶段 E：分析与交付
- [x] 汇总成功率、平均分、Final − Base 变化（+10.0 个百分点）。
- [x] 域适应：sigma=1.5 高斯模糊（`configs/policy_preprocess.json`），sharp 基线归档。
- [x] 选 4 个代表性 episode（T1/T2 × Base/Final），结合视频分析。
- [x] README：依赖、模型/checkpoint 依据、数据来源与许可、训练/评测命令、初始状态、复现步骤。
- [x] `checkpoint_link.txt`：adapter 本地路径、Base 版本与加载方式；下载链接待上传后填写。
- [x] 10 页以内报告：`report/REPORT.pdf`（6 页，含数据样本、4 张代表帧、结果表、接口验证、案例分析与参考文献）。
- [ ] Final adapter 上传（HF/网盘）并回填链接。
- [ ] 组员分工填写。
- [ ] 创建公开代码仓库并 push（`.gitignore` 已排除大文件与第三方代码）。

### 阶段 F：T1 欠拟合诊断（已完成）

- 目的：判断 T1 低分是否因为 3000 步训练不足。
- 训练：混合数据、同配置，3000 → **8000 个优化步**（约 2 小时），新 adapter 作为候选。
- 验证：`scripts/eval_val_loss.py` 在 held-out val split 上比较 Base / 3000 步 / 8000 步。
- **结果**：
  - val loss：Base 11.16，Final3000 **3.17**，Final8000 **3.67**；
  - 8000 步训练 loss 更低（约 2.6），但验证 loss 升高，属于过拟合；
  - 8000 步 adapter 开发集：T1 1/3、T2 0/3，与 3000 步完全相同。
- **结论**：保留 3000 步 Final；T1 的瓶颈不是“训练步数不够”，继续训练会过拟合。
  8000 步 adapter 仅作为消融（`results/archive/ablations/train_log_8000.txt`、`results/train/val_loss.csv`）。

## 4. 仓库结构（清理后）

```text
MLwork/
├── PLAN.md                     # 本文件
├── README.md                   # 复现说明（阶段 E）
├── configs/
│   ├── eval_states.json        # 预注册评测初始状态
│   └── openvla_finetune.yaml   # 训练配置（阶段 C）
├── scripts/
│   ├── env.sh                  # HF 镜像、Vulkan ICD、CUDA 设备
│   ├── download_bridge_subset.py
│   ├── bridge_subset_dataset.py
│   ├── build_rlds.py
│   ├── check_data.py
│   ├── evaluate_openvla.py
│   └── train_openvla.sh
├── data/bridge_subset/         # gitignore（原始数据 + RLDS）
├── downloads/openvla-7b/       # gitignore（15 GB 权重）
├── venv_openvla/               # gitignore
├── third_party/                # gitignore
│   ├── SimplerEnv/             # 含 ManiSkill2_real2sim、openvla policy
│   └── openvla/                # 官方训练代码
├── results/
│   ├── README.md
│   ├── results.csv
│   ├── episodes.csv
│   ├── eval_base.log
│   ├── eval_final.log
│   ├── videos/
│   ├── train/
│   ├── analysis/
│   └── archive/
└── checkpoint_link.txt
```

## 5. T1 数据质量对照（已完成）

当前 Final 在 T1 上与 Base 基本持平/略低（Base 2/10，Final 1/10）。分析发现 spoon_towel 类别的
400 条轨迹中只有 188 条是真正的“勺子放到毛巾/布”，179 条是移动布/毛巾等干扰任务。

对照 B（当前执行）：
- T1 数据只保留 188 条严格目标轨迹；
- 其他类别（eggplant 400、carrot 400、stack 384）不变；
- 重新构建 RLDS（`data/bridge_subset/tfds_strict_t1`），同样 3000 步、同配置训练；
- 在开发状态 id 10–12 上评测 T1/T2；若有改善，再重跑正式评测。
- 若没有改善，则保留当前 Final，并把数据混合比例作为 T1 表现差的证据。

结果（开发状态，均加 blur sigma=1.5）：
- 严格 T1 adapter：T1 dev 1/3，T2 dev 2/3；正式 T1 **0/10**；
- 当前混合 Final：T1 dev 1/3，T2 dev 0/3；正式 T1 1/10；
- Base：T1 dev 1/3，T2 dev 1/3；正式 T1 2/10。
- T1 没有改善，正式状态甚至更差；dev 的 1/3 没有复现，说明小样本波动很大。
  数据混杂不是 T1 的主要瓶颈（任务精度/域差异也在起作用）；
  T2 的 2/3 可能是副带收益，但 3 个 episode 方差大。
- 结论：保留当前 Final（正式 B-SimplerScore 30.0）；严格 T1 adapter 仅作为数据质量消融，
  不进入正式结果。

## 6. 评测协议（固定）

- SimplerEnv commit：`eddb5690749a1e34267807c0f7b5eede323918d7`。
- T1：`PutSpoonOnTableClothInScene-v0`，scene `bridge_table_1_v1`，指令 `put the spoon on the towel`，max 60 steps。
- T2：`PutEggplantInBasketScene-v0`，scene `bridge_table_1_v2`，指令 `put eggplant into yellow basket`，max 120 steps。
- 控制频率 5 Hz，仿真频率 500 Hz，3rd_view 相机 + Visual Matching overlay。
- 初始状态：object id 0–9（正式），robot init 固定，env seed 与 model seed 分开记录。
- 成功：严格读取 `info["success"]`，不用 `done`/`terminated` 代替。
- Base/Final 使用完全相同的初始状态清单；正式评测结果不用于选 checkpoint 或调参。

## 7. 风险与退路

| 风险 | 退路 |
|---|---|
| transformers 4.40.1 与 torch 2.7.1 不兼容 | 降至 torch 2.5/2.6 的 cu128 版本；或改用社区已验证组合 |
| sdpa 输出与 flash-attn 有数值差异 | 只影响速度/少量数值，先验证 1 个 episode；必要时安装支持 sm_120 的新 flash-attn |
| 社区 SimplerEnv-OpenVLA 适配器动作接口错误 | 用真实视频调试脚本对照；先验证动作方向和尺度，再跑正式评测 |
| LoRA 微调 OOM | batch 1 + 梯度检查点 + reduce rank；32 GB 单卡应足够 |
| 显存/磁盘不足 | 训练后删除中间 checkpoint；只保留最终 adapter |
| 镜像限流 | 降低并发、断点续传；必要时用代理直连 HF |

## 8. 沟通约定

- 每完成一个阶段，更新本文件的 checkbox 和 `worklog.md`。
- 长任务（下载、训练、评测）后台运行，并定期汇报进度，不长时间沉默。
- 任何破坏性清理（删除环境/权重）先确认再执行。
