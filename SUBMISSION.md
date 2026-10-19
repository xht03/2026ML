# 提交前待办与检查清单

当前结果（修正 prompt + Base/Final 统一高斯模糊 sigma=1.5）：
Base T1 2/10、T2 2/10、B-SimplerScore 20.0；Final T1 1/10、T2 5/10、B-SimplerScore 30.0；
Final − Base = +10.0 个百分点。训练 loss 约 11.9 → 3.0–3.3；
报告 `report/REPORT.pdf` 共 6 页（已更新到最终结果，含数据样本）。

补充：社区 OpenVLA 适配器最初漏掉了训练 prompt 模板，已修正并重跑。
sharp（无模糊）正式评测中 Base/Final 均为 0/10，归档在 `results/archive_sharp/`；
真实 BridgeData 帧诊断（`results/diagnostics_real_bridge.txt`）证明动作接口正确。
sigma=1.5 模糊只在开发状态 id 10–12 上选择，正式状态 0–9 未参与选择；
配置见 `configs/policy_preprocess.json`。image_aug 第二次训练在开发集上未优于原始 Final，
未采用。T1 数据质量消融：只保留 188 条严格“勺子放到毛巾/布”轨迹重新训练，
dev T1 1/3（与当前 Final 持平）、T2 2/3（当前 Final 0/3）；但正式 T1 为 0/10
（当前 Final 1/10），dev 的 1 次成功未复现。T1 未改善，因此严格 adapter
仅作为消融，不替换 Final。训练步数诊断（`results/val_loss.csv`）：val loss Base 11.16、3000 步 3.17、
8000 步 3.67；8000 步训练 loss 更低但验证 loss 升高，dev T1/T2 与 3000 步相同，
说明继续加步数会过拟合，保留 3000 步 Final。若后续更换 adapter，请同步更新结果文件、
报告和 checkpoint 链接。

## 1. 需要你填写/上传的内容

- [ ] 填写姓名、学号、组员与分工：
  - `README.md` 末尾；
  - `report/REPORT.docx` 副标题与第 8 节，然后重新生成 PDF：
    ```bash
    venv_openvla/bin/python scripts/make_report.py
    soffice -env:UserInstallation=file:///tmp/lo_docx_profile \
      --headless --norestore --convert-to pdf --outdir report report/REPORT.docx
    ```
- [ ] 创建公开代码仓库（GitHub/Gitee 等），push 本目录：
  - `.gitignore` 已排除 `third_party/`、`downloads/`、`data/`、`checkpoints/`、`venv_openvla/`、
    `results/dev/` 和助教 PPT；
  - `results/videos_openvla/`（约 8.6 MB）与 `report/REPORT.pdf` 应提交。
- [ ] 上传 Final LoRA adapter（约 484 MB）到 Hugging Face / 网盘，回填：
  - `checkpoint_link.txt`
  - `README.md` 第 9 节
  - `report/REPORT.docx` 第 8 节，并重新生成 PDF
- [ ] 确认 Base 模型链接可访问：https://huggingface.co/openvla/openvla-7b
- [ ] 确认 4 个代表视频路径可访问：见 `video_links.md`
- [ ] 可选：删除 `data/bridge_subset/raw/`（362 MB，RLDS 已生成，训练不再需要）

## 2. B 类最终检查清单（对应 pj-guide.md 第 10 节）

- [x] 所选 checkpoint 有使用 WidowX 数据训练的依据（OpenVLA 的 OXE 预训练包含 BridgeData V2）。
- [x] 数据来源、筛选方法、轨迹划分和动作归一化明确（README 第 3 节）。
- [x] 训练过可学习模块（LoRA r=32），保存配置与 loss 日志。
- [x] Final 可在独立进程加载（Base + adapter + 子集统计量），checkpoint 选择规则固定为 3000 步。
- [x] 正式评测初始状态预先固定（object id 0–9），未用于训练、调参或选 checkpoint。
- [x] Base / Final 在 T1、T2 上各完成 10 次评测（共 40 个 episode）。
- [x] 报告训练前后成功率、平均分和变化（+10.0 个百分点；sharp 消融基线为 0.0）。
- [x] 分析至少两个代表性 episode（实际分析 T1/T2 × Base/Final 四个）。
- [ ] 公开代码仓库链接（待填写）。
- [ ] checkpoint 有效下载链接（待上传）。
- [x] 逐 episode 结果：`results/episodes.csv`、`results/episodes_openvla*.csv`。
- [x] 2–4 个代表视频：见 `video_links.md`（实际保留全部 40 个视频）。
- [x] 报告不超过 10 页（当前 6 页，含数据样本、参考文献和附录）。
- [ ] 报告中的姓名/组员/分工/仓库/checkpoint 链接（待填写）。

## 3. 复现命令（简版）

```bash
# 数据（已生成，可跳过）
python scripts/download_bridge_subset.py --output data/bridge_subset/raw
python scripts/build_rlds.py --raw data/bridge_subset/raw --tfds-dir data/bridge_subset/tfds

# 训练（约 48 分钟）
source scripts/env.sh
WANDB_MODE=disabled venv_openvla/bin/torchrun --standalone --nnodes 1 --nproc-per-node 1 \
  scripts/train_openvla.py --vla_path downloads/openvla-7b \
  --data_root_dir data/bridge_subset/tfds --dataset_name bridge_subset \
  --run_root_dir checkpoints/openvla_lora/runs \
  --adapter_tmp_dir checkpoints/openvla_lora/adapters \
  --batch_size 2 --grad_accumulation_steps 4 --max_steps 3000 --save_steps 3000 \
  --learning_rate 5e-4 --lora_rank 32 --image_aug False --shuffle_buffer_size 100 \
  --run_id_note lora-r32-3000 > results/train_log.txt 2>&1

# Base 评测
venv_openvla/bin/python scripts/evaluate_openvla.py --model-label base \
  --checkpoint downloads/openvla-7b --split formal --blur-sigma 1.5

# Final 评测（替换为实际路径）
AD=checkpoints/openvla_lora/adapters/<adapter_dir>
RUN=checkpoints/openvla_lora/runs/<run_dir>
venv_openvla/bin/python scripts/evaluate_openvla.py --model-label final \
  --checkpoint downloads/openvla-7b --adapter-path "$AD" \
  --norm-stats "$RUN/dataset_statistics.json" --split formal --blur-sigma 1.5 \
  --episodes-csv results/episodes_openvla_final.csv \
  --summary-csv results/results_openvla_final.csv
```

## 4. 注意事项

- Final 是 LoRA adapter，不是合并后的 15 GB 全模型；提交时必须在报告/README 中注明 Base 版本和加载方式。
- 不要提交 `third_party/`、`downloads/`、`data/`、`checkpoints/`、`venv_openvla/`；它们已在大文件或第三方代码范围内。
- 共享服务器磁盘只剩约 16 GB；如需重新合并全模型，先申请空间或移到数据盘。
- GPU 1/2 在管理员清理进程后 CUDA context 异常；评测使用 GPU 0，协议不受影响。
