# results 目录说明

最终结果（修正 prompt + Base/Final 统一高斯模糊 sigma=1.5）：

| 模型 | T1 | T2 | B-SimplerScore |
|---|---:|---:|---:|
| Base | 2/10 | 2/10 | 20.0 |
| Final | 1/10 | 5/10 | 30.0 |

## 最终结果（根目录）

| 文件 | 内容 |
|---|---|
| `results.csv` | Base/Final × T1/T2 的成功数、成功率汇总 |
| `episodes.csv` | 40 个正式 episode 的逐条记录（模型、任务、初始状态、seed、成功、步数、视频） |
| `eval_base.log` / `eval_final.log` | Base/Final 正式评测的原始 stdout 日志 |
| `videos/` | 40 个最终评测视频（`base/`、`final/` × T1/T2） |

## 训练

| 文件 | 内容 |
|---|---|
| `train/train_log.txt` | 最终 3000 步 LoRA 训练日志（loss、action accuracy） |
| `train/val_loss.csv` | held-out val loss：Base / 3000 步 / 8000 步 |

## 分析

| 文件 | 内容 |
|---|---|
| `analysis/dev_selection.csv` | 开发集 sharp/blur、Base/Final、严格 T1、8000 步的对照 |
| `analysis/diagnostics_real_bridge.txt` | 真实 BridgeData 帧上的预测动作 vs 真实动作 |

## 归档（不属于最终结论）

| 目录 | 内容 |
|---|---|
| `archive/sharp_baseline/` | 修正 prompt、不加模糊的 sharp 基线（Base/Final 均 0/10） |
| `archive/bare_prompt/` | 最初裸 prompt 的错误结果（仅用于说明接口 bug 已修复） |
| `archive/ablations/` | 严格 T1 数据正式 T1、8000 步/image_aug/严格 T1 训练日志 |

最终结论以根目录 `results.csv` / `episodes.csv` / `videos/` 为准。
