# 代表性视频

全部 40 个最终评测 episode 的视频位于 `results/videos/{base,final}/{task}/`，约 9 MB。
最终评测使用：修正 prompt + 高斯模糊 sigma=1.5（Base/Final 相同）。

## 报告引用的 4 个代表视频

| 文件 | 模型 | 任务 | 结果 | 观察 |
|---|---|---|---|---|
| `results/videos/base/spoon_on_towel/episode_03_success.mp4` | Base | T1 | success（31 步） | 抓起勺子并放到毛巾上 |
| `results/videos/base/put_eggplant_in_basket/episode_02_success.mp4` | Base | T2 | success（23 步） | 抓起茄子放入篮子 |
| `results/videos/final/put_eggplant_in_basket/episode_00_success.mp4` | Final | T2 | success（16 步） | 快速抓起茄子放入篮子 |
| `results/videos/final/spoon_on_towel/episode_00_failure.mp4` | Final | T1 | failure | 接近勺子但未稳定抓取/放置 |

## 归档

- `results/archive/sharp_baseline/`：修正 prompt、**不加模糊**的 sharp 基线逐 episode CSV 与日志
  （正式评测 0/10；重复视频已删除以精简仓库）。
- `results/archive/bare_prompt/`：裸 prompt 的逐 episode CSV（接口错误版本，不用于最终结论）。

视频帧率为 5 fps；报告中的代表静帧位于 `report/images/`（开发帧目录未提交）。
