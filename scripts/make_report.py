#!/usr/bin/env python3
"""生成课程报告 DOCX（随后用 LibreOffice 转 PDF）。"""
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "REPORT.docx"
FONT = "Noto Serif CJK SC"
TEAL = RGBColor(0x1F, 0x6F, 0x5C)
GRAY = RGBColor(0x55, 0x55, 0x55)


def set_run_font(run, size=None, bold=None, color=None):
    run.font.name = FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def para(doc, text, size=10.5, bold=False, color=None, align=None, space_after=3):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.25
    r = p.add_run(text)
    set_run_font(r, size=size, bold=bold, color=color)
    return p


def heading(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    set_run_font(r, size=14 if level == 1 else 11.5, bold=True, color=TEAL)
    # 下划线（一级标题）
    if level == 1:
        pPr = p._p.get_or_add_pPr()
        from docx.oxml import OxmlElement
        pbdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "6")
        bottom.set(qn("w:space"), "1")
        bottom.set(qn("w:color"), "1F6F5C")
        pbdr.append(bottom)
        pPr.append(pbdr)
    return p


def table(doc, headers, rows, widths=None, font_size=8.8):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        r = p.add_run(h)
        set_run_font(r, size=font_size, bold=True)
        p.paragraph_format.space_after = Pt(0)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            r = p.add_run(str(val))
            set_run_font(r, size=font_size)
            p.paragraph_format.space_after = Pt(0)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    # 防止表格行跨页被切断
    from docx.oxml import OxmlElement

    for row in t.rows:
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
    # 跨页时重复表头
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    return t


def image_row(doc, left, left_cap, right, right_cap, width_cm=7.6):
    t = doc.add_table(rows=1, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, path, cap in [
        (t.rows[0].cells[0], left, left_cap),
        (t.rows[0].cells[1], right, right_cap),
    ]:
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.paragraphs[0].add_run().add_picture(str(path), width=Cm(width_cm))
        cp = cell.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cp.add_run(cap)
        set_run_font(r, size=8.5, color=GRAY)


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(1.5)
    sec.bottom_margin = Cm(1.5)
    sec.left_margin = Cm(1.6)
    sec.right_margin = Cm(1.6)

    # 默认样式
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

    # 标题
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("OpenVLA-7B 轻量微调与 SimplerEnv WidowX 闭环评测")
    set_run_font(r, size=19, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(
        "机器学习课程大作业 · B 类（工程实践型）\n"
        "姓名/学号：胥昊天（26213050409）  |  2026-10-19\n"
        "组员：曹奕伦（26113050003）、徐厚泽（26113050327）、"
        "彭泽宇（26213050317）、刘翔（26113050287）"
    )
    set_run_font(r, size=9.5, color=GRAY)

    # 1
    heading(doc, "1. 项目概述")
    para(doc,
         "本项目完成一次完整的 VLA 工程链路：使用公开的 OpenVLA-7B 作为基础模型，在 BridgeData V2 "
         "的 WidowX 子集上做 LoRA 轻量微调，并在 SimplerEnv 的 WidowX Visual Matching 环境中对 "
         "Base 与 Final 进行闭环评测。必做评测为 T1（勺子放到毛巾上）与 T2（茄子放入篮子），"
         "每个模型每个任务 10 个预先固定的初始状态，共 40 个 episode。")
    para(doc,
         "课程不要求 Final 一定优于 Base。本项目最终结果：Base B-SimplerScore 20.0，Final 30.0，"
         "Final − Base = +10.0 个百分点（T1 −10，T2 +30）。训练 loss 从约 11.9 降至 3.0–3.3。"
         "我们在排查中发现并修正了 prompt 模板缺失，并定位到仿真渲染锐度导致的视觉域差异；"
         "加入统一高斯模糊后，Base/Final 从 sharp 基线的 0/10 提升到 20.0/30.0。"
         "报告将区分“有证据的观察”和“尚未验证的解释”。")

    # 2
    heading(doc, "2. 模型选择与依据")
    table(doc, ["项目", "内容"], [
        ["基础模型", "openvla/openvla-7b（OpenVLA，7B 参数）"],
        ["WidowX 数据依据", "OpenVLA 在 Open X-Embodiment 上预训练；BridgeData V2 是 OXE 的组成数据集之一，"
                            "由 WidowX 真机采集。模型卡：https://huggingface.co/openvla/openvla-7b"],
        ["动作接口", "主视角 RGB（预处理为 224×224）+ 语言指令；输出 7 维末端增量 "
                     "[x, y, z, roll, pitch, yaw, gripper]；夹爪 1=张开、0=闭合"],
        ["微调模块", "LoRA（rank 32，target_modules=all-linear），可训练参数 110,828,288 / "
                     "7,652,065,472（1.45%）"],
        ["Final 形式", "LoRA adapter（约 484 MB），依赖上述 Base，不保存合并后的 15 GB 全模型"],
    ], widths=[3.4, 14.0])
    para(doc,
         "选择 OpenVLA 的原因：它是 PPT 列出的公开 VLA 之一，具备 WidowX/BridgeData V2 预训练依据，"
         "有社区维护的 SimplerEnv WidowX 适配器，且 7B 规模的 LoRA 微调能在单张 32 GB RTX 5090 上完成。"
         "相比 π0/π0.5/GR00T，OpenVLA 的 SimplerEnv WidowX 适配代码更成熟；相比 Octo，它更符合课程对"
         "“PPT 中 VLA”的要求。我们使用社区适配器 simpler_env/policies/openvla，并把其 "
         "flash_attention_2 改为 sdpa（RTX 5090 上旧版 flash-attn 不可用）。")

    # 3
    heading(doc, "3. 数据与处理")
    para(doc,
         "数据来自 BridgeData V2（WidowX），使用 Hugging Face 镜像 IPEC-COMMUNITY/bridge_orig_lerobot"
         "（LeRobot v2 格式；原始项目：https://rail-berkeley.github.io/bridgedata/）。按语言指令筛选"
         "勺子/毛巾、茄子、胡萝卜/盘子、绿黄方块叠放相关轨迹，每类最多 400 条，共 1584 条轨迹"
         "（3168 个文件，约 362 MB）。")
    table(doc, ["字段", "处理"], [
        ["图像", "主视角 image_0，256×256，5 fps；训练时缩放到 224×224"],
        ["语言", "language_instruction，来自 BridgeData V2 的原始指令"],
        ["动作", "7 维 [x,y,z,roll,pitch,yaw,gripper]；gripper 1=张开、0=闭合"],
        ["状态", "8 维 [x,y,z,roll,pitch,yaw,pad,gripper]；仅用于动作重标定/统计，不输入模型"],
        ["归一化", "Base 使用预训练 bridge_orig 统计量；Final 使用训练子集统计量"
                   "（保存在 run 目录 dataset_statistics.json，推理时显式覆盖）"],
        ["划分", "按完整轨迹划分：train 1506 条、val 78 条；不存在同一轨迹跨 split 的帧泄漏"],
    ], widths=[3.4, 14.0])
    para(doc,
         "说明：镜像的 LeRobot 版本已经去掉了官方 Bridge TFDS 的首帧零动作；因此我们注册了 "
         "bridge_subset transform，只做夹爪二值化和基于状态的 reached-action 重标定，不删除第一帧。"
         "数据检查确认视频帧数与 parquet 行数一致、图像为 256×256 uint8、动作范围为小幅末端增量、"
         "夹爪为 0/1，且 RLDS 可被 OpenVLA 的 RLDSDataset 正常读取。", size=9.2, color=GRAY)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(ROOT / "report/images/data_sample.png"), width=Cm(6.5))
    para(doc,
         "图：BridgeData V2 样本（episode 150，第 0 帧）。instruction = \"put eggplant on plate\"；"
         "action = [-0.0001, -0.0030, -0.0063, -0.0045, 0.0193, -0.0037, 1.0]"
         "（x, y, z, roll, pitch, yaw, gripper；gripper=1 表示张开）。",
         size=8.5, color=GRAY, align=WD_ALIGN_PARAGRAPH.CENTER)

    # 4
    heading(doc, "4. 训练方法")
    table(doc, ["配置", "值"], [
        ["微调方式", "LoRA，rank 32，all-linear，dropout 0"],
        ["优化器 / 学习率", "AdamW，lr 5e-4，weight decay 0.01，grad clip 1.0"],
        ["batch", "batch 2 × 梯度累积 4 = 等效 batch 8"],
        ["训练步数", "固定 3000 个优化步（不根据正式评测结果选 checkpoint）"],
        ["数据增强", "关闭（image_aug=False），减少 CPU 数据管线开销并便于复现"],
        ["计算资源", "单张 RTX 5090 32 GB；实际耗时约 48 分钟"],
        ["训练日志", "results/train/train_log.txt"],
        ["Final 选择", "预先固定 3000 步的最终 adapter；未使用正式评测结果挑权重"],
    ], widths=[3.6, 13.8])
    para(doc,
         "训练 loss 从约 11.9 降到 3.0–3.3，action accuracy 从约 0.14 升到 0.20–0.25。"
         "loss 下降说明模型在训练子集上确实学到了东西；但训练目标与 SimplerEnv 的闭环成功率并不等价。")
    para(doc,
         "磁盘约束说明：合并后的 15 GB 全模型会与 15 GB Base 同时占用磁盘，当前共享盘只剩约 16 GB，"
         "因此本项目只保存 LoRA adapter（约 484 MB）。课程提交规范允许轻量权重，只需注明依赖的 Base "
         "版本和加载方式；加载示例见第 8 节与 checkpoint_link.txt。", size=9.2, color=GRAY)

    # 5
    heading(doc, "5. 评测设置")
    table(doc, ["项目", "设置"], [
        ["环境版本", "SimplerEnv commit eddb5690749a1e34267807c0f7b5eede323918d7；"
                     "ManiSkill2_real2sim commit ef7a4d4fdf4b69f2c2154db5b15b9ac8dfe10682"],
        ["T1", "PutSpoonOnTableClothInScene-v0，scene bridge_table_1_v1，max 60 steps，"
               "指令 put the spoon on the towel"],
        ["T2", "PutEggplantInBasketScene-v0，scene bridge_table_1_v2，max 120 steps，"
               "指令 put eggplant into yellow basket"],
        ["控制", "WidowX Visual Matching；控制频率 5 Hz，仿真频率 500 Hz"],
        ["图像预处理", "Base/Final 统一高斯模糊 sigma=1.5（configs/policy_preprocess.json）；"
                       "sigma 仅在开发状态 id 10–12 上选择，正式状态 0–9 未参与"],
        ["初始状态", "object episode id 0–9，Robot init 固定；清单见 configs/eval_states.json。"
                     "开发/冒烟只用 id 10–23，正式评测未使用"],
        ["随机性", "环境 seed 与模型 seed 分开记录在逐 episode CSV；两个模型使用同一清单"],
        ["成功判定", "严格读取环境原生 info[\"success\"]，不用 done/terminated 代替"],
        ["进程隔离", "Base 与 Final 在独立进程中加载和评测；无人工干预、不筛除失败 episode"],
    ], widths=[3.4, 14.0])
    para(doc,
         "评测在 GPU 0 上串行完成。共享服务器上 GPU 1/2 在管理员清理进程后出现 CUDA context "
         "初始化异常（nvidia-smi 显示空闲但 CUDA 分配失败），因此没有使用并行评测；这不改变任务、"
         "初始状态或协议。", size=9.2, color=GRAY)
    para(doc,
         "重要修正：社区适配器最初直接把原始指令传给 processor，漏掉了 OpenVLA 训练时的 prompt 模板"
         "（In: What action should the robot take to {instruction}?\\nOut:）。修正后我们重跑了 "
         "Base/Final 的 40 个正式 episode；本报告结果均为修正 prompt 后的结果。修正前的裸 prompt "
         "结果归档在 results/archive/bare_prompt/，不用于最终结论。", size=9.2, color=GRAY)
    para(doc,
         "域适应：为缩小真实相机图像与 SimplerEnv 渲染图像的锐度差异，Base/Final 统一在送入模型前"
         "做 sigma=1.5 的高斯模糊；sigma 只在开发状态 object id 10–12 上选择，正式状态 0–9 未参与。"
         "模糊是 policy 的输入预处理，不改变环境、任务或初始状态；配置见 "
         "configs/policy_preprocess.json。", size=9.2, color=GRAY)

    # 6
    heading(doc, "6. 实验结果")
    table(doc, ["模型", "训练模块", "T1 成功数", "T2 成功数", "B-SimplerScore"], [
        ["Base", "未进行本次微调", "2/10（20%）", "2/10（20%）", "20.0"],
        ["Final", "LoRA (r=32)", "1/10（10%）", "5/10（50%）", "30.0"],
    ], widths=[2.6, 4.2, 3.2, 3.2, 3.2])
    para(doc,
         "Final − Base = +10.0 个百分点（T1 −10.0，T2 +30.0）。40 个 episode 全部按预注册清单"
         "完成，逐 episode 记录在 results/episodes.csv，汇总在 results/results.csv，"
         "视频在 results/videos/。")
    para(doc,
         "作为消融：同样修正 prompt、但不加模糊时，Base/Final 在正式 40 个 episode 上均为 0/10"
         "（归档在 results/archive/sharp_baseline/）。因此本次提升主要来自图像域适应；Final 相对 Base 的"
         " +10.0 pp 则来自 LoRA 微调（主要集中在 T2）。10 个 episode 的方差较大，"
         "+10.0 pp 和 T2 +30.0 pp 不能视为稳定优势。")

    # 7
    heading(doc, "7. 案例分析与局限")
    heading(doc, "7.1 接口验证（先排除接口错误）", level=2)
    table(doc, ["帧", "真实动作（前 6 维 + 夹爪）", "OpenVLA 预测"], [
        ["0", "[-0.0028,-0.0025,-0.0141,-0.0249,0.0361,0.0022, 1.0]",
         "[-0.0029,-0.0024,-0.0140,-0.0250,0.0361,-0.0001, 0.9961]"],
        ["10", "[0.0040,0.0069,-0.0013,-0.0160,-0.0122,0.0048, 0.0]",
         "[0.0040,-0.0024,-0.0005,0.0037,0.0006,-0.0097, 0.0]"],
        ["19", "[0.0004,-0.0218,-0.0218,0.0583,0.0519,0.1118, 0.0]",
         "[0.0005,-0.0218,-0.0174,0.0043,0.0521,-0.0275, 0.0]"],
    ], widths=[1.2, 7.6, 8.2], font_size=7.8)
    para(doc,
         "在真实 BridgeData 帧上，OpenVLA 的预测与数据集真实动作非常接近，夹爪开/闭也一致。"
         "这说明 prompt 模板、动作 token 解码、反归一化和图像预处理都是正确的；"
         "SimplerEnv 中的 0 成功不能解释为动作接口接错，主因是真实 Bridge 图像与仿真渲染图像"
         "之间的域差异，以及当前 LoRA 微调规模有限。脚本与输出：scripts/check_openvla_real.py、"
         "results/analysis/diagnostics_real_bridge.txt。进一步定位到：仿真渲染图像比真实相机图像更锐利；"
         "sharp 模式下 debug 夹爪长期为 0.9961（张开），加入 sigma=1.5 模糊后夹爪开始闭合。"
         "sharp 正式评测 Base/Final 均为 0/10；blur 正式评测 Base 20.0、Final 30.0。", size=9.5)

    heading(doc, "7.2 T1：勺子放到毛巾上（Base vs Final）", level=2)
    image_row(doc, ROOT / "report/images/t1_base_success.png", "Base 成功，obj id 3，第 25 帧",
              ROOT / "report/images/t1_final_failure.png", "Final 失败，obj id 0，最后帧")
    para(doc,
         "Base 成功 episode：机械臂接近勺子、闭合夹爪、抬起并移动到毛巾上方，最终满足“勺子在毛巾上”"
         "的判定（31 步）。Final 在 T1 的成功率略低（1/10 vs 2/10）；该失败样例中机械臂接近了勺子，"
         "但没有稳定抓取/放置。单看一个 episode 不能说明总体差异。")

    heading(doc, "7.3 T2：茄子放入篮子（Base vs Final）", level=2)
    image_row(doc, ROOT / "report/images/t2_base_success.png", "Base 成功，obj id 2，第 20 帧",
              ROOT / "report/images/t2_final_success.png", "Final 成功，obj id 0，第 14 帧")
    para(doc,
         "Base 与 Final 都有把茄子放入篮子的成功样例：Base obj id 2 用 23 步，Final obj id 0 用 16 步。"
         "Final 在 T2 的成功率为 5/10，高于 Base 的 2/10；这是本次对比中最明显的变化，"
         "但 10 个 episode 的样本量仍不足以证明稳定优势。")

    heading(doc, "7.4 有证据的观察与可能解释", level=2)
    para(doc, "有证据的观察：", bold=True)
    for s in [
        "sharp（无模糊）正式评测 Base/Final 均 0/10；blur sigma=1.5 后 Base 20.0、Final 30.0；",
        "debug 显示 sharp 下夹爪长期为 0.9961（张开），blur 后夹爪开始闭合并出现成功 episode；",
        "Base T1 2/10、T2 2/10；Final T1 1/10、T2 5/10；Final 的变化集中在 T2；",
        "训练 loss 从约 11.9 降至 3.0–3.3，action accuracy 从约 0.14 升至 0.20–0.25；",
        "image_aug=True 的第二次训练在 dev（加 blur）为 1/6，与原始 Final 持平，未采用。",
        "T1 数据质量消融：只保留 188 条严格目标轨迹重训，dev T1 1/3（与混合 Final 持平）、"
        "T2 2/3（混合 Final 0/3）；但正式 T1 为 0/10（混合 Final 1/10），dev 的 1 次成功未复现。"
        "T1 未改善，严格 adapter 未替换 Final。",
        "训练步数诊断：val loss Base 11.16、3000 步 3.17、8000 步 3.67；8000 步训练 loss 更低"
        "（约 2.6）但验证 loss 升高，开发集 T1/T2 与 3000 步相同，属于过拟合。",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(1)
        set_run_font(p.add_run(s), size=10.2)

    para(doc, "可能但尚未验证的解释：", bold=True)
    for i, s in enumerate([
        "模糊降低了仿真渲染与真实相机的锐度差异，使模型恢复“该闭合夹爪”的行为；"
        "sigma 仅在 3 个开发状态上选择，存在过拟合风险；",
        "Final 在 T2 提升更明显，可能与训练子集中茄子/篮子相关轨迹有关；T1 略低可能来自"
        "场景多样性、LoRA 容量或评测方差；",
        "LoRA 只更新 1.45% 参数、只训练 3000 步，长程抓取-放置策略的改变有限；",
        "每任务只有 10 个 episode，+10.0 pp 与 T2 +30.0 pp 不能视为稳定优势；",
        "Base 与 Final 的反归一化统计量不同（预训练 vs 子集），可能带来轻微动作尺度差异。",
    ], 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.6)
        p.paragraph_format.space_after = Pt(1)
        set_run_font(p.add_run(f"{i}. {s}"), size=10.2)

    heading(doc, "7.5 局限", level=2)
    for s in [
        "单次训练、单一种子；没有多随机种子或多数据规模消融（B 类不要求）；",
        "正式评测每任务 10 次，+10.0 pp 与 T2 +30.0 pp 不能视为稳定优势；",
        "模糊 sigma=1.5 只在 3 个开发状态上选择，存在过拟合开发状态的风险；"
        "更严格的做法是用更多开发状态或独立验证集选择；",
        "sharp（无模糊）基线为 0/10；模糊后的提升有视频和动作 debug 证据，但仍需更大样本验证；",
        "T1 数据质量消融：dev 1/3、正式 T1 0/10；数据清洗没有改善 T1，"
        "不能据此断言数据清洗无效，但也不支持“T1 差只是因为数据混杂”的解释；",
        "8000 步 val loss（3.67）高于 3000 步（3.17），继续加步数会过拟合；"
        "增加训练量不是提升 T1 的有效方向；",
        "训练数据来自公开真机数据，与仿真评测的初始布局/物体外观仍有差异；",
        "Final 为 LoRA adapter，未做全参数微调或更大规模训练；",
        "未对失败视频做物体位姿级别的定量标注，案例结论以观察为主。",
    ]:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(1)
        set_run_font(p.add_run(s), size=10.2)

    # 8
    heading(doc, "8. 复现信息与分工")
    table(doc, ["项目", "内容"], [
        ["公开代码仓库", "https://github.com/xht03/2026ML"],
        ["Final checkpoint", "LoRA adapter，约 484 MB；GitHub Release v1.0-lora："
                             "https://github.com/xht03/2026ML/releases/tag/v1.0-lora；"
                             "zip：https://github.com/xht03/2026ML/releases/download/"
                             "v1.0-lora/2026ML-openvla-lora.zip"],
        ["Base 模型", "openvla/openvla-7b（Hugging Face）"],
        ["关键依赖", "PyTorch 2.7.1+cu128、transformers 4.40.1、tokenizers 0.19.1、timm 0.9.10、"
                     "peft 0.11.1、TensorFlow 2.15 / TFDS 4.9.4、SAPIEN 2.2.2"],
        ["训练命令", "见 README 第 5 节；等效 batch 8，固定 3000 步"],
        ["评测命令", "见 README 第 6 节；Base/Final 独立进程，object id 0–9"],
        ["图像预处理", "configs/policy_preprocess.json：gaussian_blur_sigma=1.5（Base/Final 相同）"],
        ["逐 episode 结果", "results/episodes.csv；汇总 results/results.csv"],
        ["视频", "results/videos/（全部 40 个；代表视频见 README 7.2）"],
        ["组员分工", "胥昊天：项目统筹、环境/数据管线、LoRA 训练与最终评测、报告整合；"
                     "曹奕伦：BridgeData 筛选与 RLDS 构建、训练日志与验证 loss 分析；"
                     "徐厚泽：SimplerEnv 评测脚本、初始状态清单与逐 episode 结果汇总；"
                     "彭泽宇：视频采集与代表性 episode 分析、失败原因整理；"
                     "刘翔：环境部署、模型/文献调研、复现文档与报告校对"],
    ], widths=[3.4, 14.0])
    para(doc,
         "Final adapter 加载方式：PeftModel.from_pretrained(base, adapter)，其中 base 为 "
         "openvla/openvla-7b；训练子集的统计量在 run 目录 dataset_statistics.json，推理时覆盖 "
         "norm_stats[\"bridge_orig\"]。", size=9.2, color=GRAY)

    # 9
    heading(doc, "参考文献")
    refs = [
        "Li et al. Evaluating Real-World Robot Manipulation Policies in Simulation (SimplerEnv). arXiv:2405.05941, 2024.",
        "Walke et al. BridgeData V2: A Dataset for Robot Learning at Scale. arXiv:2308.12952, 2023.",
        "Kim et al. OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246, 2024.",
        "Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685, 2021.",
        "Open X-Embodiment Collaboration. Open X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv:2310.08864, 2023.",
        "OpenVLA model card: https://huggingface.co/openvla/openvla-7b",
        "SimplerEnv: https://github.com/simpler-env/SimplerEnv",
        "BridgeData V2: https://rail-berkeley.github.io/bridgedata/",
    ]
    for i, s in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.5)
        p.paragraph_format.space_after = Pt(1)
        set_run_font(p.add_run(f"[{i}] {s}"), size=8.8, color=GRAY)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
