#!/usr/bin/env python3
"""在 held-out val split 上比较 Base / 不同 adapter 的 action-token loss。

用法：
  python scripts/eval_val_loss.py \
    --checkpoint downloads/openvla-7b \
    --adapter base= \
    --adapter final3000=checkpoints/openvla_lora/adapters/<...3000> \
    --adapter final8000=checkpoints/openvla_lora/adapters/<...8000> \
    --batches 20 --batch-size 4
"""
from __future__ import annotations

import argparse
import itertools
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf  # noqa: E402

tf.config.set_visible_devices([], "GPU")

import tensorflow_datasets as tfds  # noqa: E402

_orig_builder = tfds.builder


def _local_builder(name, data_dir=None, **kwargs):
    if name == "bridge_subset":
        return tfds.builder_from_directory(
            os.path.join(data_dir, "bridge_subset", "0.1.0")
        )
    return _orig_builder(name, data_dir=data_dir, **kwargs)


tfds.builder = _local_builder

from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: E402

from prismatic.models.backbones.llm.prompting import PurePromptBuilder  # noqa: E402
from prismatic.util.data_utils import PaddedCollatorForActionPrediction  # noqa: E402
from prismatic.vla.action_tokenizer import ActionTokenizer  # noqa: E402
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset  # noqa: E402


def make_loader(args, processor):
    batch_transform = RLDSBatchTransform(
        ActionTokenizer(processor.tokenizer),
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    dataset = RLDSDataset(
        Path(args.data_root_dir),
        "bridge_subset",
        batch_transform,
        resize_resolution=(224, 224),
        shuffle_buffer_size=100,
        train=False,  # val split
        image_aug=False,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, collate_fn=collator, num_workers=0
    )
    batches = list(itertools.islice(loader, args.batches))
    print(f"val batches: {len(batches)} x {args.batch_size}, "
          f"dataset len={len(dataset)}")
    return batches


def evaluate(model, batches):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in batches:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    input_ids=batch["input_ids"].to("cuda"),
                    attention_mask=batch["attention_mask"].to("cuda"),
                    pixel_values=batch["pixel_values"].to("cuda", dtype=torch.bfloat16),
                    labels=batch["labels"].to("cuda"),
                )
            losses.append(float(out.loss))
    return sum(losses) / len(losses), losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="downloads/openvla-7b")
    parser.add_argument("--data-root-dir", default="data/bridge_subset/tfds")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--adapter", action="append", default=[],
        help="name=path；path 为空表示 Base",
    )
    args = parser.parse_args()
    assert args.adapter, "至少提供一个 --adapter name=path"

    processor = AutoProcessor.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    batches = make_loader(args, processor)

    for spec in args.adapter:
        name, _, adapter_path = spec.partition("=")
        base = AutoModelForVision2Seq.from_pretrained(
            args.checkpoint,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).cuda()
        if adapter_path:
            model = PeftModel.from_pretrained(base, adapter_path)
        else:
            model = base
        mean_loss, losses = evaluate(model, batches)
        print(
            f"VAL_LOSS {name}: mean={mean_loss:.5f} "
            f"min={min(losses):.5f} max={max(losses):.5f}"
        )
        del model, base
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
