import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import av, numpy as np, pandas as pd, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

root = "/home/xht/MLwork"
ep = 171
raw = f"{root}/data/bridge_subset/raw"
df = pd.read_parquet(f"{raw}/data/chunk-000/episode_{ep:06d}.parquet")
actions = np.asarray(df["action"].tolist(), dtype=np.float32)
with av.open(f"{raw}/videos/chunk-000/observation.images.image_0/episode_{ep:06d}.mp4") as c:
    frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
sel = {e["episode_index"]: e for e in json.load(open(f"{raw}/selected_episodes.json"))}
instr = sel[ep]["tasks"][0]
print("instruction:", instr, "frames", len(frames), "actions", actions.shape)
close_idx = np.where(actions[:, -1] < 0.05)[0]
print("close frame indices:", close_idx[:10])
pick = [0, len(frames)//2, int(close_idx[0]) if len(close_idx) else len(frames)//2]
print("picked frames:", pick)

proc = AutoProcessor.from_pretrained(f"{root}/downloads/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    f"{root}/downloads/openvla-7b", attn_implementation="sdpa",
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
).cuda()
prompt = f"In: What action should the robot take to {instr}?\nOut:"
for idx in pick:
    image = Image.fromarray(frames[idx])
    inputs = proc(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    pred = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    print(f"frame {idx}: GT={np.round(actions[idx],4).tolist()}  PRED={np.round(pred,4).tolist()}")
