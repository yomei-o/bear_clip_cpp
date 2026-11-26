# classes_src.json の言葉を CLIP のテキスト側でベクトルにして classes.json に書く（torch が要るのはここだけ）。
# 候補ごとに言葉の平均をとって正規化する。
# usage: python make_text.py classes_src.json classes.json
import sys, json, torch, open_clip
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
src = json.load(open(sys.argv[1], encoding="utf-8"))
m, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai"); m.eval()
tok = open_clip.get_tokenizer("ViT-B-32-quickgelu")
out = {"model": "openai CLIP ViT-B/32 (quickgelu)", "alarm": src["alarm"], "logit_scale": 100.0, "classes": []}
with torch.no_grad():
    for name, prompts in src["classes"]:
        t = m.encode_text(tok(prompts)); t = t / t.norm(dim=-1, keepdim=True); t = t.mean(0); t = t / t.norm()
        out["classes"].append({"name": name, "prompts": prompts, "emb": [round(float(v), 7) for v in t]})
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
print(len(out["classes"]), "classes ->", sys.argv[2])
