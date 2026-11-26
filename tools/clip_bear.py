# CLIP のゼロショットで「熊がいるか」を出す実験。画面全体＋タイルで見て、熊の確率を view ごとに保存する。
# usage: python clip_bear.py out.json model pretrained imgdir_or_list...
#   入力はフォルダ（*.jpg / *.png を全部）か、YOLO 形式の images フォルダ（labels が隣にあれば正解も入れる）
import sys, os, json, glob, numpy as np, torch, open_clip
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CLASSES = [
    ("bear", ["a photo of a bear", "a photo of a black bear walking", "a security camera image of a bear", "a night vision image of a bear"]),
    ("car", ["a photo of a car", "a photo of a parked car at night", "a security camera image of a car"]),
    ("person", ["a photo of a person", "a security camera image of a person walking"]),
    ("dog", ["a photo of a dog", "a security camera image of a dog"]),
    ("cat", ["a photo of a cat"]),
    ("deer", ["a photo of a deer"]),
    ("boar", ["a photo of a wild boar"]),
    ("raccoon", ["a photo of a raccoon"]),
    ("bird", ["a photo of a bird"]),
    ("empty", ["a security camera image of an empty driveway at night", "a photo of an empty room",
               "a photo of an empty street", "a photo of a forest with no animals", "a photo of grass and trees"]),
]

SCALES = [float(v) for v in os.environ.get("BEAR_SCALES", "0.667").split(",")]

def views(im):
    """画面全体（正方形に余白）と、短辺の SCALES 倍の正方形タイル（縦横に 1/4 ずつ重ねる）"""
    w, h = im.size; out = [("full", (0, 0, w, h))]
    for sc in SCALES:
        s = int(min(w, h) * sc)
        nx = max(2, int(np.ceil((w - s) / (s * 0.75))) + 1); ny = max(2, int(np.ceil((h - s) / (s * 0.75))) + 1)
        for j in range(ny):
            for i in range(nx):
                x = round(i * (w - s) / (nx - 1)); y = round(j * (h - s) / (ny - 1))
                out.append(("s%.2f_%d_%d" % (sc, j, i), (x, y, x + s, y + s)))
    return out

def pad(im):
    w, h = im.size; n = max(w, h); o = Image.new("RGB", (n, n), (123, 117, 104)); o.paste(im, ((n - w) // 2, (n - h) // 2)); return o

out, mn, pre = sys.argv[1], sys.argv[2], sys.argv[3]
items = []
for src in sys.argv[4:]:
    files = sorted(glob.glob(os.path.join(src, "*.jpg")) + glob.glob(os.path.join(src, "*.png")))
    lab_dir = os.path.join(os.path.dirname(src.rstrip("/\\")), "labels")
    for f in files:
        lab = None
        lf = os.path.join(lab_dir, os.path.splitext(os.path.basename(f))[0] + ".txt")
        if os.path.exists(lf): lab = int(os.path.getsize(lf) > 0)
        items.append((f, lab))
m, _, prep = open_clip.create_model_and_transforms(mn, pretrained=pre); m = m.cuda().eval(); tok = open_clip.get_tokenizer(mn)
with torch.no_grad():
    T = []
    for _, ps in CLASSES:
        t = m.encode_text(tok(ps).cuda()); t = t / t.norm(dim=-1, keepdim=True); t = t.mean(0); T.append(t / t.norm())
    T = torch.stack(T)
    res = []; feats = []
    for k, (f, lab) in enumerate(items):
        im = Image.open(f).convert("RGB")
        vs = views(im)
        x = torch.stack([prep(pad(im.crop(b))) for _, b in vs]).cuda()
        F = m.encode_image(x); F = F / F.norm(dim=-1, keepdim=True)
        P = (100 * F @ T.T).softmax(-1).cpu().numpy()
        res.append({"file": f, "label": lab, "views": [{"name": n, "box": b, "p": [round(float(v), 5) for v in p]} for (n, b), p in zip(vs, P)]})
        feats.append(F.half().cpu().numpy())   # 言葉を変えて採点し直せるように、view ごとの画像特徴も残す
        if k % 200 == 0: print(k, len(items), flush=True)
json.dump({"model": mn + "/" + pre, "classes": [c for c, _ in CLASSES], "items": res}, open(out, "w"))
np.savez_compressed(os.path.splitext(out)[0] + "_feats.npz", *feats)
print("done", len(res))
