# 検証用画像で「熊のいない画像なのに鳴った」ものを、鳴ったタイルの枠つきで並べる。
# usage: python fp_sheet.py result.json threshold out.jpg [full]  （画像はビルド機から別途コピーしてある前提: build/val/<名前>）
import sys, json, os, random
from PIL import Image, ImageDraw
d = json.load(open(sys.argv[1])); th = float(sys.argv[2]); out = sys.argv[3]; only_full = len(sys.argv) > 4
B = d["classes"].index("bear")
fps = []
for it in d["items"]:
    if it["label"] != 0: continue
    vs = it["views"][:1] if only_full else it["views"]
    best = max(vs, key=lambda v: v["p"][B])
    if best["p"][B] >= th: fps.append((it, best))
random.Random(0).shuffle(fps); fps = fps[:30]
W, tw, thh = 6, 260, 200
s = Image.new("RGB", (W * tw, ((len(fps) + W - 1) // W) * (thh + 16)), "white"); D = ImageDraw.Draw(s)
names = []
for k, (it, best) in enumerate(fps):
    name = os.path.basename(it["file"].replace("\\", "/")); names.append(name)
    p = os.path.join("build", "val", name)
    if not os.path.exists(p): continue
    im = Image.open(p).convert("RGB"); ImageDraw.Draw(im).rectangle(best["box"], outline=(255, 0, 0), width=max(3, im.width // 150))
    im.thumbnail((tw - 4, thh)); x = (k % W) * tw; y = (k // W) * (thh + 16)
    s.paste(im, (x, y + 16)); top = max(range(len(best["p"])), key=lambda i: best["p"][i])
    D.text((x + 2, y + 2), "%s %.2f" % (name[:22], best["p"][B]), fill="black")
s.save(out, quality=85)
open(out + ".txt", "w").write("\n".join(names))
print(len(fps))
