# 鳴ったコマに「鳴ったタイル」の枠を描いて並べる（数字だけでなく位置を見るため）。
# usage: python alarm_sheet.py result.json framesroot video threshold out.jpg [missed]
import sys, json, os, re
from PIL import Image, ImageDraw
d = json.load(open(sys.argv[1])); root, vid, th, out = sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
missed = len(sys.argv) > 6
B = d["classes"].index("bear")
its = [it for it in d["items"] if vid in it["file"]]
sel = []
for it in its:
    best = max(it["views"], key=lambda v: v["p"][B])
    if (best["p"][B] >= th) != missed: sel.append((it, best))
step = max(1, len(sel) // 24); sel = sel[::step][:24]
W, tw = 6, 300
s = None
for k, (it, best) in enumerate(sel):
    f = os.path.join(root, "frames_" + vid, os.path.basename(it["file"].replace("\\", "/")))
    im = Image.open(f).convert("RGB"); dr = ImageDraw.Draw(im)
    dr.rectangle(best["box"], outline=(255, 0, 0) if not missed else (255, 200, 0), width=6)
    n = re.search(r"(\d+)\.jpg", f).group(1)
    th_ = int(tw * im.height / im.width); im = im.resize((tw, th_))
    if s is None:
        rows = (len(sel) + W - 1) // W; s = Image.new("RGB", (W * tw, rows * (th_ + 16)), "white"); D = ImageDraw.Draw(s)
    x = (k % W) * tw; y = (k // W) * (th_ + 16)
    s.paste(im, (x, y + 16)); D.text((x + 2, y + 2), "%s  p(bear)=%.2f  %s" % (n, best["p"][B], d["classes"][max(range(len(best["p"])), key=lambda i: best["p"][i])]), fill="black")
if s: s.save(out, quality=85)
print(len(sel), "frames")
