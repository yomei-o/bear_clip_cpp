# 動画・検証用画像（出どころ別）・ニュース画像を、同じしきい値でまとめて数える。
# usage: python summary.py video.json val.json news.json [threshold...]
import sys, json, os, re, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
V, VA, N = (json.load(open(p)) for p in sys.argv[1:4])
ths = [float(t) for t in sys.argv[4:]] or [0.5, 0.7, 0.9]
B = V["classes"].index("bear")
score = lambda it: max(v["p"][B] for v in it["views"])

def vtruth(f):
    n = int(re.search(r"(\d+)\.jpg$", f).group(1))
    if "surveillance" in f:
        if 160 <= n <= 360 or 660 <= n <= 860: return 1
        if n <= 140 or 390 <= n <= 640 or n >= 890: return 0
        return None
    return 1 if n >= 21 else (0 if n <= 12 else None)

rows = []
def rate(xs, th): xs = np.array(xs); return 100 * (xs >= th).mean() if len(xs) else float("nan")
groups = {}
for it in V["items"]:
    t = vtruth(it["file"]); vid = "玄関カメラ" if "surveillance" in it["file"] else "図書館"
    if t is None: continue
    groups.setdefault((vid + (" 熊あり" if t else " 熊なし"), t), []).append(score(it))
for it in VA["items"]:
    n = os.path.basename(it["file"].replace("\\", "/")); src = n.split("_")[0]
    name = {"lila": "森のカメラ", "coco": "COCO 熊", "coconeg": "町の画像(COCO)", "xneg": "首都高の夜", "inat": "iNaturalist 熊"}.get(src, src)
    if src == "lila": name += " 熊" if it["label"] else " 熊なし"
    groups.setdefault((name, it["label"]), []).append(score(it))
for it in N["items"]:
    groups.setdefault(("日本のニュース 熊", 1), []).append(score(it))
print("%-22s %6s " % ("", "枚数") + " ".join("  th=%.1f" % t for t in ths) + "   （熊ありは見つけた割合、熊なしは誤報の割合）")
for (name, lab), xs in sorted(groups.items(), key=lambda kv: (-kv[0][1], kv[0][0])):
    print("%-22s %6d " % (name, len(xs)) + " ".join("%7.1f%%" % rate(xs, t) for t in ths))
