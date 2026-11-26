# 動画のコマごとの熊の確率から、前回（YOLO）と同じ数え方で警報の割合を出す。
# usage: python video_metrics.py result.json [threshold...]
import sys, json, re, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = json.load(open(sys.argv[1])); B = d["classes"].index("bear")
ths = [float(t) for t in sys.argv[2:]] or [0.3, 0.5, 0.7]

def truth(f):
    n = int(re.search(r"(\d+)\.(jpg|png)$", f).group(1))
    if "surveillance" in f:
        if 160 <= n <= 360 or 660 <= n <= 860: return 1
        if n <= 140 or 390 <= n <= 640 or n >= 890: return 0
        return None          # 出入りの境目は数えない
    if "library" in f:
        return 1 if n >= 21 else (0 if n <= 12 else None)

def tile_center(v):
    x0, y0, x1, y1 = v["box"]; return (x0 + x1) / 2, (y0 + y1) / 2

for vid in ("surveillance", "library"):
    its = [it for it in d["items"] if vid in it["file"]]
    for mode in ("full", "tiles"):
        for th in ths:
            hit = {0: 0, 1: 0}; tot = {0: 0, 1: 0}; car_side = 0
            for it in its:
                t = truth(it["file"])
                if t is None: continue
                vs = [it["views"][0]] if mode == "full" else it["views"]
                best = max(vs, key=lambda v: v["p"][B])
                tot[t] += 1
                if best["p"][B] >= th:
                    hit[t] += 1
            s = "%-12s %-5s th=%.1f  熊ありで鳴った %4d/%4d (%5.1f%%)" % (vid, mode, th, hit[1], tot[1], 100 * hit[1] / max(1, tot[1]))
            if tot[0]: s += "  熊なしで鳴った %4d/%4d (%5.1f%%)" % (hit[0], tot[0], 100 * hit[0] / tot[0])
            print(s)
