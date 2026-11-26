# 検証用画像（YOLO 形式の正解つき）で、画像ごとの警報の割合を出す。前回の bdet val と同じ「1 つでも鳴れば警報」。
# usage: python val_metrics.py result.json [threshold...]
import sys, json, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
d = json.load(open(sys.argv[1])); B = d["classes"].index("bear")
ths = [float(t) for t in sys.argv[2:]] or [0.3, 0.5, 0.7, 0.8, 0.9]
its = [it for it in d["items"] if it["label"] is not None]
y = np.array([it["label"] for it in its])
full = np.array([it["views"][0]["p"][B] for it in its])
big = np.array([max(v["p"][B] for v in it["views"] if not v["name"].startswith("s0.40")) for it in its])
allv = np.array([max(v["p"][B] for v in it["views"]) for it in its])
print("熊の画像 %d 枚、熊のいない画像 %d 枚" % ((y == 1).sum(), (y == 0).sum()))
for name, s in (("全体だけ", full), ("全体+大タイル", big), ("全体+大+小タイル", allv)):
    for th in ths:
        print("%-16s th=%.2f  熊で鳴った %5.1f%%  誤報 %5.1f%%" % (name, th, 100 * (s[y == 1] >= th).mean(), 100 * (s[y == 0] >= th).mean()))
