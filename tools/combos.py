# 言葉だけ（ゼロショット）と学習した層を、タイルごとにいろいろな方法で組み合わせて比べる。
# usage: python combos.py   （build/ の b32_*_ms.json, c27_*.json, probe_views.npz を使う）
import sys, json, os, re, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
def zs(path):
    d = json.load(open(path)); B = d["classes"].index("bear")
    return d["items"], [np.array([v["p"][B] for v in it["views"]]) for it in d["items"]]
items = {}; Z10 = {}; Z27 = {}
for k in ("val", "video", "news"):
    items[k], Z10[k] = zs("build/b32_%s_ms.json" % k)
    _, Z27[k] = zs("build/c27_%s.json" % k)
PV = np.load("build/probe_views.npz", allow_pickle=True)
PR = {k: [np.asarray(x, dtype=float) for x in PV[k]] for k in ("val", "video", "news")}

METHODS = {
    "言葉10": lambda a, b, p: a,
    "言葉27": lambda a, b, p: b,
    "層": lambda a, b, p: p,
    "平均(言葉10,層)": lambda a, b, p: (a + p) / 2,
    "幾何平均(言葉10,層)": lambda a, b, p: np.sqrt(a * p),
    "AND=min(言葉10,層)": lambda a, b, p: np.minimum(a, p),
    "OR=max(言葉10,層)": lambda a, b, p: np.maximum(a, p),
    "平均(言葉27,層)": lambda a, b, p: (b + p) / 2,
    "幾何平均(言葉27,層)": lambda a, b, p: np.sqrt(b * p),
}

def vtruth(f):
    n = int(re.search(r"(\d+)\.jpg$", f).group(1))
    if "surveillance" in f:
        return ("玄関", 1 if (160 <= n <= 360 or 660 <= n <= 860) else 0 if (n <= 140 or 390 <= n <= 640 or n >= 890) else None, n)
    return ("図書館", 1 if n >= 21 else 0 if n <= 12 else None, n)

names = {"lila": "森", "coco": "COCO熊", "coconeg": "町COCO", "xneg": "首都高夜", "inat": "iNat熊"}
def evaluate(fn, th, temporal=False):
    r = {}
    def add(key, hit): r.setdefault(key, []).append(hit)
    for it, a, b, p in zip(items["val"], Z10["val"], Z27["val"], PR["val"]):
        src = os.path.basename(it["file"].replace("\\", "/")).split("_")[0]
        key = names.get(src, src) + (("熊" if it["label"] else "なし") if src == "lila" else "")
        add(key, fn(a, b, p).max() >= th)
    # 動画はコマの順に並んでいる前提で、temporal なら直近 5 コマ中 3 コマ以上で警報
    for vid in ("玄関", "図書館"):
        seq = [(vtruth(it["file"]), fn(a, b, p).max()) for it, a, b, p in zip(items["video"], Z10["video"], Z27["video"], PR["video"]) if vtruth(it["file"])[0] == vid]
        seq.sort(key=lambda x: x[0][2])
        raw = np.array([s >= th for _, s in seq])
        if temporal:
            raw = np.array([raw[max(0, i - 4):i + 1].sum() >= 3 for i in range(len(raw))])
        for ((_, t, _), hit) in zip([x[0] for x in seq], raw):
            if t is not None: add(vid + ("熊" if t else "なし"), hit)
    for a, b, p in zip(Z10["news"], Z27["news"], PR["news"]): add("ニュース熊", fn(a, b, p).max() >= th)
    return {k: 100 * np.mean(v) for k, v in r.items()}

cols = ["ニュース熊", "玄関熊", "図書館熊", "iNat熊", "森熊", "玄関なし", "町COCO", "首都高夜", "森なし"]
print("見つけた割合（熊）と誤報の割合（なし）。玄関なし＝黒い車のコマ")
for temporal in (False, True):
    print("\n==== 動画の規則: %s" % ("直近5コマ中3コマ以上" if temporal else "1コマごと"))
    print("%-22s %4s " % ("方法", "th") + " ".join("%8s" % c for c in cols))
    for name, fn in METHODS.items():
        for th in (0.5, 0.7, 0.9):
            r = evaluate(fn, th, temporal)
            print("%-22s %4.1f " % (name, th) + " ".join("%7.1f%%" % r[c] for c in cols))
