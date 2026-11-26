# CLIP の画像特徴の上に「熊 / 熊以外」の線形層を学習して試す（snow_depth_cpp と同じ学習：中心化・10 倍・全データ勾配降下）。
# 学習は検証用画像（画面全体の特徴）だけ。5 分割 CV で検証用の数字を、全部で学習した層で動画とニュースの数字を出す。
# usage: python probe.py val.json video.json news.json [threshold...]
import sys, os, re, json, numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
XS, LAM, EPOCHS, LR = 10.0, 3e-4, 3000, 0.4
def feats(path):
    z = np.load(os.path.splitext(path)[0] + "_feats.npz"); return [z["arr_%d" % k].astype(np.float64) for k in range(len(z.files))]
VA, V, N = (json.load(open(p)) for p in sys.argv[1:4])
FVA, FV, FN = (feats(p) for p in sys.argv[1:4])
ths = [float(t) for t in sys.argv[4:]] or [0.5, 0.7, 0.9]
nz = lambda F: F / np.linalg.norm(F, axis=-1, keepdims=True)
y = np.array([it["label"] for it in VA["items"]])
LABDIR = "build/val_labels/labels"
def view_labels(it):
    """熊の画像: 枠の 6 割以上を含む view は 1、どの枠ともほぼ重ならない view は 0、それ以外は -1（使わない）。熊なし画像は全部 0"""
    vs = it["views"]
    if not it["label"]: return [0] * len(vs)
    W_, H_ = vs[0]["box"][2], vs[0]["box"][3]
    n = os.path.splitext(os.path.basename(it["file"].replace("\\", "/")))[0]
    boxes = []
    for line in open(os.path.join(LABDIR, n + ".txt")):
        _, cx, cy, bw, bh = map(float, line.split()[:5])
        boxes.append(((cx - bw / 2) * W_, (cy - bh / 2) * H_, (cx + bw / 2) * W_, (cy + bh / 2) * H_))
    out = []
    for v in vs:
        x0, y0, x1, y1 = v["box"]; best = 0.0
        for bx0, by0, bx1, by1 in boxes:
            iw = max(0, min(x1, bx1) - max(x0, bx0)); ih = max(0, min(y1, by1) - max(y0, by0))
            best = max(best, iw * ih / max(1e-6, (bx1 - bx0) * (by1 - by0)))
        out.append(1 if best >= 0.6 else 0 if best <= 0.05 else -1)
    return out
VL = [view_labels(it) for it in VA["items"]]
src = [os.path.basename(it["file"].replace("\\", "/")).split("_")[0] for it in VA["items"]]
# 同じカメラの画像が学習と評価に分かれないよう、LILA はデータセット名＋カメラらしき部分でまとめて分割する
def group(it):
    n = os.path.basename(it["file"].replace("\\", "/"))
    m = re.match(r"(lila_[^_]+(?:_[^_]+)?___[^_]+(?:_[^_]+)?)", n)
    return m.group(1) if m else n
import zlib
groups = np.array([zlib.crc32(group(it).encode()) % 5 for it in VA["items"]])   # hash() は実行ごとに変わるので使わない

def train(X, y):
    """熊あり／なしの重みの合計が同じになるように重みをつける"""
    mean = X.mean(0); Xc = (X - mean) * XS; N_, D = Xc.shape
    W = np.zeros((D + 1, 2)); Y = np.eye(2)[y]
    w = np.where(y == 1, 0.5 / max(1, (y == 1).sum()), 0.5 / max(1, (y == 0).sum()))[:, None]
    for _ in range(EPOCHS):
        z = Xc @ W[:-1] + W[-1]; p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        G = (p - Y) * w; g = np.vstack([Xc.T @ G, G.sum(0, keepdims=True)]); g[:-1] += LAM * W[:-1]; W -= LR * g
    return mean, W
def samples(idx):
    Xs, ys = [], []
    for i in idx:
        F = nz(FVA[i])
        for f, l in zip(F, VL[i]):
            if l >= 0: Xs.append(f); ys.append(l)
    return np.array(Xs), np.array(ys)
def prob(model, F):
    mean, W = model; z = ((F - mean) * XS) @ W[:-1] + W[-1]; z = z - z.max(-1, keepdims=True); p = np.exp(z); return p[..., 1] / p.sum(-1)

# CV（検証用画像、画面全体とタイルの最大）
cvs = np.zeros(len(y)); cv_views = [None] * len(y)
for k in range(5):
    tr = groups != k; te = ~tr
    m = train(*samples(np.where(tr)[0]))
    for i in np.where(te)[0]: cv_views[i] = prob(m, nz(FVA[i])); cvs[i] = cv_views[i].max()
full = train(*samples(range(len(y))))
# 組み合わせを試せるように、view ごとの層の点数を保存（検証用は CV、動画とニュースは全部で学習した層）
np.savez("build/probe_views.npz",
         val=np.array(cv_views, dtype=object), video=np.array([prob(full, nz(F)) for F in FV], dtype=object),
         news=np.array([prob(full, nz(F)) for F in FN], dtype=object), allow_pickle=True)
rows = {}
names = {"lila": "森のカメラ", "coco": "COCO 熊", "coconeg": "町の画像(COCO)", "xneg": "首都高の夜", "inat": "iNaturalist 熊"}
for i, it in enumerate(VA["items"]):
    nm = names.get(src[i], src[i]) + ((" 熊" if it["label"] else " 熊なし") if src[i] == "lila" else "")
    rows.setdefault((nm + " (CV)", it["label"]), []).append(cvs[i])
for i, it in enumerate(V["items"]):
    n = int(re.search(r"(\d+)\.jpg$", it["file"]).group(1)); s = prob(full, nz(FV[i])).max()
    if "surveillance" in it["file"]:
        t = 1 if (160 <= n <= 360 or 660 <= n <= 860) else 0 if (n <= 140 or 390 <= n <= 640 or n >= 890) else None
        nm = "玄関カメラ"
    else:
        t = 1 if n >= 21 else 0 if n <= 12 else None; nm = "図書館"
    if t is not None: rows.setdefault((nm + (" 熊あり" if t else " 熊なし"), t), []).append(s)
for i, it in enumerate(N["items"]): rows.setdefault(("日本のニュース 熊", 1), []).append(prob(full, nz(FN[i])).max())
print("%-26s %6s " % ("", "枚数") + " ".join("  th=%.1f" % t for t in ths))
for (nm, lab), xs in sorted(rows.items(), key=lambda kv: (-kv[0][1], kv[0][0])):
    xs = np.array(xs); print("%-26s %6d " % (nm, len(xs)) + " ".join("%7.1f%%" % (100 * (xs >= t).mean()) for t in ths))
np.savez("build/probe_full.npz", mean=full[0], W=full[1])
