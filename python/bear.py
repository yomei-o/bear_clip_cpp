"""bear: CLIP（clipvit.py）で写真の中の熊を探す（C++ 版 cpp/bear.hpp と同じ計算）。
  python bear.py train  train.feats text.json head.json [lam epochs lr]
  python bear.py detect weights.bin head.json image... [--views] [--feat]
"""
import sys, os, json, struct, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clipvit

XS = 10.0
LAM, EPOCHS, LR = 3e-4, 3000, 0.4

def views(w, h, scales):
    """画面全体と、短辺の scale 倍の正方形タイル（縦横に 1/4 ずつ重ねる）"""
    out = [(0, 0, w, h)]
    for sc in scales:
        s = int(min(w, h) * sc)
        nx = max(2, int(math.ceil((w - s) / (s * 0.75))) + 1); ny = max(2, int(math.ceil((h - s) / (s * 0.75))) + 1)
        for j in range(ny):
            for i in range(nx):
                x = round(i * (w - s) / (nx - 1)); y = round(j * (h - s) / (ny - 1))
                out.append((x, y, x + s, y + s))
    return out

def normalize(f):
    f = np.asarray(f, np.float64); return f / np.sqrt((f * f).sum())

def score(m, x):
    """x: 正規化済みの特徴 → (言葉の点数, 層の点数, 平均)"""
    lg = m["logit"] * (m["text"] @ x); lg = np.exp(lg - lg.max()); zs = lg[m["alarm"]] / lg.sum()
    z = ((x - m["mean"]) * XS) @ m["W"][:-1] + m["W"][-1]; z = np.exp(z - z.max()); probe = z[1] / z.sum()
    return float(zs), float(probe), float((zs + probe) / 2)

def score_box(w, m, rgb, box):
    x0, y0, x1, y1 = box
    return score(m, normalize(clipvit.encode_image(w, clipvit.preprocess(rgb[y0:y1, x0:x1], mode="pad"))))

def train(Xn, y, lam=LAM, epochs=EPOCHS, lr=LR):
    """熊あり／なしの重みの合計がそれぞれ 0.5 の 2 クラス softmax、全データの勾配降下"""
    mean = Xn.mean(0); X = (Xn - mean) * XS; N, D = X.shape
    W = np.zeros((D + 1, 2)); Y = np.eye(2)[y]
    w = np.where(y == 1, 0.5 / max(1, (y == 1).sum()), 0.5 / max(1, (y == 0).sum()))[:, None]
    for _ in range(epochs):
        z = X @ W[:-1] + W[-1]; p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        G = (p - Y) * w; g = np.vstack([X.T @ G, G.sum(0, keepdims=True)]); g[:-1] += lam * W[:-1]; W -= lr * g
    return mean, W

def load_feats(path):
    b = open(path, "rb").read(); assert b[:4] == b"BRTF"
    N, D = struct.unpack_from("<II", b, 4)
    F = np.frombuffer(b, np.float32, N * D, 12).reshape(N, D)
    y = np.frombuffer(b, np.int32, N, 12 + 4 * N * D)
    return F, y.astype(int)

def load_model(path):
    j = json.load(open(path, encoding="utf-8")); D = j["dim"]
    names = [c["name"] for c in j["classes"]]
    return {"names": names, "text": np.array([c["emb"] for c in j["classes"]]), "logit": j["logit_scale"],
            "alarm": names.index(j["alarm"]), "mean": np.array(j["mean"]), "W": np.array(j["W"]).reshape(D + 1, 2),
            "threshold": j["threshold"], "scales": j["scales"]}

def save_model(path, text_json, mean, W, note, threshold=0.7, scales=(0.667, 0.4)):
    t = json.load(open(text_json, encoding="utf-8"))
    j = {"note": note, "dim": len(mean), "logit_scale": t["logit_scale"], "threshold": threshold, "scales": list(scales),
         "alarm": t["alarm"], "classes": [{"name": c["name"], "emb": c["emb"]} for c in t["classes"]],
         "mean": [float(v) for v in mean], "W": [float(v) for v in W.ravel()]}
    json.dump(j, open(path, "w", encoding="utf-8"))

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = sys.argv[1:]
    if a and a[0] == "train" and len(a) in (4, 7):
        lam, ep, lr = (float(a[4]), int(a[5]), float(a[6])) if len(a) == 7 else (LAM, EPOCHS, LR)
        F, y = load_feats(a[1]); Xn = F.astype(np.float64)
        mean, W = train(Xn, y, lam, ep, lr)
        note = "trained by Python on %d views, lam=%g epochs=%d lr=%g" % (len(y), lam, ep, lr)
        save_model(a[3], a[2], mean, W, note); print(note, "->", a[3])
    elif a and a[0] == "detect" and len(a) >= 4:
        from PIL import Image
        w = clipvit.load_weights(a[1]); m = load_model(a[2])
        for fn in [x for x in a[3:] if not x.startswith("--")]:
            rgb = np.asarray(Image.open(fn).convert("RGB")); H, W_ = rgb.shape[:2]
            vs = views(W_, H, m["scales"])
            if "--feat" in a:   # 突き合わせ用: view ごとの特徴
                for b in vs:
                    x0, y0, x1, y1 = b
                    f = clipvit.encode_image(w, clipvit.preprocess(rgb[y0:y1, x0:x1], mode="pad"))
                    print(" ".join("%.6f" % v for v in f))
                continue
            sc = [score_box(w, m, rgb, b) for b in vs]
            k = int(np.argmax([s[2] for s in sc]))
            print("%s\tbear=%s score=%.4f (zs=%.4f probe=%.4f) box=%s" % (fn, "YES" if sc[k][2] >= m["threshold"] else "no", sc[k][2], sc[k][0], sc[k][1], vs[k]))
            if "--views" in a:
                for b, s in zip(vs, sc): print("  %s %.4f %.4f %.4f" % (b, *s))
    else:
        print(__doc__); sys.exit(1)
