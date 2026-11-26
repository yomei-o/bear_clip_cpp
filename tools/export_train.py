# 学習データ（タイルごとの特徴と正解）を書き出す。C++ と Python が同じファイルから学習する。
# 形式: "BRTF" u32 N u32 D float[N*D] i32 label[N]   （正規化済みの特徴、label 1=熊 0=熊なし）
# usage: python export_train.py val.json labels_dir out.feats
import sys, os, json, struct, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(sys.argv[1])); labdir = sys.argv[2]
z = np.load(os.path.splitext(sys.argv[1])[0] + "_feats.npz")
def view_labels(it):
    vs = it["views"]
    if not it["label"]: return [0] * len(vs)
    W_, H_ = vs[0]["box"][2], vs[0]["box"][3]
    n = os.path.splitext(os.path.basename(it["file"].replace(chr(92), "/")))[0]
    boxes = []
    for line in open(os.path.join(labdir, n + ".txt")):
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
X, Y = [], []
for k, it in enumerate(d["items"]):
    F = z["arr_%d" % k].astype(np.float32); F /= np.linalg.norm(F, axis=1, keepdims=True)
    for f, l in zip(F, view_labels(it)):
        if l >= 0: X.append(f); Y.append(l)
X = np.array(X, np.float32); Y = np.array(Y, np.int32)
with open(sys.argv[3], "wb") as f:
    f.write(b"BRTF" + struct.pack("<II", *X.shape)); f.write(X.tobytes()); f.write(Y.tobytes())
print(X.shape, "positives", int(Y.sum()))
