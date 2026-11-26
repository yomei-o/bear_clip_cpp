# C++ と Python の突き合わせ。
#  1) 同じ学習データから学習した層（head_cpp.json と head_py.json）の重み
#  2) 同じ画素（PIL で復号して PNG にしたもの）からの、タイルごとの特徴と点数
# usage: python parity.py bear_cli.exe weights.bin head_cpp.json head_py.json image...
import sys, os, json, subprocess, tempfile, numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import clipvit, bear
cli, wpath, hc, hp = sys.argv[1:5]; imgs = sys.argv[5:]
jc, jp = json.load(open(hc)), json.load(open(hp))
dw = max(np.abs(np.array(jc[k]) - np.array(jp[k])).max() for k in ("mean", "W"))
print("head: max|C++ - Py| = %.3e" % dw)
mc, mp = bear.load_model(hc), bear.load_model(hp)
w = clipvit.load_weights(wpath)
tmp = tempfile.mkdtemp(); worst_cos, worst_ds = 1.0, 0.0
for i, fn in enumerate(imgs):
    p = os.path.join(tmp, "%03d.png" % i); Image.open(fn).convert("RGB").save(p)
    out = subprocess.run([cli, "detect", wpath, hc, p, "--feat"], capture_output=True, text=True, check=True).stdout.strip().split("\n")
    rgb = np.asarray(Image.open(p).convert("RGB")); H, W = rgb.shape[:2]
    vs = bear.views(W, H, mp["scales"])
    assert len(vs) == len(out), (len(vs), len(out))
    for b, line in zip(vs, out):
        fc = np.array([float(v) for v in line.split()])
        x0, y0, x1, y1 = b
        fp = clipvit.encode_image(w, clipvit.preprocess(rgb[y0:y1, x0:x1], mode="pad"))
        c = float(fc @ fp / np.linalg.norm(fc) / np.linalg.norm(fp))
        ds = abs(bear.score(mc, bear.normalize(fc))[2] - bear.score(mp, bear.normalize(fp))[2])
        worst_cos = min(worst_cos, c); worst_ds = max(worst_ds, ds)
    print("%s: %d views  worst cos so far=%.8f  worst |d score|=%.2e" % (os.path.basename(fn), len(vs), worst_cos, worst_ds), flush=True)
ok = dw < 1e-10 and worst_cos > 0.99999 and worst_ds < 1e-3
print("WORST head=%.2e cos=%.8f dscore=%.2e -> %s" % (dw, worst_cos, worst_ds, "PASS" if ok else "FAIL"))
