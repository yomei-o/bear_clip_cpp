# 保存した画像特徴（clip_bear.py の *_feats.npz）を、別の候補一覧（classes.json）で採点し直す。
# usage: python rescore.py result.json classes.json out.json
import sys, json, os, numpy as np
d = json.load(open(sys.argv[1])); C = json.load(open(sys.argv[2], encoding="utf-8"))
z = np.load(os.path.splitext(sys.argv[1])[0] + "_feats.npz")
T = np.array([c["emb"] for c in C["classes"]], np.float32)
names = [c["name"] for c in C["classes"]]
for k, it in enumerate(d["items"]):
    F = z["arr_%d" % k].astype(np.float32)
    L = C["logit_scale"] * F @ T.T
    P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    for v, p in zip(it["views"], P): v["p"] = [round(float(x), 5) for x in p]
d["classes"] = names
json.dump(d, open(sys.argv[3], "w"))
print("rescored", len(d["items"]), "items with", len(names), "classes")
