// 推論はここ。ページは同じ worker を何個か立て、タイルを手分けして判定させる。
importScripts("bear.js");
let M = null;

async function fetchAll(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(url + " " + res.status);
  const reader = res.body.getReader(); const chunks = []; let n = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; chunks.push(value); n += value.length; onBytes(value.length); }
  const out = new Uint8Array(n); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

async function init(id) {
  M = await BearModule();
  let got = 0; const TOTAL = 96931781;
  const bufs = await Promise.all(["model/w0.bin", "model/w1.bin"].map((u) => fetchAll(u, (k) => {
    got += k; if (id === 0) postMessage({ type: "progress", got, total: TOTAL });
  })));
  const n = bufs.reduce((a, b) => a + b.length, 0);
  const p = M._malloc(n); let o = 0; for (const b of bufs) { M.HEAPU8.set(b, p + o); o += b.length; }
  const okW = M._bc_load_weights(p, n); M._free(p);
  const hj = new Uint8Array(await (await fetch("model/head.json")).arrayBuffer());
  const q = M._malloc(hj.length); M.HEAPU8.set(hj, q); const okH = M._bc_load_head(q, hj.length); M._free(q);
  if (!okW || !okH) throw new Error("モデルの読み込みに失敗しました");
  postMessage({ type: "ready", threshold: M._bc_threshold() });
}

onmessage = (e) => {
  const d = e.data;
  try {
    if (d.type === "init") { init(d.id).catch((err) => postMessage({ type: "error", message: String(err) })); return; }
    if (d.type === "image") {   // 画像を置く
      const p = M._malloc(d.rgba.length); M.HEAPU8.set(d.rgba, p); M._bc_set_image(p, d.w, d.h); M._free(p);
      postMessage({ type: "imageSet", job: d.job });
      return;
    }
    if (d.type === "views") { postMessage({ type: "views", req: d.req, views: JSON.parse(M.UTF8ToString(M._bc_views(d.w, d.h))) }); return; }
    if (d.type === "eval") {    // 置いた画像のタイル 1 つ
      const [x0, y0, x1, y1] = d.box;
      const r = JSON.parse(M.UTF8ToString(M._bc_eval(x0, y0, x1, y1)));
      postMessage({ type: "result", job: d.job, k: d.k, r });
    }
  } catch (err) { postMessage({ type: "error", message: String(err) }); }
};
