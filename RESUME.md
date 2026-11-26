# RESUME — 作業の記録と次の一手

## 2026-11-26 開始
- 経緯: oshare_cpp（CLIP）を見たユーザーが「CLIP なら住宅街の熊を検知できるかも」。bear_det_cpp（YOLO）の続き。
- 比較は bear_det_cpp と同じ素材・同じ数え方（README の表）。組み合わせの比較のあと、ユーザーの希望でデモまで作った。

## 置き場所
- ビルド機 `C:\prog\bearclip`: `frames\`（2 本の動画の全コマ）、`user_news\`（ユーザー収集のニュース画像 31 枚。**再配布しない**）、
  `b32_*_ms.json` と `*_feats.npz`（タイルごとの点数と open_clip の特徴）、`model\bear_train.feats`（学習データ）。
  CLIP の重みは `C:\prog\snow\models\vitb32_int8.bin` を共用。
- 学習データの元: `C:\prog\rl\bear\dataset2\val`（bear_det_cpp の検証用、YOLO 形式の枠つき）。
- 手元のユーザー収集画像の元: `C:\Users\yomei2\Desktop\bear_news`。

## 落とし穴
- `hash()` は実行ごとに値が変わる。CV の分割には `zlib.crc32` を使う。
- Windows で書いた一覧ファイルは CRLF。シェルの `$(cat …)` で回すとファイル名の末尾に `\r` が付いて scp が失敗する。
- ssh 越しの `copy ... >nul & copy ...` が「パスが見つかりません」で失敗したことがある（原因未確認）。ファイルは scp で送る。
- 層の学習データは**タイル**で作る。画面全体だけで学習してタイルに当てると全部鳴る（README の 4）。
- open_clip の `ViT-B-32` + `openai` は QuickGELU にならない。`ViT-B-32-quickgelu` を使う（oshare_cpp と同じ）。

## 次の一手（候補）
1. 森の自動撮影カメラの「熊以外の動物」の誤報（44%）を減らす: 熊以外の動物のタイルを層の学習にもっと入れる、言葉の候補を絞って足す。
2. 日本の住宅街の熊の写真を増やして評価する（今は 31 枚）。防犯カメラ・夜・赤外線の白黒。
3. WASM の高速化（タイルを減らす工夫：動いた部分だけ、の精度を確かめる。matmul のブロック化）。
4. iPhone 実機で未確認（worker 2 個で 97MB×2 を読む）。
