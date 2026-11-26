# bear_clip_cpp — 町の熊チェッカー（CLIP、C++ / WebAssembly / Python）

防犯カメラの映像や写真に熊が写っているかを、CLIP で探します。
前回の YOLO の検出器 [bear_det_cpp](https://github.com/yomei-o/bear_det_cpp) が
「夜に停めてある黒い車を熊と言う」「防犯カメラの歩く熊を見逃す」で止まったので、CLIP でやり直しました。

**デモ**: https://yomei-o.github.io/bear_clip_cpp/
（写真・動画・カメラ。パブリックドメインの玄関カメラと図書館の映像入り。すべてブラウザの中で計算します）

> **研究・お試し用です。** 熊がいても鳴らないことも、熊がいなくても鳴ることもあります。
> 人の安全にかかわる判断（避難・通報など）には使わないでください。

## しくみ

1. 画面全体と、短辺の 2/3・0.4 倍の正方形タイル（縦横 1/4 ずつ重ねる。1280×720 なら合計 27 枚）に切る
2. それぞれを CLIP ViT-B/32（OpenAI、MIT）に通し、2 つの点数を出す
   - **言葉だけ（ゼロショット）**: 「熊・車・人・犬・猫・鹿・イノシシ・アライグマ・鳥・何もない」10 個の言葉の softmax の「熊」（`model/text_src.json`）
   - **学習した層**: タイルごとの「熊／熊以外」の線形 softmax（`model/head.json` の `mean` と `W`）
3. 2 つの平均が 0.7 以上のタイルがあれば警報。枠はそのタイル

CLIP のエンジン（`cpp/clipvit.hpp`, `python/clipvit.py`）は [oshare_cpp](https://github.com/yomei-o/oshare_cpp) と同じものです（torch と画素単位で一致）。

## 結果

試験は前回と同じ素材・同じ数え方です（1 コマ／1 枚ごとに、どれかのタイルが鳴ったか）。
動画 2 本と日本のニュースの画像は学習に一切使っていません。検証用画像は、同じカメラが学習と評価に分かれないように分けた 5 分割の交差検証です。

| | YOLO 2 回目 | 言葉だけ th0.9 | 層だけ th0.9 | **平均 th0.7（採用）** |
|---|---|---|---|---|
| 日本のニュース映像の熊（31 枚） | — | 100% | 100% | **100%** |
| 玄関カメラ：歩く熊のコマ（402） | 3% | 23% | 60% | **60%** |
| 図書館の熊のコマ（918） | 16% | 44% | 28% | **49%** |
| 観察写真の熊（iNaturalist、684） | 85%（町なかテスト） | 87% | 97% | 94% |
| 玄関カメラ：黒い車だけのコマ（516）＝誤報 | 8% | 0% | 0% | **0%** |
| 町の画像（COCO、571）＝誤報 | 4.1%（熊なし全体） | 1.8% | 3.0% | **2.8%** |
| 首都高の夜（150）＝誤報 | 同上 | 0% | 0% | **0%** |
| 森の自動撮影カメラの熊以外の動物（1,574）＝誤報 | 同上 | 66% | 17% | 44% |

- 防犯カメラの動画で鳴ったタイルを画像で確かめ、全部熊の位置でした（車のタイルで鳴ったものはない）。
- 日本のニュースの画像（テロップ入り）は全部見つかりました。
- 残っている課題は、森の自動撮影カメラに写る熊以外の動物（ヤマアラシ・クズリ・サルなど）での誤報です。

### わかったこと（試した順）

1. **画面全体を 1 枚で見せると、防犯カメラの熊は一度も鳴らない。** CLIP は 224px に縮めて見るので、小さい熊は消える。タイルが必須。
2. **言葉だけで、黒い車での誤報は 0 になった。** YOLO が「黒い大きな塊」で覚えたのと違い、CLIP は車と熊を意味で分ける。
3. **言葉の候補を 10 → 27 種類に増やすと、森の誤報は減るが、防犯カメラの熊まで「ほかの動物かも」に流れて見逃しが増えた**（玄関 48% → 19%、th0.7）。言葉だけではここが限界。
4. **画面全体の画像だけで学習した層をタイルに当てると、すべての画像で鳴った。** 拡大したタイルは学習で見ていない種類の画像だから。前回のラベル（熊の枠）を使い、枠の 6 割以上を含むタイルを熊、枠にほぼかからないタイルを熊以外としてタイルで学習し直した。
5. **言葉と層の組み合わせは、どれも全部では勝てない**（`tools/combos.py`）。平均・しきい値 0.7 は、層だけと比べて玄関カメラが同じまま図書館が 28% → 49%、町の誤報はほぼ同じで、森の誤報が 17% → 44% に増える。住宅街向けなのでこれを採用した。
6. 動画で「直近 5 コマ中 3 コマ」にしても、ほとんど変わらなかった（1 コマだけの誤報がもともと少ない）。

### 日本のニュースの画像について

評価用に、ユーザーが集めた日本のニュース映像の画像 31 枚（住宅街・道路・川・庭の熊、防犯カメラ・ドラレコ）を使いました。
著作権は各報道機関にあるので、このリポには入れていません。学習にも使っていません。
（最初にニュース記事から機械的に画像を集めましたが、熊が写っていたのは 1 割ほどで、使いませんでした。）

## 速さ

1 タイルの CLIP は、ネイティブ x64（MSVC /arch:AVX2、1 スレッド）で約 0.3 秒、1280×720 の 27 タイルで約 8 秒です。
ブラウザ（WASM SIMD）では 1 タイル約 1 秒なので、Web Worker を 2〜4 個立てて手分けします。
動画とカメラは 0.5〜1 秒おきに、前のコマから動いた部分にかかるタイルだけを見ます。
サンプル動画の枠は、C++ 版（`bear_cli frames`）で 5 コマ/秒・全タイルを前もって計算したものです。

## C++ と Python が同じであること

| 何を | 結果 | 道具 |
|---|---|---|
| 学習（同じ学習データ 89,448 タイル、3000 回） | 重みの差 1.3e-15、平均ベクトルは完全一致 | `bear_cli train` / `python/bear.py train` |
| 推論（同じ画素の 2 枚 × 27 タイル） | 特徴の cos ≥ 0.99999994、点数の差 ≤ 5.3e-6 | `tools/parity.py` |

## 使い方

```sh
# C++（Windows は VS2022）
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
build/Release/bear_cli detect models/vitb32_int8.bin model/head.json photo.jpg [--views]
build/Release/bear_cli train  train.feats model/text.json head.json
build/Release/bear_cli frames models/vitb32_int8.bin model/head.json frames_dir 6 out.json

# Python（numpy + Pillow。サブコマンドは C++ と同じ）
python python/bear.py detect models/vitb32_int8.bin model/head.json photo.jpg
python python/bear.py train  train.feats model/text.json head.json

# WASM（docs/bear.js）
./build_wasm.sh
```

重み `models/vitb32_int8.bin` は `docs/model/w0.bin` と `w1.bin` を連結したもの。

### 学習データを作り直す

学習データは bear_det_cpp の検証用データ（LILA BC・iNaturalist・COCO・首都高の夜、4,146 枚）です。再配布できないので、このリポには入れていません。

```sh
BEAR_SCALES=0.667,0.4 python tools/clip_bear.py val.json ViT-B-32-quickgelu openai <dataset2/val/images>   # タイルごとの特徴（open_clip、GPU）
python tools/export_train.py val.json <dataset2/val/labels> train.feats                                    # 熊の枠からタイルの正解を作る
python tools/make_text.py model/text_src.json model/text.json                                              # 言葉のベクトル
```

## ライセンス

- コード: MIT
- CLIP の重み: OpenAI CLIP（MIT）
- 学習した層（`model/head.json`）: 学習データの画像は入っていません（512 次元の平均と 513×2 の重みだけ）
- サンプル映像: Wikimedia Commons のパブリックドメイン
  - 玄関カメラ: [Surveillance camera captures bear in Franklin County…](https://commons.wikimedia.org/wiki/File:Surveillance_camera_captures_bear_in_Franklin_County_for_first_time_in_nearly_20_years.webm)（PD-CCTV）
  - 図書館: [Bear inside Mammoth Lakes Library](https://commons.wikimedia.org/wiki/File:Bear_inside_Mammoth_Lakes_Library.webm)（Mono County Office of Education）
- stb_image: MIT / パブリックドメイン
