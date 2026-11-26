// bear.hpp — CLIP（clipvit.hpp）で写真の中の熊を探す。python/bear.py と同じ計算をする。
//   1. 画面全体と、短辺の 2/3・0.4 倍の正方形タイルに切る（views）
//   2. それぞれを CLIP に通し、2 つの点数を出す
//      ・言葉だけ（ゼロショット）: 「熊・車・人・犬…何もない」10 種類の言葉との softmax の「熊」
//      ・学習した層: タイルごとの「熊／熊以外」の線形 softmax
//   3. 2 つの平均がしきい値以上のタイルがあれば警報（比較実験で住宅街向けに一番釣り合いが良かった組み合わせ）
#pragma once
#include "clipvit.hpp"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

namespace bear {

constexpr double XS = 10.0;                       // 特徴の倍率（snow_depth_cpp と同じ）
constexpr double LAM = 3e-4; constexpr int EPOCHS = 3000; constexpr double LR = 0.4;

struct Box { int x0, y0, x1, y1; };

// 画面全体と、短辺の scale 倍の正方形タイル（縦横に 1/4 ずつ重ねる）。tools/clip_bear.py の views と同じ並び
inline std::vector<Box> views(int w, int h, const std::vector<double>& scales) {
    std::vector<Box> out{{0, 0, w, h}};
    for (double sc : scales) {
        int s = (int)(std::min(w, h) * sc);
        int nx = std::max(2, (int)std::ceil((w - s) / (s * 0.75)) + 1);
        int ny = std::max(2, (int)std::ceil((h - s) / (s * 0.75)) + 1);
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                int x = (int)std::nearbyint((double)i * (w - s) / (nx - 1));   // Python の round と同じ偶数丸め
                int y = (int)std::nearbyint((double)j * (h - s) / (ny - 1));
                out.push_back({x, y, x + s, y + s});
            }
    }
    return out;
}

inline clipvit::Image crop(const clipvit::Image& im, const Box& b) {
    clipvit::Image o; o.w = b.x1 - b.x0; o.h = b.y1 - b.y0; o.px.resize((size_t)o.w * o.h * 3);
    for (int y = 0; y < o.h; y++)
        std::memcpy(&o.px[(size_t)y * o.w * 3], &im.px[((size_t)(y + b.y0) * im.w + b.x0) * 3], (size_t)o.w * 3);
    return o;
}

struct Model {
    // 言葉
    std::vector<std::string> names; std::vector<std::vector<double>> text; double logit = 100; int alarm = 0;
    // 学習した層: z = ((f - mean) * XS)・W[:D] + W[D]、W は (D+1)×2
    std::vector<double> mean, W; int D = 512;
    double threshold = 0.7; std::vector<double> scales{0.667, 0.4};
};

inline std::vector<double> normalize(const std::vector<float>& f) {
    double n = 0; for (float v : f) n += (double)v * v; n = std::sqrt(n);
    std::vector<double> o(f.size()); for (size_t i = 0; i < f.size(); i++) o[i] = f[i] / n; return o;
}

struct Score { double zs, probe, s; };

// x: 正規化済みの特徴
inline Score score(const Model& m, const std::vector<double>& x) {
    Score r{};
    {   // 言葉
        std::vector<double> lg(m.text.size()); double mx = -1e300;
        for (size_t c = 0; c < m.text.size(); c++) {
            double s = 0; for (int i = 0; i < m.D; i++) s += x[i] * m.text[c][i];
            lg[c] = m.logit * s; mx = std::max(mx, lg[c]);
        }
        double sum = 0; for (auto& v : lg) { v = std::exp(v - mx); sum += v; }
        r.zs = lg[m.alarm] / sum;
    }
    {   // 学習した層
        double z0 = m.W[(size_t)m.D * 2], z1 = m.W[(size_t)m.D * 2 + 1];
        for (int i = 0; i < m.D; i++) { double xc = (x[i] - m.mean[i]) * XS; z0 += xc * m.W[(size_t)i * 2]; z1 += xc * m.W[(size_t)i * 2 + 1]; }
        double mx = std::max(z0, z1); double e0 = std::exp(z0 - mx), e1 = std::exp(z1 - mx);
        r.probe = e1 / (e0 + e1);
    }
    r.s = (r.zs + r.probe) / 2;
    return r;
}

inline Score score_box(const clipvit::Weights& w, const Model& m, const clipvit::Image& im, const Box& b) {
    return score(m, normalize(clipvit::encode_image(w, clipvit::preprocess(crop(im, b), true))));
}

// ---- 学習: 熊あり／なしの重みの合計がそれぞれ 0.5 になるよう重みをつけた 2 クラス softmax、全データの勾配降下（double）
// Xn: N×D の正規化済み特徴、y: 0/1。mean と W を m に入れる
inline void train(Model& m, const std::vector<float>& Xn, const std::vector<int>& y, int D,
                  double lam = LAM, int epochs = EPOCHS, double lr = LR) {
    const size_t N = y.size(); m.D = D;
    m.mean.assign(D, 0.0);
    for (size_t n = 0; n < N; n++) for (int i = 0; i < D; i++) m.mean[i] += Xn[n * D + i];
    for (int i = 0; i < D; i++) m.mean[i] /= (double)N;
    std::vector<double> X(N * D);
    for (size_t n = 0; n < N; n++) for (int i = 0; i < D; i++) X[n * D + i] = (Xn[n * D + i] - m.mean[i]) * XS;
    size_t npos = 0; for (int v : y) npos += v == 1;
    const double wp = 0.5 / std::max<size_t>(1, npos), wn = 0.5 / std::max<size_t>(1, N - npos);
    m.W.assign((size_t)(D + 1) * 2, 0.0);
    std::vector<double> G((size_t)(D + 1) * 2);
    for (int ep = 0; ep < epochs; ep++) {
        std::fill(G.begin(), G.end(), 0.0);
        for (size_t n = 0; n < N; n++) {
            const double* x = &X[n * D];
            double z0 = m.W[(size_t)D * 2], z1 = m.W[(size_t)D * 2 + 1];
            for (int i = 0; i < D; i++) { z0 += x[i] * m.W[(size_t)i * 2]; z1 += x[i] * m.W[(size_t)i * 2 + 1]; }
            double mx = std::max(z0, z1), e0 = std::exp(z0 - mx), e1 = std::exp(z1 - mx), p1 = e1 / (e0 + e1);
            double wt = y[n] ? wp : wn;
            double g1 = (p1 - (y[n] ? 1.0 : 0.0)) * wt, g0 = -g1;     // p0 - y0 = -(p1 - y1)
            for (int i = 0; i < D; i++) { G[(size_t)i * 2] += x[i] * g0; G[(size_t)i * 2 + 1] += x[i] * g1; }
            G[(size_t)D * 2] += g0; G[(size_t)D * 2 + 1] += g1;
        }
        for (size_t k = 0; k < G.size(); k++) {
            double g = G[k];
            if (k < (size_t)D * 2) g += lam * m.W[k];
            m.W[k] -= lr * g;
        }
    }
}

// ---- head.json（言葉のベクトル・層・しきい値・タイルの大きさ）
inline std::vector<double> vec_(const clipvit::Json& a) { std::vector<double> v; for (auto& x : a.arr) v.push_back(x.num); return v; }

inline void load_text(Model& m, const std::string& text) {   // tools/make_text.py の出力
    clipvit::Json j = clipvit::Json::parse(text);
    m.logit = j["logit_scale"].num; m.names.clear(); m.text.clear();
    const std::string alarm = j["alarm"].str;
    for (auto& c : j["classes"].arr) {
        if (c["name"].str == alarm) m.alarm = (int)m.names.size();
        m.names.push_back(c["name"].str); m.text.push_back(vec_(c["emb"]));
    }
}

inline Model load_model(const std::string& text) {
    clipvit::Json j = clipvit::Json::parse(text);
    Model m; m.D = (int)j["dim"].num; m.logit = j["logit_scale"].num; m.threshold = j["threshold"].num;
    m.scales = vec_(j["scales"]); m.mean = vec_(j["mean"]); m.W = vec_(j["W"]);
    const std::string alarm = j["alarm"].str;
    for (auto& c : j["classes"].arr) {
        if (c["name"].str == alarm) m.alarm = (int)m.names.size();
        m.names.push_back(c["name"].str); m.text.push_back(vec_(c["emb"]));
    }
    if ((int)m.mean.size() != m.D || m.W.size() != (size_t)(m.D + 1) * 2) throw std::runtime_error("head.json の形が合わない");
    return m;
}

inline std::string save_model(const Model& m, const std::string& note) {
    char b[64];
    auto num = [&](double v) { std::snprintf(b, sizeof b, "%.17g", v); return std::string(b); };
    auto arr = [&](const std::vector<double>& v) { std::string o = "["; for (size_t i = 0; i < v.size(); i++) o += (i ? "," : "") + num(v[i]); return o + "]"; };
    std::string o = "{\"note\":\"" + note + "\",\"dim\":" + std::to_string(m.D) + ",\"logit_scale\":" + num(m.logit) +
                    ",\"threshold\":" + num(m.threshold) + ",\"scales\":" + arr(m.scales) + ",\"alarm\":\"" + m.names[m.alarm] + "\",\"classes\":[";
    for (size_t c = 0; c < m.names.size(); c++) o += (c ? "," : "") + std::string("{\"name\":\"") + m.names[c] + "\",\"emb\":" + arr(m.text[c]) + "}";
    return o + "],\"mean\":" + arr(m.mean) + ",\"W\":" + arr(m.W) + "}";
}

// ---- 学習データ: "BRTF" u32 N u32 D float[N*D] i32 label[N]
struct Feats { int N = 0, D = 0; std::vector<float> F; std::vector<int> y; };
inline Feats load_feats(const std::string& b) {
    Feats f;
    if (b.size() < 12 || b.compare(0, 4, "BRTF") != 0) throw std::runtime_error("bad feats file");
    uint32_t N, D; std::memcpy(&N, b.data() + 4, 4); std::memcpy(&D, b.data() + 8, 4); f.N = (int)N; f.D = (int)D;
    size_t nf = (size_t)N * D;
    if (b.size() != 12 + nf * 4 + (size_t)N * 4) throw std::runtime_error("feats size mismatch");
    f.F.resize(nf); std::memcpy(f.F.data(), b.data() + 12, nf * 4);
    f.y.resize(N); std::memcpy(f.y.data(), b.data() + 12 + nf * 4, (size_t)N * 4);
    return f;
}

}  // namespace bear
