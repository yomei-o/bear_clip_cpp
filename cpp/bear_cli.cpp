// bear_cli — 写真の中の熊を探す（python/bear.py の CLI と同じ）。
//   bear_cli train  train.feats text.json head.json [lam epochs lr]
//   bear_cli detect weights.bin head.json image... [--views] [--feat]
//   bear_cli frames weights.bin head.json dir step out.json   コマ（dir/0001.jpg…）を step 枚おきに全 view 判定して保存
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "bear.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <chrono>

static std::string slurp(const std::string& fn) {
    std::ifstream f(fn, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + fn);
    std::ostringstream ss; ss << f.rdbuf(); return ss.str();
}
static void spit(const std::string& fn, const std::string& s) {
    std::ofstream f(fn, std::ios::binary); if (!f) throw std::runtime_error("cannot write " + fn); f << s;
}
static clipvit::Weights load_w(const std::string& fn) {
    std::string b = slurp(fn); clipvit::Weights w; w.load((const uint8_t*)b.data(), b.size()); return w;
}
static bool load_img(const std::string& fn, clipvit::Image& im) {
    int W, H, Cn; unsigned char* d = stbi_load(fn.c_str(), &W, &H, &Cn, 3);
    if (!d) return false;
    im.w = W; im.h = H; im.px.assign(d, d + (size_t)W * H * 3); stbi_image_free(d); return true;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: bear_cli train|detect|frames ...\n"); return 1; }
    std::string cmd = argv[1];
    try {
        if (cmd == "train" && (argc == 5 || argc == 8)) {
            double lam = argc == 8 ? std::atof(argv[5]) : bear::LAM;
            int epochs = argc == 8 ? std::atoi(argv[6]) : bear::EPOCHS;
            double lr = argc == 8 ? std::atof(argv[7]) : bear::LR;
            bear::Feats fs = bear::load_feats(slurp(argv[2]));
            bear::Model m; bear::load_text(m, slurp(argv[3]));
            auto t0 = std::chrono::steady_clock::now();
            bear::train(m, fs.F, fs.y, fs.D, lam, epochs, lr);
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            char note[160];
            std::snprintf(note, sizeof note, "trained by C++ on %d views, lam=%g epochs=%d lr=%g", fs.N, lam, epochs, lr);
            spit(argv[4], bear::save_model(m, note)); std::printf("%s -> %s (%.0f s)\n", note, argv[4], sec);
        } else if (cmd == "detect" && argc >= 5) {
            clipvit::Weights w = load_w(argv[2]);
            bear::Model m = bear::load_model(slurp(argv[3]));
            bool showv = false, feat = false;
            for (int i = 4; i < argc; i++) { std::string s = argv[i]; if (s == "--views") showv = true; if (s == "--feat") feat = true; }
            for (int i = 4; i < argc; i++) {
                if (std::string(argv[i]).rfind("--", 0) == 0) continue;
                clipvit::Image im; if (!load_img(argv[i], im)) { std::fprintf(stderr, "cannot load %s\n", argv[i]); continue; }
                auto vs = bear::views(im.w, im.h, m.scales);
                if (feat) {
                    for (auto& b : vs) {
                        auto f = clipvit::encode_image(w, clipvit::preprocess(bear::crop(im, b), true));
                        for (size_t k = 0; k < f.size(); k++) std::printf(k ? " %.6f" : "%.6f", f[k]);
                        std::printf("\n");
                    }
                    continue;
                }
                auto t0 = std::chrono::steady_clock::now();
                std::vector<bear::Score> sc; size_t best = 0;
                for (auto& b : vs) { sc.push_back(bear::score_box(w, m, im, b)); if (sc.back().s > sc[best].s) best = sc.size() - 1; }
                double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                auto& b = vs[best];
                std::printf("%s\tbear=%s score=%.4f (zs=%.4f probe=%.4f) box=(%d, %d, %d, %d)\t(%zu views, %.0f ms)\n", argv[i],
                            sc[best].s >= m.threshold ? "YES" : "no", sc[best].s, sc[best].zs, sc[best].probe, b.x0, b.y0, b.x1, b.y1, vs.size(), ms);
                if (showv) for (size_t k = 0; k < vs.size(); k++)
                    std::printf("  (%d, %d, %d, %d) %.4f %.4f %.4f\n", vs[k].x0, vs[k].y0, vs[k].x1, vs[k].y1, sc[k].zs, sc[k].probe, sc[k].s);
            }
        } else if (cmd == "frames" && argc == 7) {
            clipvit::Weights w = load_w(argv[2]);
            bear::Model m = bear::load_model(slurp(argv[3]));
            std::string dir = argv[4]; int step = std::atoi(argv[5]);
            std::string o = "{\"step\":" + std::to_string(step) + ",\"frames\":[";
            int nout = 0;
            for (int n = 1;; n += step) {
                char fn[64]; std::snprintf(fn, sizeof fn, "/%04d.jpg", n);
                clipvit::Image im; if (!load_img(dir + fn, im)) break;
                auto vs = bear::views(im.w, im.h, m.scales);
                o += (nout++ ? "," : "") + std::string("{\"frame\":") + std::to_string(n) + ",\"views\":[";
                for (size_t k = 0; k < vs.size(); k++) {
                    auto s = bear::score_box(w, m, im, vs[k]);
                    char b[160]; std::snprintf(b, sizeof b, "%s[%d,%d,%d,%d,%.4f]", k ? "," : "", vs[k].x0, vs[k].y0, vs[k].x1, vs[k].y1, s.s);
                    o += b;
                }
                o += "]}";
                if (nout % 20 == 0) std::fprintf(stderr, "%d frames\n", nout);
            }
            o += "]}";
            spit(argv[6], o); std::printf("%d frames -> %s\n", nout, argv[6]);
        } else { std::fprintf(stderr, "bad arguments\n"); return 1; }
    } catch (std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
    return 0;
}
