// ブラウザ用の入口。画像を 1 枚置いて、タイル（view）を 1 つずつ判定する（JS 側で複数の worker に配る）。
#include "bear.hpp"
#include <emscripten/emscripten.h>
#include <string>

static clipvit::Weights g_w;
static bear::Model g_m;
static clipvit::Image g_im;
static std::string g_json;

extern "C" {
EMSCRIPTEN_KEEPALIVE int bc_load_weights(const uint8_t* p, int n) {
    try { g_w = clipvit::Weights(); g_w.load(p, (size_t)n); return 1; } catch (...) { return 0; }
}
EMSCRIPTEN_KEEPALIVE int bc_load_head(const char* p, int n) {
    try { g_m = bear::load_model(std::string(p, (size_t)n)); return 1; } catch (...) { return 0; }
}
EMSCRIPTEN_KEEPALIVE double bc_threshold() { return g_m.threshold; }

// rgba: w*h*4 を RGB にして持っておく
EMSCRIPTEN_KEEPALIVE void bc_set_image(const uint8_t* rgba, int w, int h) {
    g_im.w = w; g_im.h = h; g_im.px.resize((size_t)w * h * 3);
    for (size_t i = 0; i < (size_t)w * h; i++) for (int c = 0; c < 3; c++) g_im.px[i * 3 + c] = rgba[i * 4 + c];
}

// この大きさの画像の view（画面全体＋タイル）を JSON [[x0,y0,x1,y1],...] で返す
EMSCRIPTEN_KEEPALIVE const char* bc_views(int w, int h) {
    auto vs = bear::views(w, h, g_m.scales);
    g_json = "[";
    for (size_t k = 0; k < vs.size(); k++)
        g_json += (k ? ",[" : "[") + std::to_string(vs[k].x0) + "," + std::to_string(vs[k].y0) + "," + std::to_string(vs[k].x1) + "," + std::to_string(vs[k].y1) + "]";
    g_json += "]";
    return g_json.c_str();
}

// 置いてある画像の box を判定して {"zs":..,"probe":..,"s":..} を返す
EMSCRIPTEN_KEEPALIVE const char* bc_eval(int x0, int y0, int x1, int y1) {
    auto s = bear::score_box(g_w, g_m, g_im, {x0, y0, x1, y1});
    char b[128]; std::snprintf(b, sizeof b, "{\"zs\":%.5f,\"probe\":%.5f,\"s\":%.5f}", s.zs, s.probe, s.s);
    g_json = b; return g_json.c_str();
}
}
