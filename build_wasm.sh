#!/bin/sh
# WASM をビルドして docs/bear.js（wasm 埋め込みの 1 ファイル）を作る。
set -e
cd "$(dirname "$0")"
EMSDK=${EMSDK:-/c/prog/emsdk/emsdk}
export EM_CONFIG="$EMSDK/.emscripten"
EMCC="$EMSDK/upstream/emscripten/em++.exe"
"$EMCC" -O3 -msimd128 -std=c++17 -Icpp cpp/wasm_api.cpp -o docs/bear.js \
  -sSINGLE_FILE=1 -sMODULARIZE=1 -sEXPORT_NAME=BearModule -sALLOW_MEMORY_GROWTH=1 \
  -sINITIAL_MEMORY=256MB -sMAXIMUM_MEMORY=2GB -sENVIRONMENT=web,worker,node \
  -sEXPORTED_FUNCTIONS=_bc_load_weights,_bc_load_head,_bc_threshold,_bc_set_image,_bc_views,_bc_eval,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,UTF8ToString
ls -la docs/bear.js
