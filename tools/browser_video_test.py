# 手持ちの動画を選んだときの経路（0.5 秒おき・動いたタイルだけ判定）を playwright で試す。
# usage: python tools/browser_video_test.py docs video.mp4 out.png
import sys, threading, http.server, functools
from playwright.sync_api import sync_playwright
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
root, vid, out = sys.argv[1:4]
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root); H.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8769), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 900, "height": 1100})
    pg.on("pageerror", lambda e: print("pageerror:", e))
    pg.goto("http://127.0.0.1:8769/index.html")
    pg.wait_for_function("document.getElementById('status').textContent.startsWith('準備')", timeout=300000)
    pg.set_input_files("#fileVideo", vid)
    pg.wait_for_function("document.getElementById('status').textContent.startsWith('解析しました')", timeout=1200000)
    print(pg.text_content("#vinfo")); print(pg.text_content("#verdict"), "|", pg.text_content("#detail"))
    pg.evaluate("(async()=>{const v=document.getElementById('video'); v.currentTime=3; await new Promise(r=>v.onseeked=r);})()")
    pg.wait_for_timeout(1500); pg.evaluate("drawVideoFrame()"); pg.screenshot(path=out)
    b.close()
srv.shutdown()
