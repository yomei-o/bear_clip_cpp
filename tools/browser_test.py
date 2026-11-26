# docs/ をローカルで配信し、playwright でサンプルを押して判定とスクショを取る。
# usage: python tools/browser_test.py docs out_prefix [video]
import sys, threading, http.server, functools
from playwright.sync_api import sync_playwright
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
root, pre = sys.argv[1], sys.argv[2]; do_video = len(sys.argv) > 3
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)
H.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8768), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 900, "height": 1000})
    pg.on("console", lambda m: print("console:", m.text) if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: print("pageerror:", e))
    pg.goto("http://127.0.0.1:8768/index.html")
    pg.wait_for_function("document.getElementById('status').textContent.startsWith('準備')", timeout=300000)
    print(pg.text_content("#status"))
    btns = pg.locator("#samples button")
    for i in range(btns.count()):
        lab = btns.nth(i).text_content()
        if ("動画" in lab) != do_video: continue
        pg.evaluate("['status','vinfo','detail'].forEach(i=>document.getElementById(i).textContent='')")
        btns.nth(i).click()
        if do_video:
            pg.wait_for_function("document.getElementById('vinfo').textContent.length > 0 && document.getElementById('detail').textContent.length > 0", timeout=120000)
        else:
            pg.wait_for_function("document.getElementById('status').textContent.startsWith('判定しました')", timeout=600000)
        print(lab, "|", pg.text_content("#verdict"), "|", pg.text_content("#detail"), "|", pg.text_content("#status"))
        pg.screenshot(path="%s_%d.png" % (pre, i), full_page=False)
    b.close()
srv.shutdown()
