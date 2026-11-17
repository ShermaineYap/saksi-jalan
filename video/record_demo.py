"""Record the live Saksi Jalan site on a phone viewport. Logs segment timestamps to marks.json."""
import json, os, time, shutil
from playwright.sync_api import sync_playwright

URL = "https://shermaine526-saksi-jalan.static.hf.space/index.html"
PROFILE = "/home/claude/video/chrome-profile"
VID = "/home/claude/video/rec"
proxy = os.environ.get("HTTPS_PROXY")
VW, VH = 390, 844
common = dict(executable_path="/usr/bin/google-chrome", args=["--ignore-certificate-errors"],
              proxy={"server": proxy} if proxy else None, viewport={"width": VW, "height": VH},
              device_scale_factor=2, is_mobile=True, has_touch=False)

def wait_verdict(pg):
    pg.wait_for_function("window.saksi && window.saksi.state.img && !document.querySelector('#verdict').hidden", timeout=300000)

def smooth_scroll_to(pg, selector, offset=0, steps=24, dur=0.9):
    y0 = pg.evaluate("window.scrollY")
    y1 = pg.evaluate(f"document.querySelector('{selector}').getBoundingClientRect().top + window.scrollY - {offset}")
    for i in range(1, steps + 1):
        t = i / steps; e = t * t * (3 - 2 * t)
        pg.evaluate(f"window.scrollTo(0, {y0 + (y1 - y0) * e})"); time.sleep(dur / steps)

with sync_playwright() as p:
    # 1) warm the HTTP cache (models) without recording
    ctx = p.chromium.launch_persistent_context(PROFILE, **common)
    pg = ctx.new_page(); pg.goto(URL); pg.locator("#samples button").nth(1).click(); wait_verdict(pg); ctx.close()
    print("warm", flush=True)
    shutil.rmtree(VID, ignore_errors=True)
    ctx = p.chromium.launch_persistent_context(PROFILE, record_video_dir=VID, record_video_size={"width": VW * 2, "height": VH * 2}, **common)
    pg = ctx.pages[0] if ctx.pages else ctx.new_page()
    pg.add_init_script("document.addEventListener('DOMContentLoaded',()=>{const s=document.createElement('style');s.textContent='html{scroll-behavior:auto!important}';document.head.appendChild(s)})")
    T0 = time.time(); marks = {}
    m = lambda k: marks.__setitem__(k, round(time.time() - T0, 2))
    pg.goto(URL, wait_until="networkidle"); pg.wait_for_timeout(800)
    # preload models silently so the first click is fast: no, keep honest; models come from cache
    m("a_start"); pg.wait_for_timeout(2500)
    smooth_scroll_to(pg, "#step-photo", 8); pg.wait_for_timeout(1200)
    pg.locator("#samples button").nth(0).click(); m("a_click")
    wait_verdict(pg); m("a_result"); pg.wait_for_timeout(300)
    smooth_scroll_to(pg, "#step-car", 8); pg.wait_for_timeout(700)
    smooth_scroll_to(pg, "#verdict", 330, dur=1.2); pg.wait_for_timeout(3500); m("a_end")
    # B: night photo -> not sure
    smooth_scroll_to(pg, "#samples", 260, dur=1.0); pg.wait_for_timeout(600); m("b_start")
    pg.locator("#samples button").nth(8).click()
    pg.wait_for_function("window.saksi.state.img && window.saksi.state.img.src.includes('myvi_7') && window.saksi.state.verdict && window.saksi.state.verdict.best", timeout=300000)
    pg.wait_for_timeout(300); m("b_result")
    smooth_scroll_to(pg, "#verdict", 300, dur=1.0); pg.wait_for_timeout(3500); m("b_end")
    # C: statement
    smooth_scroll_to(pg, "#step-statement", 8, dur=1.0); m("c_start"); pg.wait_for_timeout(600)
    for sel, txt in [("#f-where", "Jalan Ampang, near Taman Muda"), ("#f-plate", "W 7"), ("#f-dir", "towards Cheras")]:
        pg.click(sel); pg.keyboard.type(txt, delay=55); pg.wait_for_timeout(250)
    smooth_scroll_to(pg, "#form .btn", 420, dur=0.7); pg.click("#form .btn"); pg.wait_for_timeout(500)
    smooth_scroll_to(pg, "#out", 120, dur=1.2); pg.wait_for_timeout(2500)
    smooth_scroll_to(pg, "#ms", 300, dur=1.0); pg.wait_for_timeout(4000); m("c_end")
    vpath = pg.video.path(); ctx.close()
    json.dump({"marks": marks, "video": vpath}, open("/home/claude/video/marks.json", "w"), indent=1)
    print(marks, vpath)
