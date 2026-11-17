import os, sys, json, time, urllib.parse
from playwright.sync_api import sync_playwright
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/"
proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
samples = ["bezza_1","myvi_4","saga_7","persona_3","iriz_3","axia_7","bezza_2","axia_3","myvi_7"]
with sync_playwright() as p:
    kw = {"executable_path": "/usr/bin/google-chrome", "args": ["--ignore-certificate-errors"]}
    if proxy and not URL.startswith("http://127."): kw["proxy"] = {"server": proxy, "bypass": "127.0.0.1,localhost"}
    elif proxy: kw["proxy"] = {"server": proxy, "bypass": "127.0.0.1,localhost"}
    b = p.chromium.launch(**kw)
    pg = b.new_page(viewport={"width": 420, "height": 900}, device_scale_factor=2)
    logs = []; pg.on("console", lambda m: logs.append(m.text)); pg.on("pageerror", lambda e: logs.append("PAGEERROR " + str(e)))
    pg.goto(URL, wait_until="domcontentloaded"); pg.wait_for_timeout(1500)
    pg.screenshot(path="/tmp/site_top.png", full_page=False)
    out = {}
    for i, s in enumerate(samples):
        t = time.time()
        pg.locator("#samples button").nth(i).click()
        pg.wait_for_function("document.querySelector('#verdict') && !document.querySelector('#verdict').hidden && window.saksi.state.img && (window.saksi.state.verdict || document.querySelector('#verdict').classList.contains('none'))", timeout=600000)
        pg.wait_for_timeout(300)
        v = pg.evaluate("window.saksi.state.verdict")
        out[s] = {"top3": [[n, round(p, 3)] for n, p in v["top3"]] if v else None, "sure": v and v["confident"], "colour": v and v["colourEn"],
                  "cars": pg.evaluate("window.saksi.state.cars.length"), "ms": round(pg.evaluate("window.saksiLastMs") or 0), "wall_s": round(time.time() - t, 1)}
        print(s, out[s], flush=True)
        if i == 0: pg.screenshot(path="/tmp/site_car.png", full_page=True)
    pg.fill("#f-where", "Jalan Ampang"); pg.fill("#f-plate", "wa 1"); pg.click("#form .btn"); pg.wait_for_timeout(300)
    print("EN:", pg.inner_text("#en")[:160]); print("MS:", pg.inner_text("#ms")[:160])
    pg.screenshot(path="/tmp/site_full.png", full_page=True)
    json.dump(out, open("/tmp/site_results.json", "w"), indent=1)
    errs = [l for l in logs if "PAGEERROR" in l or "rror" in l]; print("console errors:", errs[:5])
    b.close()
