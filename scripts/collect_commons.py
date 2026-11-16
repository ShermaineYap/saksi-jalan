"""Collect openly licensed car photos from Wikimedia Commons, with attribution.

Usage: python3 collect_commons.py OUT_DIR "Perodua Myvi" "Proton Saga" ...
Writes OUT_DIR/<class>/<n>.jpg and OUT_DIR/attribution.csv
Only keeps CC0, public domain, CC BY and CC BY-SA files.
"""
import csv, json, os, re, subprocess, sys, time, urllib.parse
from commons_count import files_in, UA, API

OK = re.compile(r"^(cc0|public domain|pd|cc by(-sa)? [0-9.]+|cc by(-sa)?)", re.I)
MAX_PER_CLASS = int(os.environ.get("MAX_PER_CLASS", "160"))

def curl_json(params):
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    for i in range(8):
        out = subprocess.run(["curl", "-s", "--max-time", "40", "-A", UA, url], capture_output=True, text=True).stdout
        try:
            return json.loads(out)
        except Exception:
            time.sleep(4 + 4 * i)
    raise RuntimeError("Commons API failed")

def strip(html):
    return re.sub(r"<[^>]+>", "", html or "").strip()[:120]

def main():
    out_dir, classes = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    done = set()
    if os.path.exists(f"{out_dir}/attribution.csv"):
        done = {r["class"] for r in csv.DictReader(open(f"{out_dir}/attribution.csv"))}
    for cls in classes:
        rows = []
        slug = cls.lower().replace(" ", "_")
        if slug in done:
            print("skip", slug); continue
        os.makedirs(f"{out_dir}/{slug}", exist_ok=True)
        titles = sorted(t for t in files_in("Category:" + cls) if t.lower().endswith((".jpg", ".jpeg", ".png")))
        kept = 0
        for i in range(0, len(titles), 50):
            if kept >= MAX_PER_CLASS:
                break
            r = curl_json({"action": "query", "prop": "imageinfo", "titles": "|".join(titles[i:i + 50]),
                           "iiprop": "url|extmetadata", "iiurlwidth": "384"})
            for p in r.get("query", {}).get("pages", {}).values():
                if kept >= MAX_PER_CLASS:
                    break
                ii = (p.get("imageinfo") or [{}])[0]
                md = ii.get("extmetadata", {})
                lic = md.get("LicenseShortName", {}).get("value", "")
                if not OK.match(lic) or "thumburl" not in ii:
                    continue
                path = f"{out_dir}/{slug}/{kept:04d}.jpg"
                res = subprocess.run(["curl", "-s", "-L", "--max-time", "40", "-A", UA, "-o", path,
                                      "-w", "%{http_code}", ii["thumburl"]], capture_output=True, text=True)
                if res.stdout != "200" or os.path.getsize(path) < 3000:
                    if os.path.exists(path):
                        os.remove(path)
                    time.sleep(1)
                    continue
                rows.append({"class": slug, "file": path, "title": p["title"], "license": lic,
                             "author": strip(md.get("Artist", {}).get("value")),
                             "source": ii.get("descriptionurl", "")})
                kept += 1
                time.sleep(0.15)
            time.sleep(1)
        print(f"{kept:4d} kept of {len(titles):4d}  {cls}", flush=True)
        with open(f"{out_dir}/attribution.csv", "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["class", "file", "title", "license", "author", "source"])
            if f.tell() == 0:
                w.writeheader()
            w.writerows(rows)
        time.sleep(5)

if __name__ == "__main__":
    main()
