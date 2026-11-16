"""Collect openly licensed car photos via the Openverse API (mostly Flickr), with attribution.

Usage: python3 collect_openverse.py OUT_DIR "Perodua Myvi" "Proton Saga" ...
Writes OUT_DIR/<class>/ov_<n>.jpg and appends to OUT_DIR/attribution.csv.
Keeps only CC BY, CC BY-SA, CC0 and Public Domain Mark. Skips Wikimedia results
(collected separately by collect_commons.py) to avoid duplicates.
"""
import csv, json, os, subprocess, sys, time, urllib.parse

API = "https://api.openverse.org/v1/images/"
UA = "SaksiJalan/0.1 (student hackathon project; github.com/ShermaineYap/saksi-jalan)"
MAX_PER_CLASS = int(os.environ.get("MAX_PER_CLASS", "240"))
FIELDS = ["class", "file", "title", "license", "author", "source"]


def curl(url, out=None):
    args = ["curl", "-s", "-L", "--max-time", "40", "-A", UA, "-w", "%{http_code}"]
    if out:
        args += ["-o", out]
    r = subprocess.run(args + [url], capture_output=True, text=True)
    return (r.stdout[-3:], r.stdout[:-3]) if not out else (r.stdout, None)


QUERIES = {
    "perodua_myvi": ["perodua myvi", "myvi"],
    "perodua_axia": ["perodua axia", "axia"],
    "perodua_bezza": ["perodua bezza", "bezza"],
    "proton_saga": ["proton saga"],
    "proton_x70": ["proton x70", "x70 proton"],
    "proton_persona": ["proton persona", "persona proton"],
    "proton_iriz": ["proton iriz", "iriz"],
}


def search(q, page):
    url = API + "?" + urllib.parse.urlencode({"q": q, "license": "by,by-sa,cc0,pdm", "source": "flickr",
                                              "page_size": 20, "page": page})
    for i in range(8):
        code, body = curl(url)
        if code == "200":
            return json.loads(body)
        if code in ("400", "401", "404"):
            return {"results": []}
        time.sleep(10 + 10 * i)
    return {"results": []}


def main():
    out_dir, classes = sys.argv[1], sys.argv[2:]
    seen_titles = set()
    if os.path.exists(f"{out_dir}/attribution.csv"):
        for r in csv.DictReader(open(f"{out_dir}/attribution.csv")):
            seen_titles.add(r["source"])
    for cls in classes:
        slug = cls.lower().replace(" ", "_")
        os.makedirs(f"{out_dir}/{slug}", exist_ok=True)
        if any(f.startswith("ov_") for f in os.listdir(f"{out_dir}/{slug}")):
            print("skip", slug, flush=True)
            continue
        rows, kept = [], 0
        pages = [(q, p) for q in QUERIES.get(slug, [cls]) for p in range(1, 13)]
        empty_q = set()
        for q, page in pages:
            if q in empty_q:
                continue
            res = search(q, page).get("results", [])
            if not res:
                empty_q.add(q)
                continue
            for r in res:
                if kept >= MAX_PER_CLASS:
                    break
                if r.get("source") == "wikimedia" or r.get("foreign_landing_url") in seen_titles:
                    continue
                path = f"{out_dir}/{slug}/ov_{kept:04d}.jpg"
                code, _ = curl(r["thumbnail"], path)
                if code != "200" or not os.path.exists(path) or os.path.getsize(path) < 3000:
                    if os.path.exists(path):
                        os.remove(path)
                    time.sleep(1)
                    continue
                seen_titles.add(r.get("foreign_landing_url"))
                rows.append({"class": slug, "file": path, "title": (r.get("title") or "")[:120],
                             "license": f"CC {r['license'].upper()} {r.get('license_version') or ''}".strip(),
                             "author": (r.get("creator") or "")[:120],
                             "source": r.get("foreign_landing_url", "")})
                kept += 1
                time.sleep(0.5)
            time.sleep(3)
        with open(f"{out_dir}/attribution.csv", "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if f.tell() == 0:
                w.writeheader()
            w.writerows(rows)
        print(f"{kept:4d} kept  {cls}", flush=True)


if __name__ == "__main__":
    main()
