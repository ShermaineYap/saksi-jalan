"""Download CC BY 4.0 Roboflow Universe datasets (YOLO format) and save one crop per labelled car.

Usage: ROBOFLOW_API_KEY=... python3 collect_roboflow.py OUT_DIR
Each crop goes to OUT_DIR/<our_class>/rf_<dataset>_<n>.jpg and a row is appended to
OUT_DIR/attribution.csv with the dataset as "author" (used as the group for train/test splits).
The API key is read from the environment, never written anywhere.
"""
import time, csv, glob, io, json, os, shutil, subprocess, sys, tempfile, zipfile
from PIL import Image

KEY = os.environ["ROBOFLOW_API_KEY"]
# (workspace/project, version, {their class name: our class slug})
DATASETS = [
    ("car-model-gbo55/perodua-axia", 2, {"Perodua Axia": "perodua_axia"}),
    ("axia-detection/perodua-axia-detection", 2, {"Axia": "perodua_axia"}),
    ("swinburne-crrcc/bezza", 3, {"perodua bezza": "perodua_bezza"}),
    ("trainlabelstxt/trainingmyvi", 1, {"trainmyvi": "perodua_myvi"}),
    ("axia-detection/car-model-validation", 1, {"Perodua Axia": "perodua_axia", "Perodua Myvi": "perodua_myvi",
                                                "Perodua Bezza": "perodua_bezza"}),
]
MAX_PER_DATASET_CLASS = int(os.environ.get("MAX_PER_DATASET_CLASS", "300"))
MIN_SIDE = 64
FIELDS = ["class", "file", "title", "license", "author", "source"]


def curl(url, out=None):
    args = ["curl", "-s", "-L", "--max-time", "600"] + (["-o", out] if out else []) + [url]
    return subprocess.run(args, capture_output=True, text=True).stdout


def main():
    out_dir = sys.argv[1]
    rows = []
    for proj, ver, cmap in DATASETS:
        slug = proj.replace("/", "__")
        if os.environ.get("ONLY") and os.environ["ONLY"] not in proj:
            continue
        for _ in range(30):
            meta = json.loads(curl(f"https://api.roboflow.com/{proj}/{ver}/yolov8?api_key={KEY}"))
            if meta.get("export", {}).get("link"):
                break
            time.sleep(10)
        link = meta.get("export", {}).get("link")
        if not link:
            print("no export for", proj, str(meta)[:200]); continue
        tmp = tempfile.mkdtemp(prefix="rf_")
        zpath = os.path.join(tmp, "data.zip")
        curl(link, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(os.path.join(tmp, "x"))
        names = None
        for y in glob.glob(os.path.join(tmp, "x", "**", "data.yaml"), recursive=True):
            for line in open(y):
                if line.startswith("names:"):
                    names = json.loads(line.split(":", 1)[1].strip().replace("'", '"'))
        counts = {}
        for img_path in sorted(glob.glob(os.path.join(tmp, "x", "**", "images", "*"), recursive=True)):
            lbl = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep).rsplit(".", 1)[0] + ".txt"
            if not os.path.exists(lbl):
                continue
            try:
                im = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            W, H = im.size
            for line in open(lbl):
                p = line.split()
                if len(p) != 5:
                    continue  # skip polygons
                cname = names[int(p[0])] if names else None
                ours = cmap.get(cname)
                if not ours or counts.get(ours, 0) >= MAX_PER_DATASET_CLASS:
                    continue
                cx, cy, w, h = (float(v) for v in p[1:])
                x0, y0 = max(0, (cx - w / 2 * 1.1) * W), max(0, (cy - h / 2 * 1.1) * H)
                x1, y1 = min(W, (cx + w / 2 * 1.1) * W), min(H, (cy + h / 2 * 1.1) * H)
                if min(x1 - x0, y1 - y0) < MIN_SIDE:
                    continue
                os.makedirs(f"{out_dir}/{ours}", exist_ok=True)
                n = counts.get(ours, 0)
                path = f"{out_dir}/{ours}/rf_{slug}_{n:04d}.jpg"
                im.crop((x0, y0, x1, y1)).save(path, quality=92)
                counts[ours] = n + 1
                rows.append({"class": ours, "file": path, "title": os.path.basename(img_path)[:120],
                             "license": "CC BY 4.0", "author": f"Roboflow Universe: {proj}",
                             "source": f"https://universe.roboflow.com/{proj}"})
        shutil.rmtree(tmp, ignore_errors=True)
        print(proj, counts, flush=True)
    with open(f"{out_dir}/attribution.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writerows(rows)


if __name__ == "__main__":
    main()
