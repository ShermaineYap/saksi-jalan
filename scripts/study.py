"""Measured study for Saksi Jalan (Day 2).

- Exterior filter (CLIP zero-shot) and near-duplicate removal.
- 5-fold StratifiedGroupKFold, grouped by photographer / dataset, so every photo is
  tested exactly once by a model that never saw its photographer or dataset.
- Compares CLIP zero-shot with a linear probe on CLIP image embeddings.
- Degradation study on the same out-of-fold predictions.
- Refusal curve (accuracy vs coverage).
- Independent held-out test: Roboflow "car-model-validation" (never used for training).
- Trains the final probe on everything except the held-out set and saves it.

Usage: python3 study.py DATA_DIR OUT_DIR
"""
import csv, io, json, os, sys
import numpy as np, torch, open_clip, joblib
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix

torch.set_num_threads(os.cpu_count() or 2)
np.random.seed(0)
DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
NAMES = {"perodua_axia": "Perodua Axia", "perodua_bezza": "Perodua Bezza", "perodua_myvi": "Perodua Myvi",
         "proton_iriz": "Proton Iriz", "proton_persona": "Proton Persona", "proton_saga": "Proton Saga"}
# Proton X70 is left out: only 17 usable exterior photos, nearly all from one photographer.
CLS = sorted(NAMES)
HOLDOUT = "car-model-validation"

ARCH, WEIGHTS = os.environ.get("BACKBONE", "ViT-B-32/laion2b_s34b_b79k").split("/")
QUICK = os.environ.get("QUICK") == "1"  # skip degradations (backbone comparison)
model, _, pre = open_clip.create_model_and_transforms(ARCH, pretrained=WEIGHTS)
tok = open_clip.get_tokenizer(ARCH)
model.eval()


def embed(imgs, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            f = model.encode_image(torch.stack([pre(im) for im in imgs[i:i + bs]]))
            out.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(out).numpy()


def embed_text(t):
    with torch.no_grad():
        f = model.encode_text(tok(t))
    return (f / f.norm(dim=-1, keepdim=True)).numpy()


def jpeg(im, q):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


DEG_ALL = {
    "blur_r4": lambda im: im.filter(ImageFilter.GaussianBlur(4)),
    "dark_0.35": lambda im: ImageEnhance.Brightness(im).enhance(0.35),
    "lowres_64px": lambda im: im.resize((64, max(1, int(64 * im.height / im.width)))).resize(im.size),
    "dashcam_combo": lambda im: jpeg(ImageEnhance.Brightness(
        im.resize((96, max(1, int(96 * im.height / im.width)))).resize(im.size).filter(ImageFilter.GaussianBlur(1.5))).enhance(0.5), 20),
}

DEG = {} if QUICK else DEG_ALL

# ---------- load, embed (cached) ----------
rows = [r for r in csv.DictReader(open(f"{DATA}/attribution.csv")) if r["class"] in NAMES]
rows = [r for r in rows if os.path.exists(r["file"] if r["file"].startswith("/") else f"{DATA}/{r['file']}")]
for r in rows:
    r["path"] = r["file"] if r["file"].startswith("/") else f"{DATA}/{r['file']}"
imgs = [Image.open(r["path"]).convert("RGB") for r in rows]
cache = f"{os.environ.get('CACHE_DIR', OUT)}/emb_clean.npy"
E = np.load(cache) if os.path.exists(cache) and len(np.load(cache)) == len(rows) else embed(imgs)
np.save(cache, E)
print("loaded", len(rows))

scene = ["a photo of the outside of a car on a road or parking lot", "a photo of a car interior or dashboard",
         "a photo of a car engine", "a close-up photo of a car logo or badge", "a photo of a car wheel close-up",
         "a photo of a crowd at a motor show", "a photo of a car seat", "a photo with no car in it",
         "a toy model car or a drawing of a car"]
KEEP_FROM = os.environ.get("KEEP_FROM")  # reuse another run's filtered photo set (fair backbone comparison)
is_ext = (E @ embed_text(scene).T).argmax(1) == 0
# Roboflow crops come from human-labelled car bounding boxes, so they are always whole cars.
# CLIP tends to call white-background listing photos "no car" or "a drawing", so don't filter them.
is_ext |= np.array(["Roboflow" in r["author"] for r in rows])
idx = []
for i in np.where(is_ext)[0]:
    if idx and float(np.max(E[idx] @ E[i])) > 0.97:
        continue
    idx.append(i)
idx = np.array(idx)
if KEEP_FROM:
    keep = set(open(KEEP_FROM).read().split("\n"))
    idx = np.array([i for i, r in enumerate(rows) if r["file"] in keep])
open(f"{OUT}/kept.txt", "w").write("\n".join(rows[i]["file"] for i in idx))
print("exterior + deduplicated:", len(idx), "of", len(rows))

y_all = np.array([CLS.index(r["class"]) for r in rows])
grp_all = np.array([(r["author"] or r["title"]).strip() for r in rows])
hold = np.array([HOLDOUT in rows[i]["author"] for i in idx])
dev, ext = idx[~hold], idx[hold]

# ---------- zero-shot ----------
prompts = [[f"a photo of a {NAMES[c]}, a car", f"a {NAMES[c]} car parked on a street in Malaysia",
            f"a photo of the {NAMES[c].split()[1]} by {NAMES[c].split()[0]}"] for c in CLS]
T = np.stack([embed_text(p).mean(0) for p in prompts]); T /= np.linalg.norm(T, axis=1, keepdims=True)


def zs(X):
    z = 100 * X @ T.T; z = np.exp(z - z.max(1, keepdims=True)); return z / z.sum(1, keepdims=True)


def fit(X, y):
    return LogisticRegression(max_iter=4000, C=5.0, class_weight="balanced").fit(X, y)


def topk(P, y, k):
    return float(np.mean([y[i] in np.argsort(-P[i])[:k] for i in range(len(y))]))


def per_class(P, y):
    return {NAMES[CLS[c]]: (round(float(np.mean(P[y == c].argmax(1) == c)), 3), int((y == c).sum()))
            for c in range(len(CLS)) if (y == c).any()}


def refusal(P, y):
    conf, ok = P.max(1), P.argmax(1) == y
    return [{"threshold": t, "coverage": round(float((conf >= t).mean()), 3),
             "accuracy_when_answering": round(float(ok[conf >= t].mean()), 3) if (conf >= t).any() else None}
            for t in [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]


# ---------- exclusions and label fixes (applied after embedding, so caches stay aligned) ----------
EXCLUDE = set(open(os.environ["EXCLUDE"]).read().split("\n")) if os.environ.get("EXCLUDE") else set()
FIXES = {}  # file -> new class slug, or "drop"
if os.environ.get("LABEL_FIXES"):
    for r in csv.DictReader(open(os.environ["LABEL_FIXES"])):
        FIXES[r["file"]] = r["action"]
y_all = y_all.copy()
for i, r in enumerate(rows):
    a = FIXES.get(r["file"])
    if a and a != "drop":
        y_all[i] = CLS.index(a)
dev_all = dev
dev_mask = np.array([rows[i]["file"] not in EXCLUDE and FIXES.get(rows[i]["file"]) != "drop" for i in dev_all])
print("excluded / dropped:", int((~dev_mask).sum()), "| label changes:", sum(1 for a in FIXES.values() if a != "drop"))

# ---------- 5-fold grouped CV ----------
dev = dev_all[dev_mask]
X, y, g = E[dev], y_all[dev], grp_all[dev]
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
P_lp = np.zeros((len(dev), len(CLS))); folds = list(cv.split(X, y, g))
deg_X = {k: np.zeros_like(X) for k in DEG}
for k, f in DEG.items():
    cachek = f"{os.environ.get('CACHE_DIR', OUT)}/emb_{k}.npy"
    if os.path.exists(cachek) and len(np.load(cachek)) == len(dev_all):
        full = np.load(cachek)
    else:
        full = embed([f(imgs[i]) for i in dev_all]); np.save(cachek, full)
    deg_X[k] = full[dev_mask]
    print("embedded", k, flush=True)
P_deg = {k: np.zeros_like(P_lp) for k in DEG}
for tr, te in folds:
    clf = fit(X[tr], y[tr])
    P_lp[te] = clf.predict_proba(X[te])
    for k in DEG:
        P_deg[k][te] = clf.predict_proba(deg_X[k][te])
P_zs = zs(X)

res = {
    "n_photos_dev": int(len(dev)), "n_groups_dev": int(len(set(g))), "n_external_test": int(len(ext)),
    "per_class_counts": {NAMES[CLS[c]]: int((y == c).sum()) for c in range(len(CLS))},
    "cv_clean": {"zero_shot_top1": topk(P_zs, y, 1), "zero_shot_top3": topk(P_zs, y, 3),
                 "probe_top1": topk(P_lp, y, 1), "probe_top3": topk(P_lp, y, 3)},
    "cv_per_class_probe": per_class(P_lp, y),
    "cv_per_class_zero_shot": per_class(P_zs, y),
    "cv_confusion_probe": confusion_matrix(y, P_lp.argmax(1), labels=range(len(CLS))).tolist(),
    "cv_degraded": {k: {"zero_shot_top1": topk(zs(deg_X[k]), y, 1), "probe_top1": topk(P_deg[k], y, 1),
                        "probe_top3": topk(P_deg[k], y, 3)} for k in DEG},
    "cv_refusal_clean": refusal(P_lp, y),
    "cv_refusal_dashcam": refusal(P_deg["dashcam_combo"], y) if DEG else None,
    "backbone": f"{ARCH}/{WEIGHTS}",
    "classes": [NAMES[c] for c in CLS],
}

# ---------- final model + independent test ----------
final = fit(X, y)
if len(ext):
    Pe = final.predict_proba(E[ext]); ye = y_all[ext]
    res["external"] = {"probe_top1": topk(Pe, ye, 1), "probe_top3": topk(Pe, ye, 3),
                       "zero_shot_top1": topk(zs(E[ext]), ye, 1), "per_class_probe": per_class(Pe, ye),
                       "refusal": refusal(Pe, ye)}
joblib.dump({"clf": final, "classes": [NAMES[c] for c in CLS], "clip": [ARCH, WEIGHTS]},
            f"{OUT}/saksi_probe.joblib")
json.dump(res, open(f"{OUT}/results.json", "w"), indent=2)
with open(f"{OUT}/oof_predictions.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "label", "predicted", "confidence", "p_label", "group"])
    for j, i in enumerate(dev):
        w.writerow([rows[i]["file"], CLS[y[j]], CLS[int(P_lp[j].argmax())], round(float(P_lp[j].max()), 3),
                    round(float(P_lp[j][y[j]]), 3), g[j]])
print(json.dumps({k: res[k] for k in ["n_photos_dev", "n_groups_dev", "cv_clean", "cv_per_class_probe"]}, indent=1))
print("degraded", json.dumps(res["cv_degraded"]))
print("external", json.dumps(res.get("external"), indent=1))
