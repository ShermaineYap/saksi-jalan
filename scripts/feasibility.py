"""Day-1 feasibility study for Saksi.

1. Filter Commons photos to car exteriors with CLIP zero-shot.
2. Split train/test BY PHOTOGRAPHER so the same car never appears on both sides.
3. Compare: (a) CLIP zero-shot, (b) linear probe on CLIP image embeddings.
4. Degradation study on the test set: blur, dark, low-res, heavy JPEG.
5. Refusal: accuracy vs coverage when the model may answer "not sure".

Usage: python3 feasibility.py DATA_DIR OUT_DIR
"""
import csv, io, json, os, sys, random
import numpy as np, torch, open_clip
from PIL import Image, ImageFilter, ImageEnhance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix

torch.set_num_threads(2)
random.seed(0); np.random.seed(0)
DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

NAMES = {"perodua_myvi": "Perodua Myvi", "perodua_axia": "Perodua Axia", "perodua_bezza": "Perodua Bezza",
         "proton_saga": "Proton Saga", "proton_x70": "Proton X70", "proton_persona": "Proton Persona",
         "proton_iriz": "Proton Iriz"}

model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tok = open_clip.get_tokenizer("ViT-B-32")
model.eval()

def embed_imgs(imgs, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            x = torch.stack([pre(im) for im in imgs[i:i + bs]])
            f = model.encode_image(x)
            out.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(out).numpy()

def embed_text(texts):
    with torch.no_grad():
        f = model.encode_text(tok(texts))
    return (f / f.norm(dim=-1, keepdim=True)).numpy()

# ---------- load ----------
rows = [r for r in csv.DictReader(open(f"{DATA}/attribution.csv")) if r["class"] in NAMES and os.path.exists(r["file"])]
imgs, keep = [], []
for r in rows:
    try:
        imgs.append(Image.open(r["file"]).convert("RGB")); keep.append(r)
    except Exception:
        pass
rows = keep
print("loaded", len(rows))
E = embed_imgs(imgs)

# ---------- 1. exterior filter ----------
scene = ["a photo of the outside of a car on a road or parking lot", "a photo of a car interior or dashboard",
         "a photo of a car engine", "a close-up photo of a car logo or badge", "a photo of a car wheel close-up",
         "a photo of a crowd at a motor show", "a photo of a car seat",
         "a photo with no car in it", "a toy model car or a drawing of a car"]
S = embed_text(scene)
is_ext = (E @ S.T).argmax(1) == 0
print("exterior kept", int(is_ext.sum()), "of", len(rows))
idx = np.where(is_ext)[0]
# drop near-duplicates (same photo hosted on both Commons and Flickr, or burst shots)
keep_idx, Ek = [], []
for i in idx:
    if Ek and float(np.max(np.stack(Ek) @ E[i])) > 0.97:
        continue
    keep_idx.append(i); Ek.append(E[i])
print("after near-duplicate removal", len(keep_idx), "of", len(idx))
idx = np.array(keep_idx)
y_names = sorted(NAMES)
y = np.array([y_names.index(rows[i]["class"]) for i in idx])
groups = np.array([rows[i]["author"] or rows[i]["title"] for i in idx])
X = E[idx]

# ---------- 2. split by photographer ----------
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
tr, te = next(gss.split(X, y, groups))
print("train", len(tr), "test", len(te), "| photographers train", len(set(groups[tr])), "test", len(set(groups[te])))

# ---------- 3a. zero-shot ----------
prompts = [[f"a photo of a {NAMES[c]}, a car", f"a {NAMES[c]} car parked on a street in Malaysia",
            f"a photo of the {NAMES[c].split()[1]} by {NAMES[c].split()[0]}"] for c in y_names]
T = np.stack([embed_text(p).mean(0) for p in prompts]); T /= np.linalg.norm(T, axis=1, keepdims=True)

def zs_probs(Xe):
    z = 100 * Xe @ T.T
    z = np.exp(z - z.max(1, keepdims=True)); return z / z.sum(1, keepdims=True)

# ---------- 3b. linear probe ----------
clf = LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced").fit(X[tr], y[tr])

def topk(P, yt, k):
    return float(np.mean([yt[i] in np.argsort(-P[i])[:k] for i in range(len(yt))]))

res = {"n_train": int(len(tr)), "n_test": int(len(te)), "n_photographers_test": int(len(set(groups[te]))), "classes": [NAMES[c] for c in y_names],
       "per_class_test": {NAMES[c]: int((y[te] == j).sum()) for j, c in enumerate(y_names)}}
Pzs, Plp = zs_probs(X[te]), clf.predict_proba(X[te])
res["clean"] = {"zero_shot_top1": topk(Pzs, y[te], 1), "zero_shot_top3": topk(Pzs, y[te], 3),
                "probe_top1": topk(Plp, y[te], 1), "probe_top3": topk(Plp, y[te], 3)}
cm = confusion_matrix(y[te], Plp.argmax(1), labels=range(len(y_names)))
res["confusion_probe"] = cm.tolist()

# ---------- 4. degradation ----------
def jpeg(im, q):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
DEG = {
    "blur_r2": lambda im: im.filter(ImageFilter.GaussianBlur(2)),
    "blur_r4": lambda im: im.filter(ImageFilter.GaussianBlur(4)),
    "dark_0.35": lambda im: ImageEnhance.Brightness(im).enhance(0.35),
    "lowres_64px": lambda im: im.resize((64, int(64 * im.height / im.width))).resize(im.size),
    "jpeg_q10": lambda im: jpeg(im, 10),
    "dashcam_combo": lambda im: jpeg(ImageEnhance.Brightness(
        im.resize((96, int(96 * im.height / im.width))).resize(im.size).filter(ImageFilter.GaussianBlur(1.5))).enhance(0.5), 20),
}
test_imgs = [imgs[idx[i]] for i in te]
res["degraded"] = {}
Pdeg = {}
for name, f in DEG.items():
    Xd = embed_imgs([f(im) for im in test_imgs])
    Pz, Pl = zs_probs(Xd), clf.predict_proba(Xd)
    Pdeg[name] = Pl
    res["degraded"][name] = {"zero_shot_top1": topk(Pz, y[te], 1), "probe_top1": topk(Pl, y[te], 1),
                             "probe_top3": topk(Pl, y[te], 3)}
    print(name, res["degraded"][name], flush=True)

# ---------- 5. refusal ----------
def coverage_curve(P, yt):
    conf, ok = P.max(1), P.argmax(1) == yt
    out = []
    for t in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        m = conf >= t
        out.append({"threshold": t, "coverage": float(m.mean()), "accuracy_when_answering": float(ok[m].mean()) if m.any() else None})
    return out
res["refusal_clean"] = coverage_curve(Plp, y[te])
res["refusal_dashcam_combo"] = coverage_curve(Pdeg["dashcam_combo"], y[te])

json.dump(res, open(f"{OUT}/results.json", "w"), indent=2)
print(json.dumps({k: res[k] for k in ["n_train", "n_test", "clean"]}, indent=2))

# save model for the app
import joblib
joblib.dump({"clf": clf, "classes": [NAMES[c] for c in y_names], "clip": "ViT-B-32/laion2b_s34b_b79k"}, f"{OUT}/probe.joblib")
