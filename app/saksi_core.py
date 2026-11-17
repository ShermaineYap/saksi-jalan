"""Core logic for Saksi Jalan: find cars, name the model, name the colour, decide, write the statement.

Kept free of any UI code so it can be unit-tested.
"""
from __future__ import annotations

import colorsys
import os
from dataclasses import dataclass, field
from datetime import datetime

import joblib
import numpy as np
import open_clip
import torch
from PIL import Image
from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                                          fasterrcnn_mobilenet_v3_large_fpn)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("SAKSI_MODEL", os.path.join(HERE, "saksi_probe.joblib"))

# Answer only when the classifier is at least this sure. Chosen from the cross-validated
# refusal curve (see docs/results.md): at 0.70 the model answers about half of photos
# and is right about 87% of the time when it does.
CONFIDENCE_THRESHOLD = float(os.environ.get("SAKSI_THRESHOLD", "0.70"))
COCO_CAR_LABELS = {3: "car", 8: "truck"}  # COCO ids used by torchvision

# Colour names: (English, Bahasa Malaysia)
COLOURS = {
    "white": ("white", "putih"), "black": ("black", "hitam"), "grey": ("grey", "kelabu"),
    "silver": ("silver", "perak"), "red": ("red", "merah"), "orange": ("orange", "jingga"),
    "yellow": ("yellow", "kuning"), "green": ("green", "hijau"), "blue": ("blue", "biru"),
    "brown": ("brown", "coklat"), "purple": ("purple", "ungu"),
}


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    score: float


@dataclass
class Verdict:
    top3: list[tuple[str, float]]
    confident: bool
    colour_en: str
    colour_ms: str
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> str:
        return self.top3[0][0]


# ---------------------------------------------------------------- colour

def name_colour(rgb: tuple[float, float, float]) -> str:
    """Map an average RGB (0-255) to a plain colour name."""
    r, g, b = (c / 255 for c in rgb)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if l > 0.84 or (l > 0.78 and s < 0.18):
        return "white"
    if l < 0.16:
        return "black"
    if s < 0.18:
        return "silver" if l > 0.50 else "grey"
    deg = h * 360
    if deg < 15 or deg >= 340:
        return "brown" if l < 0.30 else "red"
    if deg < 40:
        return "brown" if l < 0.40 else "orange"
    if deg < 70:
        return "yellow"
    if deg < 170:
        return "green"
    if deg < 255:
        return "blue"
    return "purple" if deg < 300 else "red"


def dominant_colour(crop: Image.Image) -> str:
    """Body colour: the most common colour among the central body pixels.

    Uses the middle band of the crop (skips windows at the top and tyres/road at the bottom),
    groups pixels into colour names and returns the most frequent name.
    """
    im = crop.convert("RGB").resize((96, 96))
    a = np.asarray(im)[38:78, 12:84].reshape(-1, 3)
    names = [name_colour(tuple(p)) for p in a[:: 3]]
    vals, counts = np.unique(names, return_counts=True)
    return str(vals[np.argmax(counts)])


# ---------------------------------------------------------------- decision

def decide(probs: dict[str, float], threshold: float = CONFIDENCE_THRESHOLD) -> tuple[list[tuple[str, float]], bool]:
    """Return the top-3 (name, probability) and whether the model is sure enough to name one."""
    top3 = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
    return top3, top3[0][1] >= threshold


# ---------------------------------------------------------------- statement

def witness_statement(verdict: Verdict | None, when: str = "", where: str = "", plate: str = "",
                      direction: str = "", extra: str = "") -> tuple[str, str]:
    """Bilingual witness statement (English, Bahasa Malaysia). Nothing is invented: blanks stay blank."""
    when = when.strip() or "[time not given]"
    where = where.strip() or "[place not given]"
    plate = plate.strip().upper()
    if verdict is None:
        car_en, car_ms = "a vehicle I could not identify", "sebuah kenderaan yang tidak dapat saya kenal pasti"
    elif verdict.confident:
        car_en = f"a {verdict.colour_en} {verdict.best}"
        car_ms = f"sebuah {verdict.best} berwarna {verdict.colour_ms}"
    else:
        alts = ", ".join(n for n, _ in verdict.top3)
        car_en = f"a {verdict.colour_en} car, possibly one of: {alts}"
        car_ms = f"sebuah kereta berwarna {verdict.colour_ms}, mungkin salah satu daripada: {alts}"
    en = [f"On {when}, at {where}, I saw {car_en}."]
    ms = [f"Pada {when}, di {where}, saya telah melihat {car_ms}."]
    if plate:
        en.append(f"The registration number I could read was: {plate}.")
        ms.append(f"Nombor pendaftaran yang dapat saya baca ialah: {plate}.")
    else:
        en.append("I could not read the registration number.")
        ms.append("Saya tidak dapat membaca nombor pendaftaran.")
    if direction.strip():
        en.append(f"The vehicle left heading {direction.strip()}.")
        ms.append(f"Kenderaan itu beredar menuju ke arah {direction.strip()}.")
    if extra.strip():
        en.append(extra.strip())
        ms.append(extra.strip())
    note_en = ("The vehicle description was suggested by Saksi Jalan, an AI tool, from a photo. "
               "It is an aid to my statement, not proof.")
    note_ms = ("Keterangan kenderaan ini dicadangkan oleh Saksi Jalan, sebuah alat AI, berdasarkan gambar. "
               "Ia hanya membantu keterangan saya, bukan bukti.")
    return " ".join(en) + "\n\n" + note_en, " ".join(ms) + "\n\n" + note_ms


# ---------------------------------------------------------------- models

class Saksi:
    def __init__(self, model_path: str = MODEL_PATH):
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        bundle = joblib.load(model_path)
        self.clf, self.classes = bundle["clf"], bundle["classes"]
        arch, weights = bundle["clip"]
        self.clip, _, self.pre = open_clip.create_model_and_transforms(arch, pretrained=weights)
        self.clip.eval()
        w = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        self.det = fasterrcnn_mobilenet_v3_large_fpn(weights=w, box_score_thresh=0.5).eval()
        self.det_tf = w.transforms()

    @torch.no_grad()
    def find_cars(self, img: Image.Image, min_frac: float = 0.01) -> list[Detection]:
        """Cars in the photo, largest first. Ignores tiny boxes (< min_frac of the image)."""
        img = img.convert("RGB")
        out = self.det([self.det_tf(img)])[0]
        W, H = img.size
        dets = []
        for box, lab, sc in zip(out["boxes"], out["labels"], out["scores"]):
            if int(lab) not in COCO_CAR_LABELS:
                continue
            x0, y0, x1, y1 = (int(v) for v in box.tolist())
            if (x1 - x0) * (y1 - y0) < min_frac * W * H:
                continue
            dets.append(Detection((x0, y0, x1, y1), float(sc)))
        return sorted(dets, key=lambda d: -(d.box[2] - d.box[0]) * (d.box[3] - d.box[1]))

    @staticmethod
    def crop(img: Image.Image, det: Detection | None, pad: float = 0.05) -> Image.Image:
        if det is None:
            return img.convert("RGB")
        x0, y0, x1, y1 = det.box
        dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
        W, H = img.size
        return img.convert("RGB").crop((max(0, x0 - dx), max(0, y0 - dy), min(W, x1 + dx), min(H, y1 + dy)))

    @torch.no_grad()
    def classify(self, crop: Image.Image) -> dict[str, float]:
        f = self.clip.encode_image(self.pre(crop).unsqueeze(0))
        f = (f / f.norm(dim=-1, keepdim=True)).numpy()
        p = self.clf.predict_proba(f)[0]
        return {c: float(v) for c, v in zip(self.classes, p)}

    def analyse(self, crop: Image.Image) -> Verdict:
        top3, ok = decide(self.classify(crop))
        colour = dominant_colour(crop)
        en, ms = COLOURS[colour]
        notes = []
        if crop.width < 120 or crop.height < 80:
            notes.append("The car is very small in this photo, so the answer is less reliable.")
        return Verdict(top3=top3, confident=ok, colour_en=en, colour_ms=ms, notes=notes)


def now_text() -> str:
    return datetime.now().strftime("%d %B %Y, %I:%M %p")
