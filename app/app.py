"""Saksi Jalan: Gradio app.

Photo in, police-ready vehicle description out (Bahasa Malaysia and English).
Photos are processed in memory and never stored.
"""
import os

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from saksi_core import CONFIDENCE_THRESHOLD, Saksi, now_text, witness_statement

HERE = os.path.dirname(os.path.abspath(__file__))
S = Saksi()

INTRO = """
# 🚗 Saksi Jalan: road witness
Upload a dashcam frame or a photo from your phone. Saksi Jalan finds the car, suggests its **model and colour**,
and drafts a **witness statement in Bahasa Malaysia and English** that you can bring to the police.

It recognises six of Malaysia's most common cars: **Perodua Axia, Bezza, Myvi** and **Proton Iriz, Persona, Saga**.
When it isn't sure, it **says so** and shows its top 3 guesses instead of pretending.

🔒 Your photo is processed in memory and **never stored**. Saksi Jalan does not read number plates or look up owners.
"""


def draw_boxes(img, dets, chosen):
    im = img.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w = max(3, im.width // 200)
    try:
        font = ImageFont.load_default(size=max(16, im.width // 30))
    except TypeError:
        font = ImageFont.load_default()
    for i, det in enumerate(dets):
        col = "#ffcc00" if i == chosen else "#00b3ff"
        d.rectangle(det.box, outline=col, width=w if i != chosen else w * 2)
        d.text((det.box[0] + w * 2, det.box[1] + w * 2), f"{i + 1}", fill=col, font=font)
    return im


def result_markdown(v):
    rows = "\n".join(f"| {n} | {'█' * max(1, round(p * 20))} {p:.0%} |" for n, p in v.top3)
    if v.confident:
        head = f"### ✅ Most likely: **{v.colour_en} {v.best}**\n*{v.colour_ms.capitalize()}, {v.best}*"
    else:
        head = (f"### 🤔 Not sure\nThe model is less than {CONFIDENCE_THRESHOLD:.0%} confident, so it won't name one model. "
                f"Colour looks **{v.colour_en}** (*{v.colour_ms}*). Top 3 candidates:")
    notes = "".join(f"\n\n⚠️ {n}" for n in v.notes)
    return f"{head}\n\n| Model | Confidence |\n|---|---|\n{rows}{notes}"


def analyse(img, which):
    if img is None:
        return None, None, "Upload a photo first.", gr.update(choices=[], value=None), None
    dets = S.find_cars(img)
    if not dets:
        msg = ("### 🚫 No car found\nI couldn't find a car in this photo, so I won't guess. "
               "Try a photo or dashcam frame where the car is more visible.")
        return img, None, msg, gr.update(choices=[], value=None), None
    labels = [f"Car {i + 1}" for i in range(len(dets))]
    i = labels.index(which) if which in labels else 0
    crop = S.crop(img, dets[i])
    v = S.analyse(crop)
    if len(dets) > 1:
        v.notes.append(f"{len(dets)} cars found. Showing car {i + 1} (largest first); pick another below.")
    return draw_boxes(img, dets, i), crop, result_markdown(v), gr.update(choices=labels, value=labels[i]), v


def statement(v, when, where, plate, direction, extra):
    return witness_statement(v, when, where, plate, direction, extra)


examples_dir = os.path.join(HERE, "examples")
examples = sorted(os.path.join(examples_dir, f) for f in os.listdir(examples_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))) if os.path.isdir(examples_dir) else []

THEME = gr.themes.Soft(primary_hue="amber")

with gr.Blocks(title="Saksi Jalan") as demo:
    verdict = gr.State()
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column():
            photo = gr.Image(type="pil", label="Photo or dashcam frame", sources=["upload", "clipboard", "webcam"])
            which = gr.Dropdown(label="Which car?", choices=[], interactive=True)
            go = gr.Button("Identify the car", variant="primary")
            if examples:
                gr.Examples(examples, inputs=photo, label="Try a sample photo (openly licensed, see credits)")
        with gr.Column():
            boxed = gr.Image(label="Cars found", interactive=False)
            crop = gr.Image(label="Car being described", interactive=False, height=180)
            result = gr.Markdown()
    gr.Markdown("## Witness statement\nFill in what you know. Anything left blank stays blank: nothing is made up.")
    with gr.Row():
        when = gr.Textbox(label="When / Bila", value=now_text())
        where = gr.Textbox(label="Where / Di mana", placeholder="e.g. Jalan Ampang, near the Taman Muda crossing")
    with gr.Row():
        plate = gr.Textbox(label="Plate characters you could read / Nombor plat", placeholder="e.g. WA 1")
        direction = gr.Textbox(label="Direction it went / Arah", placeholder="e.g. towards Cheras")
    extra = gr.Textbox(label="Anything else / Lain-lain", placeholder="e.g. dent on the rear bumper")
    write = gr.Button("Write statement", variant="primary")
    with gr.Row():
        en = gr.Textbox(label="English", lines=6, buttons=["copy"])
        ms = gr.Textbox(label="Bahasa Malaysia", lines=6, buttons=["copy"])
    gr.Markdown("Saksi Jalan is an aid for your police report, not evidence. "
                "Built for the STEM2Connect AI Hackathon 2026. [Source code and measured results]"
                "(https://github.com/ShermaineYap/saksi-jalan)")

    outs = [boxed, crop, result, which, verdict]
    go.click(analyse, [photo, which], outs)
    photo.upload(lambda img: analyse(img, None), photo, outs)
    which.input(analyse, [photo, which], outs)
    write.click(statement, [verdict, when, where, plate, direction, extra], [en, ms])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), theme=THEME)
