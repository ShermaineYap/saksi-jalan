# Saksi Jalan

*Saksi jalan* means "road witness" in Malay.

Saksi Jalan turns a dashcam frame or phone photo into a police-ready vehicle description (make, model, colour) in Bahasa Malaysia and English. When the picture is too poor to be sure, it says so and shows its top candidates instead of guessing.

Entry for the **STEM2Connect AI Hackathon 2026**, Community Impact track.

**▶ Try it: https://huggingface.co/spaces/Shermaine526/saksi-jalan** (runs entirely in your browser; your photo is never uploaded)

## The problem

Malaysia recorded about 6,500 road deaths in 2025, around 18 a day, and about two-thirds of them were motorcyclists and pillion riders ([The Star, 27 Jan 2026](https://www.thestar.com.my/news/nation/2026/01/27/motorcyclists-make-up-two-thirds-of-road-deaths-in-2025-says-loke)). When a car drives off after a crash, witnesses often cannot name it. In an April 2026 hit-and-run in Ampang, police could only describe a car of unidentified make and registration ([Malay Mail](https://www.malaymail.com/amp/news/malaysia/2026/04/15/ampang-cops-hunt-driver-in-taman-muda-hit-and-run-that-left-pedestrian-seriously-injured/216349)).

Global car-recognition datasets don't include Malaysia's most common cars (Perodua Myvi, Axia, Bezza, Proton Saga and others), so off-the-shelf tools can't help.

## How it uses AI

```
photo ──► find cars ──► crop chosen car ──► name the model ──► decide ──► witness statement (BM + EN)
          (Faster R-CNN,                      (OpenCLIP ViT-B/16      answer if ≥70% sure,
           COCO)                               + my trained head)     else "not sure" + top 3
                                └──► body colour (Malay + English names)
```

1. **Find the car.** A pretrained object detector (torchvision Faster R-CNN MobileNet-V3, COCO) finds every car; you can pick which one. No car found means no guess.
2. **Name the model.** An image model (OpenCLIP ViT-B/16, DFN-2B weights) turns the crop into features, and a classifier I trained on 1,301 photos of Malaysian cars picks among six models.
3. **Decide or refuse.** It names a model only when it is at least 70% confident. That threshold comes from the measured refusal curve below.
4. **Colour.** The most common colour in the car body, named in English and Bahasa Malaysia.
5. **Statement.** Fills a bilingual statement from what you type (time, place, plate characters you could read). Blanks stay blank: nothing is invented.

## Results (measured, not claimed)

Full details, method and limitations: [`docs/results.md`](docs/results.md).

| | Top-1 | Top-3 |
|---|---|---|
| CLIP zero-shot (no training) | 52% | 95% |
| **Saksi Jalan, 5-fold CV grouped by photographer/dataset** | **74%** | **95%** |
| **Saksi Jalan, 30 held-out photos** | **100%** | 100% |

- **When it answers, it's usually right:** at the 70% confidence threshold it answers 56% of photos and is right on 94% of them. On dashcam-like photos it answers 37% and is right on 92%.
- **Weak spot:** modern Perodua Myvi (43% in cross-validation), because open photos of the 2018+ Myvi are scarce. See limitations.
- **How I got here:** Day 1 started at 84% on three classes with a lopsided test set; cleaning the data (115 bad photos, including 35 Irizar buses labelled "Iriz") was the biggest single gain. See [`docs/day1-feasibility.md`](docs/day1-feasibility.md) and [`docs/results.md`](docs/results.md).

## What it will not do

- No number-plate lookup and no owner identification. The user types any plate characters they can read.
- Photos are processed and discarded, not stored.
- The output is an aid for a police report, not evidence.

## How to run

**Live:** https://huggingface.co/spaces/Shermaine526/saksi-jalan. The `web/` folder is a static page that runs both models in the browser with [onnxruntime-web](https://onnxruntime.ai). The first visit downloads about 130 MB of models, which the browser then caches. Rebuild the model files with `scripts/export_web_models.py`.

Weight-only int8 compression keeps the browser model almost identical to the Python one (cosine similarity 0.9996); full int8 quantisation would have cost about 6 points of accuracy, so I didn't use it. `scripts/test_web.py` runs the page in headless Chrome on all samples.

**Python reference app (Gradio):**

```bash
cd app
pip install -r requirements.txt gradio
python app.py            # opens on http://localhost:7860
python -m pytest -q      # unit tests
```

To reproduce the study, collect the data with `scripts/collect_commons.py`, `scripts/collect_openverse.py` and `scripts/collect_roboflow.py` (needs a free Roboflow API key in `ROBOFLOW_API_KEY`), then run `scripts/study.py`.

## Data and credits

- **Photos:** [Wikimedia Commons](https://commons.wikimedia.org/) and Flickr via [Openverse](https://openverse.org) (CC0, public domain, CC BY, CC BY-SA), plus these CC BY 4.0 [Roboflow Universe](https://universe.roboflow.com) datasets: `car-model-gbo55/perodua-axia`, `axia-detection/perodua-axia-detection`, `swinburne-crrcc/bezza`, `trainlabelstxt/trainingmyvi`, `axia-detection/car-model-validation`. Author, licence and source for every photo: `data/attribution.csv`. Images aren't redistributed in this repo, except the app's sample photos (credits in `app/examples/CREDITS.md`).
- **Models:** [OpenCLIP](https://github.com/mlfoundations/open_clip) (MIT) with Apple's DFN-2B ViT-B/16 weights; [torchvision](https://github.com/pytorch/vision) Faster R-CNN (BSD-3) in the Python app; DETR ResNet-50 (Facebook AI, Apache-2.0; ONNX export by [Xenova](https://huggingface.co/Xenova/detr-resnet-50)) in the browser app.
- **Browser runtime:** [onnxruntime-web](https://github.com/microsoft/onnxruntime) (MIT).
- **Libraries:** PyTorch, scikit-learn, Gradio, Pillow, NumPy.
- **Road-safety figures:** The Star and Malay Mail, linked above.

## AI tools disclosure

I used **Claude (Anthropic)** as a coding and research assistant throughout. It helped research the hackathon and the problem, write the data-collection, study and app code, and run the experiments. I chose the problem and the scope, decided which trade-offs to accept (dropping X70, the 70% threshold, not republishing Roboflow images), reviewed the flagged photos and the results, and tested the app. Every number in this README comes from running `scripts/study.py`, not from the assistant.

## Licence

AGPL-3.0. See `LICENSE`.
