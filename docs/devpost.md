# Saksi Jalan: Devpost submission draft

**Track:** Community Impact
**Tagline:** Turns a dashcam frame or phone photo into a police-ready vehicle description in BM and English, and says "not sure" instead of guessing.
**Links:** live app https://huggingface.co/spaces/Shermaine526/saksi-jalan · code https://github.com/ShermaineYap/saksi-jalan · video [VIDEO LINK]

---

## Inspiration (Problem and Impact)

Malaysia recorded about 6,500 road deaths in 2025, around 18 a day, and about two-thirds of the dead were motorcyclists or pillion riders. When a car drives off after a crash, the people left behind often can't say what it was. In an April 2026 hit-and-run in Ampang, police could only describe "a car of unidentified make and registration".

Most Malaysians can't tell a Myvi from an Axia from a Bezza in a blurry frame, and global car-recognition tools don't know these cars at all: Malaysia's best-sellers aren't in the standard datasets. Saksi Jalan ("road witness") is for the witness, rider or family member with a dashcam clip or phone photo who needs to file a police report.

## What it does

1. **Upload** a photo or dashcam frame (or try a sample).
2. **Find the car.** It detects every car and shows the largest; you can pick another. If there's no car, it says so and doesn't guess.
3. **Name the model and colour.** One of six common Malaysian cars (Perodua Axia, Bezza, Myvi; Proton Iriz, Persona, Saga), with confidence.
4. **Say "not sure" when it should.** Below 70% confidence it won't name a model; it shows the top 3 instead.
5. **Write the statement.** Add the time, place and any plate characters you could read, and it drafts a witness statement in **Bahasa Malaysia and English**. Blanks stay blank: nothing is invented.

Privacy by design: everything runs on your own device and the photo is never uploaded; it doesn't read plates or look up owners; the statement says it's an aid, not proof.

## How it uses AI

- **Detection:** a pretrained DETR detector (COCO) finds the cars.
- **Recognition:** OpenCLIP ViT-B/16 (DFN-2B weights) turns each car into image features, and a classifier **I trained** on 1,301 photos of Malaysian cars picks the model.
- **Calibrated refusal:** the 70% threshold was chosen from a measured accuracy-versus-coverage curve, so "not sure" is a measured decision, not a guess.
- **Data cleaning with AI:** CLIP zero-shot checks flagged buses, close-ups and aerial shots in the training data, and I reviewed every flag by eye.

**AI coding assistant disclosure:** I used Claude (Anthropic) as a coding and research assistant: researching the problem, writing the data-collection, study and app code, and running experiments. I chose the problem and scope, made the trade-offs (dropping Proton X70, the 70% threshold, not republishing third-party images), reviewed flagged photos and results, and tested the app. Every number here comes from running `scripts/study.py`.

## How I built it (Functionality)

- **Data:** about 2,400 openly licensed photos from Wikimedia Commons, Flickr (via Openverse) and CC BY 4.0 Roboflow Universe datasets, with author and licence recorded for every photo.
- **Honest testing:** 5-fold cross-validation **grouped by photographer or dataset**, so the model is always tested on photographers it never saw, plus a 30-photo held-out dataset never used in training.
- **App:** a static web page that runs both models **in your browser** (onnxruntime-web), so photos never leave the phone. The image model is compressed from 345 MB to 89 MB with weight-only int8, keeping it nearly identical to the original (cosine similarity 0.9996). I first tried full int8 quantisation, which cost 6 points of accuracy, so I measured and switched. A Python/Gradio version of the same pipeline has unit tests for the colour, decision and statement logic.

### Results

| | Top-1 | Top-3 |
|---|---|---|
| CLIP zero-shot (no training) | 52% | 95% |
| **Saksi Jalan, cross-validated** | **74%** | **95%** |
| **Saksi Jalan, 30 held-out photos** | **100%** | 100% |

At the 70% threshold it answers 56% of photos and is right on 94% of them; on dashcam-quality photos (blurred, dark, low-res, compressed) it answers 37% and is right on 92%.

## Challenges and what I learned (Learning and Growth)

- **Data, not the model, was the hard part.** On Day 1 one photographer had taken most of my Axia photos, mostly at motor shows. A model trained on that can learn the show floor, not the car.
- **A random split would have lied to me.** Splitting by photographer showed the real accuracy; my first split left only 2–4 test photos for some cars, so I moved to grouped 5-fold cross-validation.
- **Look at your data.** My "Iriz" class contained 35 Irizar coaches, 9 oil slicks and 3 private jets, because "iriz" matches other things on Flickr. Cleaning 115 bad photos raised Iriz from 38% to 89%.
- **Measure choices.** I compared three image backbones on the same photos and chose by result (65% → 70%). I also caught a mistake of my own: I had loaded one model with the wrong activation function, so the first comparison was invalid.
- **Know the limits.** Open photos of the 2018+ Myvi are scarce, so cross-validated Myvi accuracy is only 43%, even though the held-out modern Myvis were all correct. Saga photos skew to older models. I say this openly rather than hide it.

**What I'd change:** collect a small, consented set of street photos of current models, add an "other car" class, and fine-tune the image model on a GPU instead of training only the final layer.

## What's next

- Add an "other car" class and more models (Proton X50 and X70, Perodua Alza and Ativa, Toyota Vios, Honda City).
- Motorcycles, the vehicle type in two-thirds of road deaths.
- A WhatsApp or Telegram bot, since that's where Malaysians already share dashcam clips.
- Talk to PDRM traffic investigators and the Malaysian Institute of Road Safety Research (MIROS) about what a useful witness statement needs.

## Built with

python · pytorch · open-clip · scikit-learn · onnx · onnxruntime-web · javascript · gradio · hugging-face-spaces · wikimedia-commons · openverse · roboflow
