# Day 1 feasibility study (16 Nov 2026)

Question: can a model trained on openly licensed photos tell Malaysia's look-alike Perodua cars apart, and does it know when it is unsure?

## Setup

- **Data:** 445 photos of Perodua Myvi, Axia and Bezza from Wikimedia Commons (CC BY-SA, CC BY, CC0), collected with `scripts/collect_commons.py`. Author, licence and source for each photo are recorded.
- **Exterior filter:** CLIP zero-shot keeps only photos that show the outside of a whole car. It kept 201 of 445. A by-eye check of 48 random rejects found 47 correct rejections (interiors, badges, door handles, engines) and 1 wrong one.
- **Split by photographer:** train and test never share a photographer, so the same car can't appear on both sides. That gives 151 train and 50 test photos, from 15 and 7 photographers.
- **Models:** (a) CLIP ViT-B/32 zero-shot with text prompts; (b) logistic regression ("linear probe") on CLIP image embeddings.
- Runs on a 2-core CPU in 46 seconds.

## Results (50 test photos; roughly ±10 percentage points at 95% confidence)

| Model | Top-1 | Top-3 |
|---|---|---|
| Chance (3 classes) | 33% | 100% |
| CLIP zero-shot | 50% | 90% |
| **Linear probe (trained)** | **84%** | 100% |

Confusion (probe): Bezza, a sedan, is almost always right (21/22). The two hatchbacks get mixed up: 3 of 10 Axia were called Myvi, and 4 of 18 Myvi were called Axia.

### Degradation (probe top-1; zero-shot in brackets)

| Condition | Probe | Zero-shot |
|---|---|---|
| Clean | 84% | 50% |
| Blur radius 2 | 86% | 60% |
| Blur radius 4 | 82% | 44% |
| Dark (35% brightness) | 84% | 36% |
| Low-res (64 px wide) | 82% | 54% |
| JPEG quality 10 | 84% | 66% |
| Dashcam-like combo | 88% | 36% |

The probe barely drops. Zero-shot falls to 36% in the dark and dashcam-like conditions. The small rises above clean (86%, 88%) are noise from a 50-photo test set, not real gains.

### Refusing when unsure (probe, clean)

| Answer only if confidence ≥ | Answers given | Accuracy when answering |
|---|---|---|
| 0 (always answer) | 100% | 84% |
| 0.6 | 80% | 90% |
| 0.8 | 40% | 95% |

## What I learned

1. **The data is the bottleneck, not the model.** One photographer took 147 of 160 Axia photos, 97 of 160 Myvi and 77 of 125 Bezza, mostly at motor shows. A model trained on that may learn the show floor, not the car. Splitting by photographer exposes this, but it leaves a small test set.
2. **Training beats zero-shot clearly.** General CLIP doesn't really know Malaysian cars (50%); a small trained head on the same features reaches 84%.
3. **Refusal is worth building.** Giving up 20% of answers raises accuracy from 84% to 90%.

## Next

- Photograph real cars in car parks and on streets (plates blurred) to add photographers, angles and lighting, and to build a real-world test set.
- Add Proton Saga, X70, Persona and Iriz (download in progress, slowed by Commons rate limits).
- Repeat the study with more test photos to tighten the numbers.
