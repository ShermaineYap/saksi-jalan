# Measured results (17 Nov 2026)

Every number on this page comes from `scripts/study.py`. Raw output: `results/final/results.json`; every photo's out-of-fold prediction: `results/final/oof_predictions.csv`.

## Setup

- **Classes (6):** Perodua Axia, Bezza, Myvi; Proton Iriz, Persona, Saga. Proton X70 was dropped: only 17 usable exterior photos, nearly all from one photographer.
- **Data:** 1,301 cleaned photos from 61 photographers or datasets: Wikimedia Commons, Flickr (via Openverse) and CC BY 4.0 Roboflow Universe datasets. Licence and author for every photo are in `data/attribution.csv`.
- **Model:** frozen OpenCLIP ViT-B/16 (DFN-2B weights) image features plus a logistic-regression head (a "linear probe").
- **Test design:** 5-fold cross-validation grouped by photographer or dataset, so every photo is scored by a model that never saw its photographer or dataset. Separately, 30 photos from a dataset never used in training (`axia-detection/car-model-validation`) are a held-out test.

## Headline numbers

| | Top-1 | Top-3 |
|---|---|---|
| Chance (6 classes) | 17% | 50% |
| CLIP zero-shot (no training) | 52% | 95% |
| **Saksi Jalan, cross-validated** | **74%** | **95%** |
| **Saksi Jalan, held-out 30 photos** | **100%** | 100% |

Per class (cross-validated top-1): Saga 93%, Axia 92%, Persona 89%, Iriz 89%, Bezza 62%, Myvi 43%.

### Saying "not sure"

The app names a model only when it is at least 70% confident; otherwise it shows its top 3.

| Confidence needed | Clean photos: answered | Accuracy when answering | Dashcam-like photos: answered | Accuracy when answering |
|---|---|---|---|---|
| none | 100% | 74% | 100% | 64% |
| 0.6 | 67% | 89% | 54% | 83% |
| **0.7 (used)** | **56%** | **94%** | **37%** | **92%** |
| 0.8 | 44% | 97% | 22% | 97% |

On worse photos the model answers less often, while its accuracy when it does answer stays above 90%.

### Bad photos (top-1, same cross-validation)

| Condition | Saksi Jalan | Zero-shot |
|---|---|---|
| Clean | 74% | 52% |
| Dark (35% brightness) | 76% | 52% |
| Blur (radius 4) | 67% | 44% |
| Low resolution (64 px wide) | 69% | 39% |
| Dashcam-like (low-res + blur + dark + heavy JPEG) | 64% | 45% |

## What went wrong, and what I changed

1. **One photographer dominated the data.** On Day 1, one Commons photographer had taken most Axia photos, mostly at motor shows. *Fix:* added Flickr and Roboflow sources, and split train and test by photographer, not at random.
2. **My first test split was lopsided.** A single photographer-based split left only 2–4 Axia, Bezza and X70 photos in the test set. *Fix:* grouped 5-fold cross-validation, so every photo is tested once.
3. **The filter threw away good data.** CLIP called white-background listing photos "not a car". *Fix:* human-labelled Roboflow boxes skip the filter, after I checked a sample of rejected photos by eye.
4. **"Iriz" was full of buses.** Flickr searches for "iriz" returned 35 Irizar coaches, 9 oil-slick photos and 3 private jets. *Fix:* an automatic check for buses, close-ups and aerial shots, reviewed by eye, plus a manual review of the photos the model most strongly disagreed with. 115 photos dropped (`data/label_fixes.csv`). Iriz rose from 38% to 89%, and overall from 70% to 74%.
5. **I chose the backbone by measurement.** On the same photos: ViT-B/32 65%, ViT-B/16 DataComp 67%, **ViT-B/16 DFN-2B 70%** (before cleaning). An early comparison was invalid because I loaded the DFN weights with the wrong activation function; I caught it from a warning and reran.

## Honest limitations

- **Myvi generations.** Openly licensed Myvi photos are mostly the 2011–2017 model; nearly all modern (2018+) Myvi photos come from one dataset. When cross-validation holds that dataset out, modern Myvis get called Bezza or Axia, hence 43%. On the held-out set, which has modern Myvis while the training set does too, all 10 were right.
- **Saga skews old.** Most Saga photos are 1985–2015 models, not the current Saga.
- **Closed set.** It only knows six models. A Honda City or Toyota Vios will be forced into one of them, though usually with low confidence, so the app says "not sure".
- **Roboflow image provenance.** Those datasets are published as CC BY 4.0, but some photos look like car-listing photos. I credit each dataset and don't republish its images; the app's sample photos come only from Commons and Flickr.
- **Small held-out set.** 30 photos, 3 classes. 100% on 30 photos is encouraging, not proof.
- **Not evidence.** The output is an aid for a witness statement, not identification.
