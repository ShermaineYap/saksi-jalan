# Saksi Jalan

*Saksi jalan* means "road witness" in Malay.

Saksi Jalan turns a dashcam frame or phone photo into a police-ready vehicle description (make, model, colour) in Bahasa Malaysia and English. When the picture is too poor to be sure, it says so and shows its top candidates instead of guessing.

Entry for the **STEM2Connect AI Hackathon 2026**, Community Impact track.

> Status: work in progress (Day 1 of the build, 16 Nov 2026). Sections marked TODO will be filled in as the project is built and measured.

## The problem

Malaysia recorded about 6,500 road deaths in 2025, around 18 a day, and about two-thirds of them were motorcyclists and pillion riders ([The Star, 27 Jan 2026](https://www.thestar.com.my/news/nation/2026/01/27/motorcyclists-make-up-two-thirds-of-road-deaths-in-2025-says-loke)). When a car drives off after a crash, witnesses often cannot name it. In an April 2026 hit-and-run in Ampang, police could only describe a car of unidentified make and registration ([Malay Mail](https://www.malaymail.com/amp/news/malaysia/2026/04/15/ampang-cops-hunt-driver-in-taman-muda-hit-and-run-that-left-pedestrian-seriously-injured/216349)).

Global car-recognition datasets don't include Malaysia's most common cars (Perodua Myvi, Axia, Bezza, Proton Saga and others), so off-the-shelf tools can't help.

## How it uses AI

TODO: pipeline diagram and description.

1. Find the vehicle in the photo.
2. Recognise the model with a classifier trained for Malaysian cars.
3. Estimate the colour.
4. Decide: answer, or say "not sure" when confidence is below a threshold chosen on validation data.
5. Fill a bilingual witness statement.

## Results (measured, not claimed)

TODO: accuracy on a held-out test set split by photographer, degradation study (blur, low light, low resolution, heavy compression), and accuracy versus coverage when the model is allowed to refuse.

## What it will not do

- No number-plate lookup and no owner identification. The user types any plate characters they can read.
- Photos are processed and discarded, not stored.
- The output is an aid for a police report, not evidence.

## How to run

TODO

## Data and credits

- Training and test photos: openly licensed images from [Wikimedia Commons](https://commons.wikimedia.org/) (CC0, public domain, CC BY, CC BY-SA). Per-image author, licence and source are listed in `data/attribution.csv`.
- [OpenCLIP](https://github.com/mlfoundations/open_clip) (MIT).
- TODO: other libraries and datasets.

## AI tools disclosure

TODO: Claude (Anthropic) is used as a coding and research assistant. This section will list what it contributed and what I wrote, changed or tested myself.

## Licence

AGPL-3.0. See `LICENSE`.
