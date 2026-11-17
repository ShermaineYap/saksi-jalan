---
title: Saksi Jalan
emoji: 🚗
colorFrom: blue
colorTo: yellow
sdk: static
pinned: false
license: agpl-3.0
short_description: Photo to police-ready car description, in BM and English
---

# Saksi Jalan (road witness)

Upload a dashcam frame or phone photo. Saksi Jalan finds the car, suggests its model and colour, and drafts a
witness statement in Bahasa Malaysia and English. When it isn't sure, it says so.

Everything runs in your browser with onnxruntime-web: the photo never leaves your device.

- Car detector: DETR ResNet-50 (Facebook AI, Apache-2.0), ONNX export by Xenova, int8.
- Car model recogniser: OpenCLIP ViT-B/16 (Apple DFN-2B weights) exported to ONNX and int8-quantised, plus a
  six-class classifier trained on 1,301 openly licensed photos of Malaysian cars.

Source code, measured results, data credits: https://github.com/ShermaineYap/saksi-jalan
