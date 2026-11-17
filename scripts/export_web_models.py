"""Export the models used by the in-browser app (web/).

1. OpenCLIP ViT-B/16 (DFN-2B) image encoder -> ONNX, output L2-normalised.
2. Weight-only int8: each large MatMul weight is stored as int8 + per-channel scale and
   dequantised at load, so compute stays fp32. 345 MB -> 89 MB, cosine similarity to fp32 0.9996
   (full dynamic int8 quantisation only reached 0.91 and lost 6 points of accuracy).
3. The trained classifier -> probe.json (weights, bias, class names).

Usage: python3 export_web_models.py PROBE_JOBLIB OUT_DIR
The detector is Xenova/detr-resnet-50 onnx/model_quantized.onnx from the Hugging Face Hub, used unchanged.
"""
import json, os, sys

import joblib, numpy as np, onnx, open_clip, torch
from onnx import helper, numpy_helper

probe_path, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

m, _, _ = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="dfn2b")
m.eval()


class Enc(torch.nn.Module):
    def __init__(self, v):
        super().__init__(); self.v = v

    def forward(self, x):
        f = self.v(x); return f / f.norm(dim=-1, keepdim=True)


fp32 = os.path.join(out, "clip_visual_fp32.onnx")
torch.onnx.export(Enc(m.visual).eval(), torch.randn(1, 3, 224, 224), fp32, input_names=["pixels"],
                  output_names=["embedding"], dynamic_axes={"pixels": {0: "b"}, "embedding": {0: "b"}},
                  opset_version=17, dynamo=False)

g = onnx.load(fp32)
inits = {i.name: i for i in g.graph.initializer}
new_inits, new_nodes = [], []
for node in g.graph.node:
    if node.op_type != "MatMul":
        continue
    for k, name in enumerate(node.input):
        if name not in inits:
            continue
        W = numpy_helper.to_array(inits[name]).astype(np.float32)
        if W.ndim != 2 or W.size < 4096:
            continue
        sc = np.abs(W).max(0) / 127.0; sc[sc == 0] = 1e-8
        q = np.clip(np.round(W / sc), -127, 127).astype(np.int8)
        new_inits += [numpy_helper.from_array(q, name + "_q"), numpy_helper.from_array(sc.astype(np.float32), name + "_s"),
                      numpy_helper.from_array(np.zeros_like(sc, dtype=np.int8), name + "_z")]
        new_nodes.append(helper.make_node("DequantizeLinear", [name + "_q", name + "_s", name + "_z"], [name + "_dq"], axis=1))
        node.input[k] = name + "_dq"; inits.pop(name)
keep = [i for i in g.graph.initializer if i.name in inits]
del g.graph.initializer[:]; g.graph.initializer.extend(keep + new_inits)
nodes = list(g.graph.node); del g.graph.node[:]; g.graph.node.extend(new_nodes + nodes)
onnx.save(g, os.path.join(out, "clip_visual_w8.onnx"))
os.remove(fp32)

b = joblib.load(probe_path)
json.dump({"classes": b["classes"], "W": b["clf"].coef_.round(6).tolist(), "b": b["clf"].intercept_.round(6).tolist()},
          open(os.path.join(out, "probe.json"), "w"))
print("wrote", os.listdir(out))
