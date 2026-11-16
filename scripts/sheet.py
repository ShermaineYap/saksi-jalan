import csv, sys, numpy as np, torch, open_clip
from PIL import Image, ImageDraw
torch.set_num_threads(2)
DATA, OUT = sys.argv[1], sys.argv[2]
model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k"); tok = open_clip.get_tokenizer("ViT-B-32"); model.eval()
scene = ["a photo of the outside of a car on a road or parking lot", "a photo of a car interior or dashboard","a photo of a car engine", "a close-up photo of a car logo or badge", "a photo of a car wheel close-up","a photo of a crowd at a motor show", "a photo of a car seat"]
rows = [r for r in csv.DictReader(open(f"{DATA}/attribution.csv"))][:445]
imgs = [Image.open(r["file"]).convert("RGB") for r in rows]
with torch.no_grad():
    E = torch.cat([model.encode_image(torch.stack([pre(i) for i in imgs[k:k+32]])) for k in range(0,len(imgs),32)]); E/=E.norm(dim=-1,keepdim=True)
    S = model.encode_text(tok(scene)); S/=S.norm(dim=-1,keepdim=True)
lab = (E@S.T).argmax(1).numpy()
rej = [i for i in range(len(rows)) if lab[i]!=0]
np.random.seed(1); pick = np.random.choice(rej, 48, replace=False)
W=160; sheet = Image.new("RGB",(8*W,6*(W+14)),"white"); d=ImageDraw.Draw(sheet)
for n,i in enumerate(pick):
    t=imgs[i].copy(); t.thumbnail((W,W)); x,y=(n%8)*W,(n//8)*(W+14); sheet.paste(t,(x,y)); d.text((x+2,y+W),str(lab[i]),fill="red")
sheet.save(OUT)
import collections; print(collections.Counter(lab.tolist()))
