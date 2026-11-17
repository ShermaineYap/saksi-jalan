"""Compose the Saksi Jalan demo video: 1920x1080, 30 fps, narration + burnt-in captions."""
import json, os, subprocess
from PIL import Image, ImageDraw, ImageFont

V = "/home/claude/video"
T = json.load(open(f"{V}/timing.json"))
M = json.load(open(f"{V}/marks.json")); mk, REC = M["marks"], M["video"]
LEAD, GAP, TAIL, FADE = 0.5, 0.4, 0.9, 0.3
ORDER = ["s01_title", "s02_problem", "s03_why", "s04_demo_a", "s04_demo_b", "s04_demo_c",
         "s05_how", "s06_results", "s07_learning", "s08_next", "s09_end"]
SLIDE = {"s04_demo_a": "demo_a", "s04_demo_b": "demo_b", "s04_demo_c": "demo_c"}
ENC = ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-r", "30"]
SX, SY, SW, SH = 336, 46, 401, 868   # phone screen area on the demo slides
os.makedirs(f"{V}/parts", exist_ok=True)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr[-2000:])


def dur(sid):
    lines = T[sid]
    return LEAD + sum(d for _, d, _ in lines) + GAP * (len(lines) - 1) + TAIL


# label shown while the first analysis is sped up
lab = Image.new("RGBA", (380, 46), (0, 0, 0, 0)); d = ImageDraw.Draw(lab)
d.rounded_rectangle((0, 0, 379, 45), 8, fill=(47, 50, 55, 230))
d.text((16, 9), "Sped up 6x while the models start", font=ImageFont.truetype(f"{V}/fonts/AtkinsonHyperlegible-Bold.ttf", 22), fill="white")
lab.save(f"{V}/parts/spedup.png")


def fade(D):
    return f"fade=t=in:st=0:d={FADE},fade=t=out:st={D - FADE:.3f}:d={FADE}"


def still(sid, D):
    out = f"{V}/parts/{sid}.mp4"; tmp = out.replace(".mp4", ".tmp.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-framerate", "30", "-i", f"{V}/slides/{sid}.png",
         "-vf", f"format=yuv420p,loop=-1:1,trim=duration={D:.3f},setpts=PTS-STARTPTS,{fade(D)}", *ENC, tmp])
    os.replace(tmp, out)


def demo(sid, D):
    """Phone recording over the demo slide. pieces: (start, end, speed)."""
    if sid == "s04_demo_a":
        pieces = [(mk["a_start"], mk["a_click"], 1.0), (mk["a_click"], mk["a_result"], 6.0), (mk["a_result"], mk["a_end"], 1.0)]
    elif sid == "s04_demo_b":
        pieces = [(mk["b_start"], mk["b_end"], 1.0)]
    else:
        span = mk["c_end"] - mk["c_start"]; sp = max(1.0, span / (D - 0.3))
        pieces = [(mk["c_start"], mk["c_end"], sp)]
    f, labels, t = [], [], 0.0
    for k, (a, b, sp) in enumerate(pieces):
        f.append(f"[1:v]trim=start={a:.3f}:end={b:.3f},setpts=(PTS-STARTPTS)/{sp},crop=390:844:0:0,scale={SW}:{SH}:flags=lanczos,fps=30[p{k}]")
        if sp > 1.5:
            labels.append((t, t + (b - a) / sp))
        t += (b - a) / sp
    f.append("".join(f"[p{k}]" for k in range(len(pieces))) + f"concat=n={len(pieces)}:v=1:a=0,tpad=stop_mode=clone:stop_duration={D:.3f},trim=duration={D:.3f}[ph]")
    f.append(f"[0:v][ph]overlay={SX}:{SY}:shortest=0[v1]")
    last = "v1"
    for i, (a, b) in enumerate(labels):
        f.append(f"[{last}][2:v]overlay={SX + 10}:{SY + SH - 60}:enable='between(t,{a:.2f},{b:.2f})'[v{i + 2}]"); last = f"v{i + 2}"
    f.append(f"[{last}]format=yuv420p,trim=duration={D:.3f},{fade(D)}[out]")
    out = f"{V}/parts/{sid}.mp4"; tmp = out.replace(".mp4", ".tmp.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-framerate", "30", "-loop", "1", "-t", f"{D:.3f}", "-i", f"{V}/slides/{SLIDE[sid]}.png",
         "-i", REC, "-loop", "1", "-i", f"{V}/parts/spedup.png", "-filter_complex", ";".join(f), "-map", "[out]", "-t", f"{D:.3f}", *ENC, tmp])
    os.replace(tmp, out)


def ts(x):
    h = int(x // 3600); m = int(x % 3600 // 60); s = x % 60
    return f"{h}:{m:02d}:{s:05.2f}"


# ---- render scenes, build audio + captions
ass = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1920", "PlayResY: 1080", "WrapStyle: 0", "",
       "[V4+ Styles]",
       "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
       "Style: Cap,Atkinson Hyperlegible,40,&H00FFFFFF,&H00FFFFFF,&H00373230,&H00373230,1,0,0,0,100,100,0,0,3,14,0,2,260,260,34,1",
       "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
audio_inputs, concat_list, t0 = [], [], 0.0
for sid in ORDER:
    D = dur(sid)
    (demo if sid in SLIDE else still)(sid, D)
    concat_list.append(f"file '{V}/parts/{sid}.mp4'")
    t = t0 + LEAD
    for f, d, text in T[sid]:
        audio_inputs.append((f, t))
        ass.append(f"Dialogue: 0,{ts(t)},{ts(t + d)},Cap,,0,0,0,,{text}")
        t += d + GAP
    t0 += D
    print(sid, round(D, 2), flush=True)
TOTAL = t0
open(f"{V}/parts/list.txt", "w").write("\n".join(concat_list) + "\n")
open(f"{V}/parts/captions.ass", "w").write("\n".join(ass) + "\n")
run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", f"{V}/parts/list.txt", "-c", "copy", f"{V}/parts/video_nosound.mp4"])

# narration track: place each line at its time
inputs, flt = [], []
for i, (f, t) in enumerate(audio_inputs):
    inputs += ["-i", f]; flt.append(f"[{i}:a]aresample=48000,adelay={int(t * 1000)}|{int(t * 1000)}[a{i}]")
mix = "".join(f"[a{i}]" for i in range(len(audio_inputs))) + f"amix=inputs={len(audio_inputs)}:normalize=0,apad,atrim=duration={TOTAL:.3f},loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(flt + [mix]), "-map", "[aout]", "-ac", "2", "-ar", "48000", f"{V}/parts/narration.wav"])

# final: burn captions, add sound (one re-encode)
out = f"{V}/saksi_jalan_demo.mp4"; tmp = out.replace(".mp4", ".tmp.mp4")
run(["ffmpeg", "-y", "-v", "error", "-i", f"{V}/parts/video_nosound.mp4", "-i", f"{V}/parts/narration.wav",
     "-vf", f"subtitles={V}/parts/captions.ass:fontsdir={V}/fonts", "-map", "0:v", "-map", "1:a",
     *ENC, "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", tmp])
os.replace(tmp, out)
print("TOTAL", round(TOTAL, 1), out)
