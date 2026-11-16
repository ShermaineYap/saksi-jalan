import json, sys, subprocess, time, urllib.request, urllib.parse
UA = "SaksiResearch/0.1 (student hackathon project; contact via github)"
API = "https://commons.wikimedia.org/w/api.php"
def get(params):
    params = {**params, "format": "json"}
    url = API + "?" + urllib.parse.urlencode(params)
    for i in range(12):
        out = subprocess.run(["curl","-s","--max-time","30","-A",UA,url],capture_output=True,text=True).stdout
        try:
            return json.loads(out)
        except Exception:
            time.sleep(5 + i*5)
    raise SystemExit("failed")
def files_in(cat, depth=2, seen=None):
    seen = seen if seen is not None else set()
    out = set()
    cont = {}
    while True:
        r = get({"action":"query","list":"categorymembers","cmtitle":cat,"cmlimit":"500","cmtype":"file|subcat", **cont})
        for m in r["query"]["categorymembers"]:
            if m["ns"] == 6: out.add(m["title"])
            elif m["ns"] == 14 and depth > 0 and m["title"] not in seen:
                seen.add(m["title"]); out |= files_in(m["title"], depth-1, seen)
        if "continue" in r: cont = r["continue"]
        else: break
        time.sleep(0.3)
    return out
if __name__ == "__main__":
    for c in sys.argv[1:]:
        f = files_in("Category:" + c)
        print(f"{len(f):5d}  {c}", flush=True)
        time.sleep(0.5)
