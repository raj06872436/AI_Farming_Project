"""Download the PlantDoc classes that overlap with AGRI-X's 15 PlantVillage classes (field photos)."""
import json, os, sys, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
REPO = "pratikkayal/PlantDoc-Dataset"
MAP = {
    "Bell_pepper leaf spot": "Pepper__bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper__bell___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Tomato leaf bacterial spot": "Tomato_Bacterial_spot",
    "Tomato Early blight leaf": "Tomato_Early_blight",
    "Tomato leaf late blight": "Tomato_Late_blight",
    "Tomato mold leaf": "Tomato_Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato_Septoria_leaf_spot",
    "Tomato leaf yellow virus": "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato leaf mosaic virus": "Tomato__Tomato_mosaic_virus",
    "Tomato leaf": "Tomato_healthy",
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PlantDoc")

def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "agrix"}), timeout=60).read()

jobs = []
for split in ("train", "test"):
    for src, dst in MAP.items():
        items = json.loads(get(f"https://api.github.com/repos/{REPO}/contents/{urllib.parse.quote(split + '/' + src)}"))
        for it in items:
            if it["type"] == "file":
                jobs.append((it["download_url"], os.path.join(OUT, dst, f"{split}_{it['name']}")))
print(len(jobs), "files"); sys.stdout.flush()

def dl(job):
    url, path = job
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        data = get(url)
        open(path, "wb").write(data)
        return 1
    except Exception as e:
        return str(e)

with ThreadPoolExecutor(16) as ex:
    res = list(ex.map(dl, jobs))
print("downloaded", sum(r == 1 for r in res), "errors", [r for r in res if isinstance(r, str)][:5])
