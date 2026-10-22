"""Download all official AudioMCQ metadata, without downloading any other model."""
import json
from concurrent.futures import ThreadPoolExecutor
import requests
from samsone_clm.common import ROOT, sha256, write_json
from samsone_clm.data import download

def main():
    api = "https://huggingface.co/api/datasets/inclusionAI/AudioMCQ"
    response = requests.get(api, timeout=30)
    response.raise_for_status()
    revision = response.json()["sha"]
    response = requests.get(f"{api}/tree/{revision}/datasets", timeout=30)
    response.raise_for_status()
    def fetch(item):
        path = ROOT / "raw" / item["path"].split("/")[-1]
        print(f"metadata start {path.name}: {item['size']} bytes", flush=True)
        download(f"https://huggingface.co/datasets/inclusionAI/AudioMCQ/resolve/{revision}/{item['path']}", path, item["size"])
        result = dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))
        print(f"metadata ready {path.name}", flush=True)
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(fetch, [x for x in response.json() if x["path"].endswith(".jsonl")]))
    write_json(ROOT / "raw/all_metadata_provenance.json", dict(repo="inclusionAI/AudioMCQ", revision=revision, files=files))

if __name__ == "__main__":
    main()
