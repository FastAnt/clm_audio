from samsone_clm.common import ROOT, sha256, write_json
from samsone_clm.data import download


def main():
    assets = [
        ("https://github.com/SamsungLabs/samsone/releases/download/v1.0.0/samsone_356m_model_bf16.ckpt",
         ROOT / "models/samsone_356m_model_bf16.ckpt", 713899355,
         "8ed28e35aff5cb724bc84d1b2ea1559492e88e9c53abd1439f7e0a397214fea8"),
        ("https://huggingface.co/Contrastive-LM/CLM-v0.1-8B/resolve/main/CLM_v0.1-8B.pt",
         ROOT / "models/clm/CLM_v0.1-8B.pt", 75557149,
         "b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5"),
    ]
    report = []
    for url, target, size, expected in assets:
        download(url, target, size)
        actual = sha256(target)
        if actual != expected:
            raise ValueError(f"Official checkpoint checksum mismatch: {target}")
        report.append(dict(url=url, path=str(target), sha256=actual, bytes=size))
    write_json(ROOT / "models/provenance.json", report)


if __name__ == "__main__":
    main()
