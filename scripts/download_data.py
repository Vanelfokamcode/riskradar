# scripts/download_data.py
"""
Téléchargement et organisation des données brutes.
Usage: python scripts/download_data.py --source inpi --year 2022
"""
import argparse
import requests
from pathlib import Path

RAW_DIR = Path("data/raw")

SOURCES = {
    "bodacc": {
        "url": "https://bodacc-datadila.opendatasoft.com/api/explore/v2.1/catalog/datasets/annonces-commerciales/exports/json",
        "dest": RAW_DIR / "bodacc",
    },
    # INPI nécessite un compte sur data.inpi.fr — téléchargement manuel
    # puis déposer dans data/raw/inpi/
}

def download_bodacc(dest: Path, year: int):
    dest.mkdir(parents=True, exist_ok=True)
    url = f"{SOURCES['bodacc']['url']}?where=dateparution%3E%3D%27{year}-01-01%27%20AND%20dateparution%3C%3D%27{year}-12-31%27&limit=100000"
    print(f"Téléchargement BODACC {year}...")
    r = requests.get(url, stream=True)
    out = dest / f"bodacc_{year}.json"
    with open(out, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Sauvegardé : {out} ({out.stat().st_size / 1e6:.1f} Mo)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["bodacc", "inpi"], required=True)
    parser.add_argument("--year", type=int, default=2022)
    args = parser.parse_args()

    if args.source == "bodacc":
        download_bodacc(SOURCES["bodacc"]["dest"], args.year)
    elif args.source == "inpi":
        print("INPI : téléchargement manuel requis sur data.inpi.fr")
        print("Déposer les CSV dans data/raw/inpi/ puis relancer l'ingestion.")