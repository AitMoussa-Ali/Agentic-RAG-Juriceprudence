
import requests
import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

BASE_URL = os.getenv("Base_url")

async def get_data(
    juridication: str,
    offset: int = 0,
    length: int = 200
) -> pd.DataFrame:

    if not BASE_URL:
        raise ValueError("La variable Base_url est absente du fichier .env")

    # if not 1 <= length <= 200:
    #     raise ValueError("length doit être compris entre 1 et 200")

    try:
        response = requests.get(
            f"{BASE_URL.rstrip('/')}/rows",
            params={
                "dataset": "antoinejeannot/jurisprudence",
                "config": "default",
                "split": juridication,
                "offset": offset,
                "length": length,
            },
            timeout=30
        )
        response.raise_for_status()

        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                "La réponse de Hugging Face n'est pas un JSON valide "
                f"(HTTP {response.status_code}, "
                f"Content-Type: {response.headers.get('Content-Type')})"
            ) from e

        rows = [item["row"] for item in data["rows"]]

        data = pd.DataFrame(rows)

        return data[["id","themes","text", "location", "numbers", "jurisdiction", "decision_date", "summary"]]

    except requests.RequestException as e:
        print(f"Erreur lors de la requête HTTP : {e}")
        raise

async def create_data():
    cour_d_appel = await get_data(juridication="cour_d_appel", length=100)
    tribunal_judiciaire = await get_data(juridication="tribunal_judiciaire", length=100)
    cour_de_cassation = await get_data(juridication="cour_de_cassation", length=100)
    data = pd.concat(
    [cour_d_appel, tribunal_judiciaire, cour_de_cassation],
    ignore_index=True
    )

    return data

