import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import torch
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from Dataset.upload import get_data
from typing import Literal
import pandas as pd

import unicodedata
import re


# DB_PATH = Path(".").resolve().parent / "Collections"

# print("PyTorch:", torch.__version__)
# print("CUDA build:", torch.version.cuda)
# print("CUDA available:", torch.cuda.is_available())

# device = "cuda" if torch.cuda.is_available() else "cpu"

# Helping functions

def clear_text(text: str) -> str:
    # Normalize Unicode characters
    text = unicodedata.normalize("NFC", text)

    # Remove common page-break and control characters
    text = text.replace("\x0c", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Remove spaces around hyphens used as line-wrap artifacts
    # (Avoid blindly joining words around every hyphen.)

    return text.strip()

def chunk_characters(text : str, overlap : int, chunk_size : int, chunk_type : Literal["recursive", "simple"] = "recursive") -> list : 
    
    if chunk_type == "simple" : 
        splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
    elif chunk_type == "recursive" : 
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
        )


    chunks = splitter.split_text(text)

    return chunks

def chunk_docs(
        data : pd.DataFrame,
        chunk_size : int = 300,
        chunk_overlap : int = 100
) :
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for row in data.itertuples(index = False):

        if pd.isna(row.text) or not str(row.text).strip():
            continue

        source_id = str(row.id)
        source_text = str(row.text)
        location = str(row.location) if row.location is not None else ""
        source_themes = row.themes
        decision_date = row.decision_date

        if isinstance(source_themes, (list, tuple, set)):
            themes_text = ", ".join(map(str, source_themes))
        elif source_themes is None or (
            isinstance(source_themes, float) and pd.isna(source_themes)
        ):
            themes_text = ""
        else : 
            themes_text = str(source_themes)

        #Chunking part :
        clean_text = clear_text(source_text)
        chunks = chunk_characters(
            text=clean_text,
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )
        #Building the metadata
        numbers = ""
        
        for num in row.numbers :
            numbers += f"{str(num)}, "

        for chunk_index, chunk in enumerate(chunks):
            chunk = str(chunk).strip()

            if not chunk:
                continue
            # Building the chunk id
            chunk_id = f"{source_id}_chunk_{chunk_index}"

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "source_id": source_id,
                    "juridiction" : row.jurisdiction,
                    "themes" : themes_text,
                    "location" : location, 
                    "Numéros" : numbers,
                    "Date de decision" : row.decision_date,
                    "Résumé" : row.summary if row.summary is not None else ""
                }
            )

    return documents, ids, metadatas

def embed_docs(
        documents : list,
        ids : list,
        metadatas : list,
        batch_size : int,
        collection : chromadb.Collection
):
    #embed the documents
    try:
        for start in range(0, len(documents), batch_size):
                end = start + batch_size

                batch_documents = documents[start:end]
                batch_ids = ids[start:end]
                batch_metadatas = metadatas[start:end]

                embeddings = embedding_model.encode(
                    batch_documents,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=True,
                    batch_size=batch_size,
                    device=device
                )

                collection.add(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    embeddings=embeddings.tolist(),
                )      
        print("The data are uploaded to the collection successfully ! ")
        return True
    except Exception:
        return False

def create_and_upload_to_collection(
        batch_size : int,
        data : pd.DataFrame,
        chunk_overlap : int = 100,
        chunk_size : int = 300,
        collection_name : str = "jurisprudence",
):

    #Creating the collection
    client = chromadb.PersistentClient(path=str(DB_PATH))

    collection = client.get_or_create_collection(
        name=collection_name,
        configuration={
            "hnsw" : {
                "space":"cosine"
            }
        }
    )

    #Getting the data after chunking
    documents, ids, metadatas = chunk_docs(data = data, chunk_size=chunk_size, chunk_overlap = chunk_overlap)

    #Embedding the data and upload to the collection
    result = embed_docs(
        documents=documents,
        ids = ids,
        metadatas=metadatas,
        batch_size=batch_size,
        collection=collection
    )

    if result and collection.count() != 0:
        print(f"Uploaded {collection.count()} chunks into the the collection '{collection_name}'")
    else :
        print("An error has occured during the upload of the data")

    return collection
