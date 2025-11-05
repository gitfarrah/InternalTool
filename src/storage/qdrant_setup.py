import os
from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient, models

# Safely load .env even when running from stdin (-) or interactive shells
try:
    # Default behavior (search relative to the caller file)
    load_dotenv()
except AssertionError:
    # Workaround python-dotenv stack inspection issue; search from CWD upwards
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION = "docs"
VECTOR_NAME = "content_vector"
DIM = 768

def recreate_collection():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    try:
        client.delete_collection(COLLECTION)
        print(f" Deleted existing collection '{COLLECTION}'")
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            VECTOR_NAME: models.VectorParams(
                size=DIM,
                distance=models.Distance.COSINE,
            )
        },
    )
    print(f"Created collection '{COLLECTION}' with {DIM}-dim cosine vector '{VECTOR_NAME}'")

    for field, ftype in [
        ("source", models.PayloadSchemaType.KEYWORD),
        ("url", models.PayloadSchemaType.KEYWORD),
        ("page_id", models.PayloadSchemaType.INTEGER),
        ("chunk_id", models.PayloadSchemaType.INTEGER),
    ]:
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=ftype,
        )
    print("Created payload indexes: source, url, page_id, chunk_id")

if __name__ == "__main__":
    recreate_collection()
