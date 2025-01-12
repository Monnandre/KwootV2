import os
import requests
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# === Étape 1 : Configurer Hugging Face ===
model_id = os.environ.get("EMBEDDINGS_MODEL_ID")
hf_token = os.environ.get("HF_TOKEN")
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

# === Étape 2 : Configurer Pinecone ===
pinecone_api_key = os.environ.get("PINECONE_TOKEN")
index_name = os.environ.get("PINECONE_INDEX")
pinecone_environment = os.environ.get("PINECONE_ENV")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)


def embedd_texts(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
    return response.json()

def embedd_text(text):
    return embedd_texts([text])[0]