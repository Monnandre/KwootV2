from langchain.text_splitter import CharacterTextSplitter
from tqdm import tqdm
from uuid import uuid4
import json
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

def open_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()
    
def get_chunks(raw_text):
    splitter = CharacterTextSplitter(
        separator=".",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = splitter.split_text(raw_text)
    
    # Calculate line numbers for each chunk
    lines = raw_text.split('\n')
    line_starts = []
    current_line = 0
    current_pos = 0
    
    for chunk in chunks:
        while current_pos < len(raw_text) and raw_text[current_pos:current_pos+10] != chunk[0:10]:
            if raw_text[current_pos] == '\n':
                current_line += 1
            current_pos += 1
        line_starts.append(current_line)
    
    return list(zip(chunks, line_starts))

def save_chunks_to_json(chunks_with_lines, filename):
    data = {}
    for chunk, line in chunks_with_lines:
        chunk_id = str(uuid4())
        data[chunk_id] = {
            "text": chunk.replace("\n", " "),
            "filename": filename,
            "line_start": line + 1
        }
    
    if os.path.exists("data.json"):
        with open("data.json", "r", encoding="utf-8") as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = {}

    existing_data.update(data)

    with open("data.json", "w", encoding="utf-8") as json_file:
        json.dump(existing_data, json_file, ensure_ascii=False, indent=4)
    
    return data

def embedd_texts(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options": {"wait_for_model": True}})
    return response.json()

def save_embedds(embeddings, chunks, namespace):
    formated_embeddings = []
    for i, vector in enumerate(embeddings): 
        formated_embeddings.append({
            "id": chunks[i]["id"],
            "values": vector,
            "metadata": {
                'text': chunks[i]["text"],
                'filename': chunks[i]["filename"],
                'line_start': chunks[i]["line_start"]
                }
        })

    index.upsert(
        vectors=formated_embeddings,
        namespace=namespace
    )

files = ["wharton.txt", "suppliantes.txt", "spinoza.txt", "septs_contre_thebes.txt"]
for file in files:
    raw_text = open_file(file)
    chunks_with_lines = get_chunks(raw_text)

    chunks_objects = [{"id": key, **value} for key, value in save_chunks_to_json(chunks_with_lines, file).items()]

    batch_size = 50
    for i in tqdm(range(0, len(chunks_objects), batch_size)):
        chunk_batch = chunks_objects[i:i + batch_size]
        chunk_batch_texts = [chunk["text"] for chunk in chunk_batch]
        vectors = embedd_texts(chunk_batch_texts)
        save_embedds(vectors, chunk_batch, file[:-4])