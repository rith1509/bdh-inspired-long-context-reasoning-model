import pandas as pd
import ollama
import numpy as np
import os
import re
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_DIR = "data_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_NAME = "nomic-embed-text" # Make sure to run: ollama pull nomic-embed-text
CHUNK_SIZE = 1000

# --- COLUMN MAPPING (Based on your "inspect_files.py" output) ---
COL_BACKSTORY = "content"       # The backstory text is in 'content'
COL_BOOK_NAME = "book_name"     # The book title
COL_LABEL = "label"             # "consistent"
COL_ID = "id"

# --- HELPER 1: FILE MATCHING ---
def find_novel_file(book_name_from_csv, folder_path="."):
    """Matches CSV book name to the actual .txt file on disk (fuzzy match)"""
    if pd.isna(book_name_from_csv): return None
    
    # Normalize CSV name: remove punctuation, lowercase
    clean_csv_name = re.sub(r'[^\w\s]', '', str(book_name_from_csv)).lower().strip()
    
    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"): continue
        
        # Normalize disk filename
        clean_disk_name = re.sub(r'[^\w\s]', '', filename.replace(".txt", "")).lower().strip()
        
        # Check intersection
        if clean_csv_name in clean_disk_name or clean_disk_name in clean_csv_name:
            return filename
            
    return None

# --- HELPER 2: CHUNKING & EMBEDDING ---
def smart_chunking(text, chunk_size=1000):
    if pd.isna(text): return [""]
    text = str(text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    if current_chunk: chunks.append(" ".join(current_chunk))
    return chunks

def get_embeddings_ollama(text_chunks):
    vectors = []
    for chunk in text_chunks:
        if not chunk.strip(): continue
        try:
            response = ollama.embeddings(model=MODEL_NAME, prompt=chunk)
            vectors.append(response['embedding'])
        except Exception as e:
            print(f"Error embedding chunk: {e}")
    if not vectors: return np.zeros((1, 768), dtype=np.float32)
    return np.array(vectors, dtype=np.float32)

# --- HELPER 3: ENTITY OVERLAP SCORE (The Fix for Hallucinations) ---
def calculate_entity_score(backstory_text, novel_text):
    """
    Returns a score 0.0 to 1.0. 
    Low score = Backstory mentions Proper Nouns NOT found in the Novel (Hallucination).
    """
    if pd.isna(backstory_text) or pd.isna(novel_text): return 0.5
    
    # Find Capitalized Words > 3 chars (Simple Named Entity Recognition)
    # Ignore start of sentences to avoid common words.
    b_entities = set(re.findall(r'(?<!^)\b[A-Z][a-z]{3,}\b', str(backstory_text)))
    
    if not b_entities: return 1.0 # No entities to contradict
    
    novel_str = str(novel_text)
    found_count = 0
    for w in b_entities:
        if w in novel_str:
            found_count += 1
            
    return found_count / len(b_entities)

# --- MAIN PROCESSOR ---
def process_dataset(csv_path, is_train=True):
    print(f"\n=== Processing {csv_path} ===")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}")
        return

    novel_cache = {} # Store loaded novel vectors in RAM
    text_cache = {}  # Store loaded novel text strings in RAM

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        row_id = str(row[COL_ID])
        
        # 1. PROCESS LABEL
        if is_train:
            label_str = str(row[COL_LABEL]).lower().strip()
            # "consistent" = 1.0, anything else = 0.0
            label_val = 1.0 if "consistent" in label_str else 0.0
            np.save(os.path.join(OUTPUT_DIR, f"{row_id}_label.npy"), np.array([label_val], dtype=np.float32))
        
        # 2. PROCESS BACKSTORY
        backstory_text = row[COL_BACKSTORY]
        b_chunks = smart_chunking(backstory_text, CHUNK_SIZE)
        b_vecs = get_embeddings_ollama(b_chunks)
        np.save(os.path.join(OUTPUT_DIR, f"{row_id}_backstory.npy"), b_vecs)
        
        # 3. PROCESS NOVEL & ENTITY SCORE
        book_name = row[COL_BOOK_NAME]
        actual_filename = find_novel_file(book_name)
        
        if not actual_filename:
            print(f" [!] Missing file for: {book_name}. Saving dummies.")
            np.save(os.path.join(OUTPUT_DIR, f"{row_id}_novel.npy"), np.zeros((1, 768), dtype=np.float32))
            np.save(os.path.join(OUTPUT_DIR, f"{row_id}_entity_score.npy"), np.array([0.5], dtype=np.float32))
            continue

        # Load Text and Embeddings (Check Cache)
        if actual_filename in novel_cache:
            n_vecs = novel_cache[actual_filename]
            novel_text = text_cache[actual_filename]
        else:
            try:
                with open(actual_filename, 'r', encoding='utf-8', errors='ignore') as f:
                    novel_text = f.read()
                n_chunks = smart_chunking(novel_text, CHUNK_SIZE)
                n_vecs = get_embeddings_ollama(n_chunks)
                
                novel_cache[actual_filename] = n_vecs
                text_cache[actual_filename] = novel_text
            except Exception as e:
                print(f"Error reading {actual_filename}: {e}")
                continue
        
        # Save Novel Embeddings
        np.save(os.path.join(OUTPUT_DIR, f"{row_id}_novel.npy"), n_vecs)
        
        # Calculate & Save Entity Score
        score = calculate_entity_score(backstory_text, novel_text)
        np.save(os.path.join(OUTPUT_DIR, f"{row_id}_entity_score.npy"), np.array([score], dtype=np.float32))

if __name__ == "__main__":
    if os.path.exists("train.csv"): process_dataset("train.csv", is_train=True)
    if os.path.exists("test.csv"): process_dataset("test.csv", is_train=False)