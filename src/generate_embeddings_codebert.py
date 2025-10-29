"""
Generate CodeBERT Mean Embeddings and Labels
--------------------------------------------
This script converts combined_data.pkl into NumPy arrays of
mean-pooled CodeBERT embeddings (768-dim) and binary labels.

Model: microsoft/codebert-base
Output: <project>_embeddings.npy, <project>_labels.npy
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel


# ==============================
# Configuration
# ==============================
DATA_PATH = "dataset/combined_data.pkl"
OUTPUT_DIR = "dataset/"
MODEL_NAME = "microsoft/codebert-base"
MAX_LEN = 512
BATCH_SIZE = 8
DEVICE = "cuda"  # or "cpu"
# ==============================


def load_dataset(path):
    print(f"Loading dataset from {path} ...")
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"Projects found: {list(data.keys())}")
    return data


def get_codebert_model():
    import torch
    from torch import no_grad
    print(f"Loading CodeBERT model ({MODEL_NAME}) ...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaModel.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)
    model.eval()
    return tokenizer, model, no_grad, torch


def embed_code_snippets(snippets, tokenizer, model, no_grad, torch):
    """Generate mean-pooled embeddings for each code snippet."""
    all_embeddings = []

    with no_grad():
        for i in tqdm(range(0, len(snippets), BATCH_SIZE)):
            batch = snippets[i:i + BATCH_SIZE]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN
            ).to(DEVICE)
            outputs = model(**inputs)
            # Mean pooling across all tokens (dim=1)
            mean_embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(mean_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def save_embeddings(project, embeddings, labels):
    emb_path = os.path.join(OUTPUT_DIR, f"{project.lower()}_embeddings.npy")
    lab_path = os.path.join(OUTPUT_DIR, f"{project.lower()}_labels.npy")
    np.save(emb_path, embeddings)
    np.save(lab_path, labels)
    print(f"Saved {project}: {embeddings.shape} → {emb_path}, {lab_path}")


if __name__ == "__main__":
    import torch

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined_data = load_dataset(DATA_PATH)
    tokenizer, model, no_grad, torch = get_codebert_model()

    for project_name, samples in combined_data.items():
        print(f"\nProcessing project: {project_name}")

        # Each sample expected as either dict {"func": code, "target": label}
        # or tuple (code, label)
        codes = [s["func"] if isinstance(s, dict) else s[0] for s in samples]
        labels = np.array([s["target"] if isinstance(s, dict) else s[1] for s in samples])

        embeddings = embed_code_snippets(codes, tokenizer, model, no_grad, torch)
        save_embeddings(project_name, embeddings, labels)

    print("\nAll mean-pooled CodeBERT embeddings (dim=768) generated successfully.")
