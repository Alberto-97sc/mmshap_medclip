# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="9b467b94"
# # 📑 Evaluación de PubMedCLIP en ROCO
#
# Este notebook forma parte del proyecto de tesis sobre **medición del balance multimodal en modelos CLIP aplicados a dominios médicos**.
#
# Modelo actual: **PubMedCLIP**  
# Dataset: **ROCO (Radiology Objects in COntext)**  
# Tarea: **ISA (Image-Sentence Alignment)**
# ---
#

# %% [markdown] id="10c7622f"
# ## Clonar repositorio

# %% vscode={"languageId": "plaintext"} colab={"base_uri": "https://localhost:8080/"} id="87c53e96" outputId="44f82fe9-adba-41f1-c1c3-bbc237dd3176"
# 📌 Código
REPO_URL  = "https://github.com/Alberto-97sc/mmshap_medclip.git"
LOCAL_DIR = "/content/mmshap_medclip"
BRANCH    = "main"

# %cd /content
import os, shutil, subprocess, sys

if not os.path.isdir(f"{LOCAL_DIR}/.git"):
    # No está clonado aún
    # !git clone $REPO_URL $LOCAL_DIR
else:
    # Ya existe: actualiza a la última versión del remoto
    # %cd $LOCAL_DIR
    # !git fetch origin
    # !git checkout $BRANCH
    # !git reset --hard origin/$BRANCH
# %cd $LOCAL_DIR
# !git rev-parse --short HEAD

# %% [markdown] id="0efe36cb"
# ## Instalar dependencias y montar google drive

# %% vscode={"languageId": "plaintext"} colab={"base_uri": "https://localhost:8080/"} id="35d8329f" outputId="2387124d-942d-4e72-e4c7-4054a40d8f0f"
from google.colab import drive; drive.mount('/content/drive')

# === Instalar en modo editable (pyproject.toml) ===
# %pip install -e /content/mmshap_medclip

# %% vscode={"languageId": "plaintext"} id="adb0cf00"
# 📌 Código
from google.colab import drive; drive.mount('/content/drive')

# %% [markdown] id="6f4ab762"
# ## Cargar modelos y datos

# %% vscode={"languageId": "plaintext"} colab={"base_uri": "https://localhost:8080/", "height": 473, "referenced_widgets": ["de2c53d0f9e64a4cab7972d6ef9d98b1", "e570b553c7584941b613a65cb506c288", "86338ae4272e481686b909819d70a542", "8c6047cda0ec465e8843827e3be04637", "9f4108320d684ba0b5d9264da86f03d6", "2ccb325ae31e4ecdbc3823c04cfaa1c9", "af7f19730575498bbf2785647558fc8d", "9eca6ad57e4849328bc2841f62c8a39f", "20101afd1efa449cb79b1878ba1152c4", "23faddb15a3c459db714eed13d638fe6", "c6a6acb4b5944a08bc994abdada22a70", "b116552bf46249e7a0e30fa572e21614", "657a1a4e506f4079973cbc8315dd2059", "c3322eb1a57f4a64977741de42abfe4e", "b4ada1d2150b49018b0300932a72cd74", "ac5ad6a0a79a459593fec40228877812", "f3278c2e40044f5f825941400738d5ad", "cfafc12e192d4412905e26fb70d08d8e", "c5c2d1f14c234b9aac19b1ad16f6dadb", "e215ed73d2bc491b950d74f496f4709a", "3ae637a0a04d481e9576f9e056060e80", "a65822cf2412422e9800ac973daca759", "78ccf890ffef416db532f9bcefe59f15", "4c6822fe503d4f798f8161cdc4ff4294", "8aa003e0a63f4a7a90509b1ac438e571", "7633cc71835f4bd2a0c308b0e4209a5b", "a614326cb7b04205a7529722f4d9c765", "5d6964c6af43453693900495a1fd1380", "0e2079440ce143ada4aa93d893a7594f", "eaf5f2fb05d04274b640f21fff3922d9", "0286d56898494095be87710a16e814ec", "f17308f7a04a4a04bcbd5ec8e847bccd", "abce466a37034bdba3571a2a3bcbcae1", "46486eec1953451c91a3306627e53da4", "e556a950185b429197c60963ee9399c0", "d1503184caa24acc9e890d1372274421", "c0b3c6173acc4c3fa5b62ce7521c5c2e", "2325356f4bfe40a6bc2ea61e66aa6ab4", "097cc735e9314a64bff24c1e28c2a4b8", "207473e328c64eebacea53fef3b65d79", "c1cfa53376674ce1a175c220d225a916", "58db5cbbf5324a88b930381c74d1f79d", "d3c0ea883da341fe9872837a3f5d2cb5", "59a19ed9b9614906a9a6f0ef74892706", "87828897ccbd40fa895c492ef34e1e5e", "c05bd3f05ae742e1a443f14cc77d3ef3", "3708c5ddd8484d369fb19a6014ecfbfa", "6db609dbd78241a084eaa8335190ebfc", "743d9bbd970d4215911e5b7b9b531b6d", "3970d70447174170ae130be6a36c2a07", "1641861cd2304a9499246ff1173caa02", "5417ad4925434a2e8dd86e6a49991c0c", "e3f148843bfc431ba3097075d375af18", "b04853de40c742879a35bd3cf35e0eab", "de52381e64ff482daa4a5e1206d9b4d5", "d8e7adf8643f4007ab64f410d1567ad3", "da2eb4f0e9524dd4b64b8a19019c0be7", "8e6334f7ef5d45bea5e5a2fca76be409", "06746309e33541e9b6e67abacc020f82", "3253bb31fef0471192c85c239393613b", "973ee88f1175493fa6db4b7cd4e8c87e", "0bd27710a07e44ca8e1115d27e5b7261", "4afec91eaa714235a65516a3512439d4", "313c30e24fce4daa8695157e774bb3a8", "faab3b08fd3f47b6bc98c3d813f71e6b", "e6179765703b4c7ab1df507f04126b63", "a4b92f5c946a43669d8d4467a4d932ca", "f87b95efa6e04ebf9a0bb52cb5fee281", "339726464e01421e9f3cff6cb0cde32f", "97c9e02c704940ea93c7da7500abe8a6", "443ccf2370b343b79d0b48c49a40c321", "4e05b862255f46578c996e0606d0d2c3", "03c7d6e21fd74de2b298af2f90a2c2b9", "0a35453caa9c407984637543fdae50a8", "e03a595f676c45cd807cf5755044f0fe", "8abd60440a354f8b949cd980ccf62903", "347a681ea03340d3a5c0929b941638d3", "c029bf8f78674a199b9c5e5a9043a03d", "179dfbd824244b55aaa1a2e41ce21cde", "82f495de126d4ab79d3540e973ac160f", "3616e42d3c854d40be661f4945799b51", "42eba3a3168b4d40b1bfa5441be44b69", "698b1b0fb9f44c2caa21a76206c1df2c", "bd8e7c08566945c58f1b8fce8be59348", "7f1d620e9e7943cea5c3eab38222d466", "9868bf42f9724839aa62c04a2b631bb6", "219e47a702a540f0a58e356ea3eea927", "15526193fdfa41a59f9e01a3c9ba282f", "aaf5e4fcf89245438333ace091b06002", "c4c5fc4c59d94efb9934f75ca915cc11", "826c0df956554ac2b1e742a4e1a72ce6", "1b7ce1107eb54e9ba851be5834403709", "5731900572c341008bb196eb430aa45a", "cb129883177546e9854c454772cda296", "c5bb9f2e7f084eeb817c127dcff6e917", "67637003b52c48bbb43cb9c62ec80f21", "6992061753c5481eae322b5a5aafd51a", "0baaad4dfd104a64a0ded68f0ed9f8d5", "5666842ea4ec47918cfa4606be22f75e"]} id="b6485339" outputId="552032cc-4242-48cf-cc5a-447c96a68d9a"
# 📌 Código
CFG_PATH="/content/mmshap_medclip/configs/roco_isa_pubmedclip.yaml"

# Asegura que cfg, device, dataset y model estén listos en esta sesión
if not all(k in globals() for k in ("cfg", "device", "dataset", "model")):
    from mmshap_medclip.io_utils import load_config
    from mmshap_medclip.devices import get_device
    from mmshap_medclip.registry import build_dataset, build_model

    cfg = load_config(CFG_PATH)
    device  = get_device()
    dataset = build_dataset(cfg["dataset"])
    model   = build_model(cfg["model"], device=device)

print("OK → len(dataset) =", len(dataset), "| device =", device)

# %% [markdown] id="55231ba0"
# ## Ejecutar SHAP en una muestra

# %% vscode={"languageId": "plaintext"} colab={"base_uri": "https://localhost:8080/", "height": 647} id="0d06718a" outputId="02b489ac-f76d-41c4-a94f-b00e78b13e6a"
# 📌 Código
from mmshap_medclip.tasks.isa import run_isa_one

muestra = 154
sample  = dataset[muestra]
image, caption = sample['image'], sample['text']

res = run_isa_one(model, image, caption, device, explain=True, plot=True)
print(f"logit={res['logit']:.4f}  TScore={res['tscore']:.2%}  IScore={res['iscore']:.2%}")



# %% colab={"base_uri": "https://localhost:8080/", "height": 877} id="tzOwIfL0D5-5" outputId="8737fc46-dd5b-4202-f6d9-ce3911db744d"
# ==== Config ====
zip_path = "/content/drive/MyDrive/MAESTRIA-TESIS/datasets/ROCO/dataset_roco.zip"
use_negatives = False      # True para zero-shot con negativos
num_negatives = 15
reproducible = False       # <-- CAMBIO: por defecto aleatorio distinto cada corrida
seed = 42                  # se usa SOLO si reproducible=True

# ==== Funciones auxiliares ====
import zipfile, io, re, random
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO

# si quieres reproducibilidad, fija semillas aquí
if reproducible:
    random.seed(seed)
    np.random.seed(seed)

def clean_roco_caption(text):
    t = re.sub(r"^\s*ROCO_\d+\s*[\t,|;]\s*", "", str(text)).strip()
    t = re.sub(r"\s+", " ", t)
    return t

def load_radiology_df_from_zip(zf, split="train"):
    candidates = [n for n in zf.namelist() if n.startswith(f"all_data/{split}/radiology/") and n.lower().endswith(".csv")]
    if not candidates:
        raise RuntimeError("No encontré CSVs de radiology en el split.")
    raw = zf.read(candidates[0])
    df = None
    for enc in ("utf-8","utf-8-sig","latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            if not df.empty:
                break
        except Exception:
            df = None
    if df is None or df.empty:
        raise RuntimeError("No pude leer el CSV.")
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("name")
    cap_col  = cols.get("caption")
    if name_col is None or cap_col is None:
        raise RuntimeError("No encontré columnas 'name' y/o 'caption'.")
    return df, name_col, cap_col

def pick_chestxray_sample(zf, split="train"):
    df, name_col, cap_col = load_radiology_df_from_zip(zf, split)
    # filtrar por captions que contengan 'chest x-ray' o 'lung'
    mask = df[cap_col].astype(str).str.lower().str.contains("chest x-ray|lung")
    filtered = df[mask]
    if filtered.empty:
        raise RuntimeError("No encontré captions con 'chest x-ray' o 'lung'.")

    # === CAMBIO: random_state dinámico ===
    rs = seed if reproducible else None
    row = filtered.sample(1, random_state=rs).iloc[0]

    img_name = str(row[name_col])
    cap = clean_roco_caption(row[cap_col])
    img_rel = f"all_data/{split}/radiology/images/{Path(img_name).name}"
    if img_rel not in zf.namelist():
        cand = f"all_data/{split}/radiology/{img_name}"
        if cand in zf.namelist():
            img_rel = cand
        else:
            matches = [n for n in zf.namelist() if n.endswith(Path(img_name).name)]
            if matches:
                img_rel = matches[0]
            else:
                raise RuntimeError("No encontré la imagen en el ZIP.")
    return img_rel, cap, df, name_col, cap_col

def sample_negative_captions(df, cap_col, positive_caption, k=15):
    # === CAMBIO: random_state dinámico y sin barajar con semilla fija ===
    rs = seed if reproducible else None
    candidates = (
        df[cap_col]
        .astype(str)
        .map(clean_roco_caption)
        .dropna()
        .drop_duplicates()
    )
    candidates = candidates[candidates.str.len() > 0]
    candidates = candidates[candidates.str.lower() != str(positive_caption).lower()]
    if candidates.empty:
        return []
    cands_list = candidates.sample(min(k, len(candidates)), random_state=rs).tolist()
    if not reproducible:
        random.shuffle(cands_list)
    return cands_list[:k]

# ==== Seleccionar imagen y caption ====
from IPython.display import display

with zipfile.ZipFile(zip_path, "r") as z:
    image_rel, caption_clean, df_all, name_col, cap_col = pick_chestxray_sample(z, split="train")
    img = Image.open(BytesIO(z.read(image_rel))).convert("RGB")

print("Imagen seleccionada:", image_rel)
print("Caption limpio:", caption_clean)
display(img)

# ==== WhyXrayCLIP ====
# !pip install -q open_clip_torch pillow torch torchvision --upgrade
import torch, open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:yyupenn/whyxrayclip")
tokenizer = open_clip.get_tokenizer("ViT-L-14")
model.to(device).eval()

image_tensor = preprocess(img).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image_tensor)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

if not use_negatives:
    with torch.no_grad():
        text_pos = tokenizer([caption_clean]).to(device)
        text_features = model.encode_text(text_pos)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        sim = (image_features @ text_features.T).item()      # [-1, 1] aprox
        score_01 = (sim + 1.0) / 2.0                         # [0, 1]

    print("\n=== Alineación imagen–caption (sin negativos) ===")
    print(f"Similitud coseno: {sim:.4f}   |   Score[0,1]: {score_01:.4f}")

else:
    negatives = sample_negative_captions(df_all, cap_col, caption_clean, k=num_negatives)
    candidates = [caption_clean] + negatives

    with torch.no_grad():
        text_tok = tokenizer(candidates).to(device)
        text_features = model.encode_text(text_tok)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * (image_features @ text_features.T)  # [1, 1+N]
        probs = logits.softmax(dim=-1).squeeze(0).tolist()

        pairs = list(zip(candidates, probs))
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        rank_true = 1 + next(i for i, (c, _) in enumerate(pairs_sorted) if c == caption_clean)

    print("\n=== Zero-shot elección (con negativos) ===")
    print(f"Total candidatos: {len(candidates)} | Rank del caption verdadero: {rank_true}")
    print("\nTop-5 candidatos por probabilidad:")
    for lbl, p in pairs_sorted[:5]:
        short = (lbl[:120] + "…") if len(lbl) > 120 else lbl
        print(f"{p:7.4f} | {short}")

