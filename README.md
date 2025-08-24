# mmshap_medclip

Pipeline modular para medir el **balance multimodal** con **SHAP** en modelos tipo **CLIP** (incluye PubMedCLIP) sobre datasets médicos (p. ej., ROCO). Diseñado para ejecutarse en **Colab + Google Drive**, versionar en **GitHub**, y escalar a más modelos/datasets/tareas.

> Esta versión asume **instalación con `pyproject.toml`** y uso de **`pip install -e .`**.

---

## Estructura del repo

```
mmshap_medclip/
├── notebooks/
│   └── 01_heatmaps_and_figures.ipynb      # SOLO figuras/tablas del paper
├── src/mmshap_medclip/
│   ├── __init__.py
│   ├── devices.py                          # manejo de device (CUDA/CPU)
│   ├── registry.py                         # registro de modelos y datasets
│   ├── models.py                           # wrappers de CLIP (openai/pubmed…)
│   ├── datasets/
│   │   ├── base.py                         # interfaz DatasetBase
│   │   └── roco.py                         # loader ROCO (lee ZIP en Drive)
│   ├── tasks/
│   │   └── utils.py                        # prepare_batch, token lengths, etc.
│   ├── shap_tools/
│   │   ├── masker.py                       # build_masker (BOS/EOS safe)
│   │   └── predictor.py                    # Predictor callable para SHAP
│   ├── metrics.py                          # MM-score, IScore
│   ├── vis/
│   │   └── heatmaps.py                     # mapas de calor imagen+texto
│   └── io_utils.py                         # cargar configs YAML
├── configs/
│   └── roco_isa_pubmedclip.yaml            # config de ejemplo
├── README.md
├── pyproject.toml                          # instalación editable
├── .gitignore
└── requirements.txt                        # opcional (si necesitas fijar versiones)
```

---

## Instalación (editable) con `pyproject.toml`

En **Colab** o local, tras clonar el repo:

```bash
REPO_URL  = "https://github.com/Alberto-97sc/mmshap_medclip.git"
LOCAL_DIR = "/content/mmshap_medclip"
BRANCH    = "main"

%cd /content
import os, shutil, subprocess, sys

if not os.path.isdir(f"{LOCAL_DIR}/.git"):
    # No está clonado aún
    !git clone $REPO_URL $LOCAL_DIR
else:
    # Ya existe: actualiza a la última versión del remoto
    %cd $LOCAL_DIR
    !git fetch origin
    !git checkout $BRANCH
    !git reset --hard origin/$BRANCH
%cd $LOCAL_DIR
!git rev-parse --short HEAD

```

```bash
# === Instalar en modo editable (pyproject.toml) ===
%pip install -e /content/mmshap_medclip

```


- `-e` instala el paquete en **modo editable**: puedes hacer `from mmshap_medclip...` y cualquier cambio en `src/` se refleja sin reinstalar.
- Las **dependencias** se resuelven automáticamente desde `pyproject.toml` (`[project].dependencies`).

> Si además prefieres un `requirements.txt` con versiones fijas, mantenlo en la raíz y ejecútalo **antes** o **después** de `-e` según tu flujo.

---

## Quickstart (Colab)

1) **Montar Google Drive** (para leer ROCO desde ZIP)
```python
from google.colab import drive; drive.mount('/content/drive')
```

2) **Imports y carga de config/dataset/modelo**
```python

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

```
3) **Imprimir muestra**

```python
from mmshap_medclip.tasks.isa import run_isa_one

muestra = 266
sample  = dataset[muestra]
image, caption = sample['image'], sample['text']

res = run_isa_one(model, image, caption, device, explain=True, plot=True)
print(f"logit={res['logit']:.4f}  TScore={res['tscore']:.2%}  IScore={res['iscore']:.2%}")
# Si quieres la figura:
# display(res['fig'])

```



## Config de ejemplo (`configs/roco_isa_pubmedclip.yaml`)

```yaml
experiment_name: demo_roco_sample
device: auto

dataset:
  name: roco
  params:
    zip_path: /content/drive/MyDrive/MAESTRIA-TESIS/datasets/ROCO/dataset_roco.zip
    split: validation
    n_rows: all
    columns:
      image_key: name
      caption_key: caption
      images_subdir: all_data/validation/radiology/images

model:
  name: pubmedclip-vit-b32
  params: {}
```

---

## Notas y consejos
- **CUDA**: activa GPU en Colab para acelerar; `get_device()` la detecta solo.
- **AMP**: el `Predictor` usa `autocast` en CUDA; desactívalo con `use_amp=False` si ves warnings.
- **`patch_size`**: se infiere de `model.config.vision_config.patch_size`; pásalo manual si tu wrapper no lo expone.
- **Batch B>1**: todas las utilidades (masker/predictor/concat) están vectorizadas para B≥1.

---

## Troubleshooting
- **`ModuleNotFoundError: mmshap_medclip`** → asegúrate de haber corrido `pip install -e .` en la raíz del repo.
- **`patch_size` no detectado** → `Predictor(..., patch_size=32)`.
- **SHAP muy lento** → trabaja con B=1 y/o menos muestras; SHAP es costoso por diseño.

---

## Licencia
MIT

## Créditos
Proyecto de tesis: **Medición del balance multimodal con SHAP en CLIP médico**.
