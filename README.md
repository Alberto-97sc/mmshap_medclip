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
%cd /content
!git clone https://github.com/<tu_usuario>/mmshap_medclip.git
%cd mmshap_medclip
%pip install -e .
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
from mmshap_medclip.io_utils import load_config
from mmshap_medclip.devices import get_device
from mmshap_medclip.registry import build_dataset, build_model

cfg = load_config('/content/mmshap_medclip/configs/roco_isa_pubmedclip.yaml')
device  = get_device()
dataset = build_dataset(cfg['dataset'])
model   = build_model(cfg['model'], device=device)

muestra = 0
sample  = dataset[muestra]
image, caption = sample['image'], sample['text']
```

3) **Batch + SHAP + métricas**
```python
from mmshap_medclip.tasks.utils import (
    prepare_batch, compute_text_token_lengths, make_image_token_ids, concat_text_image_tokens
)
from mmshap_medclip.shap_tools.masker import build_masker
from mmshap_medclip.shap_tools.predictor import Predictor
from mmshap_medclip.metrics import compute_mm_score, compute_iscore
import shap

inputs, logits = prepare_batch(model, [caption], [image], device=device)

nb_text_tokens_tensor, _ = compute_text_token_lengths(inputs, model.tokenizer)
image_token_ids_expanded, imginfo = make_image_token_ids(inputs, model)
X_clean, _ = concat_text_image_tokens(inputs, image_token_ids_expanded, device=device)

masker = build_masker(nb_text_tokens_tensor, tokenizer=model.tokenizer)
predict_fn = Predictor(model, inputs, patch_size=imginfo['patch_size'], device=device, use_amp=True)

explainer = shap.Explainer(predict_fn, masker, silent=True)
shap_values = explainer(X_clean.cpu())

tscore, word_shap = compute_mm_score(shap_values, model.tokenizer, inputs, i=0)
iscore = compute_iscore(shap_values, inputs, i=0)
print(f'TScore: {tscore:.2%} | IScore: {iscore:.2%}')
```

4) **Mapas de calor**
```python
from mmshap_medclip.vis.heatmaps import plot_text_image_heatmaps
plot_text_image_heatmaps(
    shap_values=shap_values,
    inputs=inputs,
    tokenizer=model.tokenizer,
    images=image,
    texts=[caption],
    mm_scores=[(tscore, word_shap)],
    model_wrapper=model,
)
```

---

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
MIT (o la que prefieras).

## Créditos
Proyecto de tesis: **Medición del balance multimodal con SHAP en CLIP médico**.
