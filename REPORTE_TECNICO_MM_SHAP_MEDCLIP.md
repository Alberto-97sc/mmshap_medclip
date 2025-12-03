# Informe técnico de `mmshap_medclip`

_Fecha: 2 de diciembre de 2025_

---

## 1. Propósito y alcance

`mmshap_medclip` es una plataforma de investigación que mide el **balance multimodal texto‑imagen** de modelos tipo CLIP adaptados al dominio médico (PubMedCLIP, BioMedCLIP, RCLIP, WhyXRayCLIP, entre otros) sobre datasets radiológicos como **ROCO** (Image‑Sentence Alignment) y **VQA‑Med 2019** (Visual Question Answering). El repositorio empaqueta:

- Carga declarativa de modelos/datasets vía YAML.
- Pipelines ISA/VQA con generación automática de explicaciones SHAP y métricas TScore/IScore.
- Visualizaciones listas para notebooks o scripts batch.
- Scripts de instalación y descarga de datos para ejecución local o en Colab.

---

## 2. Mapa de directorios relevantes

| Ruta | Rol |
| --- | --- |
| `README.md` | Guía de uso, instalación y estructura general. |
| `configs/` | Plantillas YAML para ROCO (ISA) y VQA‑Med, reutilizadas por scripts/notebooks. |
| `data/` | Actualmente versionada con `dataset_roco.zip` y `VQA-Med-2019.zip`, listos para lectura directa sin descomprimir. |
| `experiments/` | Scripts Jupytext (`*.py`) y sus notebooks sincronizados (`*.ipynb`) para ejecutar los pipelines ISA/VQA. |
| `experiments/analyze_batch_results.[py\|ipynb]` | Notebook/CLI para auditar CSVs batch, generar tablas y figuras consolidadas en `outputs/analysis/`. |
| `outputs/analysis/` | Ejemplos de figuras y dashboards generados por el análisis batch. |
| `outputs/*.csv` | Artefactos reanudables: `batch_shap_results_test.csv` e `vqa_batch_shap_results_test.csv` sirven como plantillas. |
| `scripts/` | Descarga automatizada (`download_dataset.py`, `download_vqa_med_2019.py`). |
| `src/mmshap_medclip/` | Paquete instalable con toda la lógica de modelos, datasets, SHAP y visualizaciones. |
| `setup.sh` | Instalación “one click”: verifica Python, configura Git, instala dependencias, descarga datasets y genera notebooks. |
| `documentation_tecnica.md` | Documento previo; el presente informe amplía y actualiza la fotografía del repositorio. |

---

## 3. Flujo funcional end‑to‑end

1. **Configuración**
   - Se selecciona una configuración YAML en `configs/` con rutas de dataset y nombre de modelo.
   - `io_utils.load_config()` la carga y se inicializa el dispositivo vía `devices.get_device()`.

2. **Construcción de componentes**
   - `registry.build_dataset()` y `registry.build_model()` instancian clases registradas mediante decoradores `@register_dataset` y `@register_model`.

3. **Preparación de lotes**
   - `tasks.utils.prepare_batch()` normaliza PIL/captions, aplica el `processor` del wrapper y traslada tensores al device, incorporando truncado seguro para tokenizadores de Hugging Face.

4. **Inferencia + SHAP**
   - `tasks.isa.run_isa_one()` o `tasks.vqa.run_vqa_one()` calculan logits y, si se solicita, ejecutan `_compute_*_shap()` que:
     - Calcula longitudes reales de texto y la rejilla de parches.
     - Genera un `masker` que preserva tokens especiales y un `Predictor`/`VQAPredictor` que enmascara parches antes de llamar al modelo.
     - Ajusta automáticamente `max_evals` del `shap.Explainer` a `2 * (#features) + 1`.

5. **Métricas y visualizaciones**
   - `metrics.compute_mm_score()` y `compute_iscore()` derivan TScore/IScore y desglose palabra→peso.
   - `vis.heatmaps.plot_text_image_heatmaps()` produce una figura combinada texto‑imagen, desnormalizando imágenes y aplicando overlays controlados (alpha configurable, coarsening opcional).

6. **Consumo**
   - Scripts en `experiments/` llaman al pipeline sobre un índice o lote, generan notebooks sincronizables (Jupytext) y pueden usar `comparison.py` para evaluar múltiples modelos en paralelo.
   - El análisis posterior se centraliza en `experiments/analyze_batch_results` (py/notebook), que resume cualquier CSV batch y alimenta las figuras en `outputs/analysis/`.

---

## 4. Componentes del paquete `src/mmshap_medclip`

### 4.1 Núcleo de configuración y registro

- `io_utils.py`: función `load_config(path)` que encapsula `yaml.safe_load`.
- `devices.py`: selección automática de CUDA/CPU y utilitario `move_to_device` que maneja tensores anidados en diccionarios.
- `registry.py`: mantiene los diccionarios `_REG_MODELS` / `_REG_DATASETS`, ofrece decoradores de registro y funciones `build_model`/`build_dataset` que garantizan la importación diferida de submódulos antes de instanciar la fábrica indicada en YAML.

### 4.2 Wrappers y factorías de modelos

Ubicados en `models.py`:

- **`CLIPWrapper`** (modelos Hugging Face) y **`OpenCLIPWrapper`** (modelos `open_clip`): homogenizan la API (`processor`, `tokenizer`, inferencia de `patch_size`, `vision_input_size` y soporte para `logits_per_image`).
- **`RclipWrapper`**: adapta `VisionTextDualEncoderModel` calculando manualmente similitudes y normalizando embeddings.
- Modelos registrados listos: `pubmedclip-vit-b32`, `openai-clip-vit-b32`, `whyxrayclip`, `rclip`, `biomedclip`. Cada fábrica gestiona carga, to(device) y ajustes específicos (por ejemplo, `vqa_heatmap_prefs` para mantener o escalar la rejilla de parches en visualizaciones).

### 4.3 Datasets

- **`RocoDataset` (`datasets/roco.py`)**: lee el CSV correspondiente directamente desde el ZIP, indexa rutas por basename priorizando coincidencias con `/images/` y el split activo, soporta `images_subdir` explícito y devuelve diccionarios con `{image, text, meta}`.
- **`VQAMed2019Dataset` (`datasets/vqa_med_2019.py`)**:
  - Permite usar el ZIP padre (`VQA-Med-2019.zip`) o el hijo (`ImageClef-2019-VQA-Med-Training.zip`) sin descomprimir.
  - Filtra archivos `QAPairsByCategory` para C1–C3 (`*_train.txt`), construye listas de candidatos únicos por categoría y valida que cada muestra conserve al menos un candidato disponible.
  - Implementa estrategias de resolución de imágenes (por `images_subdir`, basename, ID numérico) y rechaza splits diferentes de Training para mantener consistencia.

### 4.4 Utilidades y tareas

- **`tasks/utils.py`**: agrupa funciones reutilizadas por ISA y VQA:
  - `prepare_batch`, `compute_text_token_lengths`, `make_image_token_ids`, `concat_text_image_tokens`.
  - Maneja diferencias entre tokenizadores con longitud fija (OpenCLIP) y dinámica (BERT‑like), evitando que SHAP use padding innecesario.
- **`tasks/isa.py`**: ofrece `run_isa_one`, `run_isa_batch`, `explain_isa` y `plot_isa`.
- **`tasks/vqa.py`**: extiende la lógica a VQA múltiple‑choice, generando prompts combinados “Question: … Answer: …”, seleccionando candidatos, exponiendo `run_vqa_one/batch`, `evaluate_vqa_dataset`, `explain_vqa` y `plot_vqa`.
- **`tasks/whyxrayclip.py`**: herramientas especializadas para filtrar ROCO mediante keywords, muestrear negativos, puntuar alineación y obtener vistas reproducibles (`RocoKeywordSubset`).

### 4.5 Toolchain SHAP y métricas

- **`shap_tools/masker.py`**: crea un masker que respeta tokens especiales BOS/EOS (por defecto IDs 49406/49407) y evita IDs fuera de vocabulario.
- **`shap_tools/predictor.py`**: reconstruye tensores desde la secuencia `[texto | parches]`, aplica enmascaramiento directo sobre `pixel_values`, controla AMP en CUDA e impone validaciones de geometría.
- **`shap_tools/vqa_predictor.py`**: versión específica para VQA que reconstruye el prompt objetivo (respetando `target_logit` “correct” o “predicted”), permite agrupar parches para grids objetivo (ej. 7×7), calcula similitudes contra todos los candidatos y devuelve el logit pertinente.
- **`metrics.py`**: funciones `compute_mm_score` (TScore + desglose por palabra) e `compute_iscore` (complemento visual), con soporte para tokenizadores que generan subpalabras y heurísticas de limpieza para diferentes alfabetos (`Ġ`, `##`, `</w>`, etc.).

### 4.6 Visualizaciones

- **`vis/heatmaps.py`**:
  - Desnormaliza las imágenes con los promedios CLIP (`_CLIP_MEAN/_CLIP_STD`), reescala el grid SHAP a un tamaño objetivo (14×14 por defecto) o mantiene el grid nativo si se solicita.
  - Ajusta la opacidad (`PLOT_ISA_ALPHA_IMG` = 0.40), agrupa parches vecinos (`coarsen_factor`) y posiciona palabras con bounding boxes auto‑centrados.
  - Añade barras de color independientes para texto e imagen, calculando los percentiles adecuados para evitar saturación.

### 4.7 Comparadores y ejecución batch

- **`comparison.py`**:
  - `load_all_models`, `run_shap_on_all_models`, `plot_comparison_simple`, `plot_individual_heatmaps`, `print_summary`, `save_comparison`.
  - Incluye funciones batch (`analyze_multiple_samples`, `batch_shap_analysis`) que guardan CSVs reanudables con métricas por modelo (`Iscore_*`, `Tscore_*`, `Logit_*`).
  - Al generar heatmaps comparativos replica grids de parches pequeños (ej. modelos con patch 32) a 14×14 para que coincidan visualmente con modelos patch 16.
- **`comparison_vqa.py`**: versión especializada para VQA‑Med 2019 con `load_vqa_models`, `run_vqa_shap_on_models`, `plot_vqa_comparison`, `print_vqa_summary` y `batch_vqa_shap_analysis` (persistencia de IScore, TScore, logit y exactitud por modelo).

---

## 5. Experimentos y scripts disponibles

| Script | Objetivo | Entradas clave |
| --- | --- | --- |
| `experiments/pubmedclip_roco_isa.py` | Evalúa PubMedCLIP en ROCO; ejecuta `run_isa_one` sobre un índice configurable. | `configs/roco_isa_pubmedclip.yaml` |
| `experiments/whyxrayclip_roco_isa.py`, `rclip_roco_isa.py`, `biomedclip_roco_isa.py` | Variantes para cada modelo soportado, idéntica interfaz. | Config YAML correspondiente. |
| `experiments/compare_all_models.py` | Carga simultáneamente los cuatro modelos, ejecuta SHAP, imprime resumen y heatmaps individuales; `batch_shap_analysis` persiste en `outputs/batch_shap_results_test.csv` (renombrable). | `configs/roco_isa_pubmedclip.yaml`, `outputs/batch_shap_results_test.csv`. |
| `experiments/compare_vqa_models.py` | Comparación PubMedCLIP vs BioMedCLIP en VQA‑Med 2019, incluye versión batch reanudable que escribe `outputs/vqa_batch_shap_results_test.csv`. | `data/VQA-Med-2019.zip`, parámetros `dataset_params`. |
| `experiments/analyze_batch_results.py` | Notebook/CLI que analiza cualquier CSV batch (ISA o VQA), genera estadísticas, dashboards y tablas en `outputs/analysis/`. | `outputs/batch_shap_results*.csv` o `outputs/vqa_batch_shap_results*.csv`. |
| `experiments/README_compare_models.md` | Documenta el uso del comparador, incluye ejemplos de código y recomendaciones. | — |
| `test_alpha_adjustment.py` | Script rápido para verificar alpha en heatmaps (ejecuta RCLIP + PubMedCLIP). | Configs de ROCO. |

Todos los scripts están en formato Jupytext (`py:percent`), por lo que `jupytext --sync --to notebook` crea/actualiza el `.ipynb` correspondiente sin perder versionado limpio.

---

## 6. Gestión de datos y assets

- **Descarga automática**
  - `scripts/download_dataset.py`: descarga `dataset_roco.zip` (~6.6 GB) con `gdown`, valida tamaño y evita descargar de nuevo salvo confirmación.
  - `scripts/download_vqa_med_2019.py`: descarga `VQA-Med-2019.zip` (ZIP padre completo) para permitir lectura anidada.
  - El repositorio actual ya versiona ambos ZIP dentro de `data/`, facilitando pruebas offline a costa de mayor peso en Git.

- **Configs**
  - `configs/roco_isa_*.yaml`: mismos parámetros de dataset; solo cambia `model.name`/`params`.
  - `configs/vqa_med_2019_pubmedclip.yaml`: ejemplo de configuración VQA; se puede duplicar para otros modelos.

- **Outputs**
  - `outputs/analysis/`: contiene PNGs (balance, heatmaps, dashboards) generados por `experiments/analyze_batch_results`.
  - Los scripts batch guardan CSV/JSON en `outputs/` con métricas por muestra (ej. `batch_shap_results_test.csv`, `vqa_batch_shap_results_test.csv`) más figuras comparativas (`comparison_sample_<idx>.png`).

---

## 7. Dependencias e instalación

- `pyproject.toml` (único, en la raíz) declara las dependencias base: `torch`, `torchvision`, `transformers`, `open-clip-torch`, `shap`, `matplotlib`, `pandas`, `pillow`, `pyyaml`, `tqdm`, `gdown`.
- Extras:
  - `notebooks`: `jupytext`, `notebook`, `ipykernel`.
  - `dev`: `pytest`, `black`, `ruff`.
- `setup.sh` automatiza:
  1. Verificación/instalación de Python 3 + `python3-venv`.
  2. Configuración de Git con el usuario del proyecto.
  3. Instalación editable `pip install -e .`.
  4. Descarga de datasets (ROCO + VQA) usando los scripts de `scripts/`.
  5. Conversión y sincronización de todos los experiments a notebooks (`jupytext --sync --to notebook`).
- Actualmente no se versionan entornos virtuales precreados; se espera que cada colaborador genere su propio `venv` o entorno de Conda antes de ejecutar `pip install -e .`.

---

## 8. Observaciones relevantes del estado actual

1. **Estructura consolidada**
   - El paquete solo vive en `src/mmshap_medclip/`; ya no existe una copia paralela en la raíz, lo que simplifica la elección de “fuente de verdad”.

2. **Datasets versionados**
   - `data/dataset_roco.zip` y `data/VQA-Med-2019.zip` están presentes en Git. Esto agiliza la reproducción, pero incrementa tamaño y puede convenir migrarlos a almacenamiento externo o Git LFS.

3. **Outputs de referencia**
   - Se versionan CSV de ejemplo (`batch_shap_results_test.csv`, `vqa_batch_shap_results_test.csv`) y múltiples PNG en `outputs/analysis/`. Son útiles como baseline, aunque conviene definir una política de rotación para evitar crecimiento descontrolado.

4. **Sincronización Py/IPynb**
   - `experiments/` mantiene cada cuaderno en doble formato (`.py` + `.ipynb`). Cualquier cambio debe sincronizarse con `jupytext --sync` para prevenir divergencias difíciles de revisar.

5. **Procesos batch**
   - `comparison.py`/`comparison_vqa.py` y `analyze_batch_results` dependen de columnas específicas en los CSV; si se agregan nuevos modelos o métricas hay que regenerar todos los archivos derivados para mantener consistencia.

---

## 9. Guía rápida para extender o integrar

### Añadir un nuevo modelo CLIP
1. Implementar un wrapper en `src/mmshap_medclip/models.py` (heredando de `torch.nn.Module`).
2. Decorar la fábrica con `@register_model("nombre-amigable")`.
3. Asegurar que el wrapper exponga `processor`, `tokenizer` y, si es posible, atributos `patch_size`, `vision_input_size`, `vqa_heatmap_prefs`.
4. Crear o actualizar un YAML en `configs/` referenciando el nuevo nombre.

### Añadir un dataset
1. Crear un archivo en `src/mmshap_medclip/datasets/` con la clase `DatasetBase` correspondiente.
2. Registrar la fábrica con `@register_dataset`.
3. Exponer el módulo en `datasets/__init__.py`.
4. Añadir parámetros en un YAML (ruta, columnas, etc.).

### Integrar en un pipeline
1. Cargar configuración y device.
2. Instanciar dataset/modelo con el registro.
3. Llamar a `run_isa_one` / `run_vqa_one` o a las funciones de comparación según el caso.
4. Usar `plot_isa`/`plot_vqa` o las funciones de `comparison.py` para generar visualizaciones y resúmenes.

---

## 10. Próximos pasos sugeridos

- Evaluar mover los ZIP de `data/` a almacenamiento externo (o Git LFS) para mantener liviano el repositorio sin perder reproducibilidad.
- Definir una política de limpieza/rotación para `outputs/analysis/` y los CSV batch, incluyendo scripts que regeneren las figuras bajo demanda.
- Integrar `jupytext --sync` en una tarea de CI/linting para asegurar que los pares `.py`/`.ipynb` permanezcan alineados.
- Añadir pruebas unitarias (el extra `dev` ya incluye `pytest`) para validar nuevas incorporaciones de modelos/datasets.

---

Este informe resume la lógica, componentes y estado actual del repositorio, proporcionando una vista integrada que facilita la incorporación de nuevos colaboradores y la planificación de mejoras futuras.
