# mmshap_medclip

Pipeline modular para medir el **balance multimodal** con **SHAP** en modelos tipo **CLIP** (incluye PubMedCLIP, WhyXrayCLIP, Rclip y BiomedCLIP) sobre datasets médicos como **ROCO** (Image-Sentence Alignment) y **VQA-Med 2019** (Visual Question Answering). Diseñado para **ejecución local** con datasets descargados desde **Google Drive** u orígenes oficiales.

> 🚀 **Instalación en un solo click**: Ejecuta `./setup.sh` y tendrás todo listo automáticamente. Ver [Instalación Rápida](#-instalación-rápida-un-solo-click).

> Esta versión utiliza **instalación con `pyproject.toml`** y uso de **`pip install -e .`**.

---

## 📋 Tabla de Contenidos

- [Instalación Rápida (Un Solo Click)](#-instalación-rápida-un-solo-click)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Experimentos disponibles](#experimentos-disponibles)
- [Herramientas de comparación y análisis](#-herramientas-de-comparación-y-análisis)
- [Resultados y dashboards incluidos](#-resultados-y-dashboards-incluidos)
- [Instalación Manual](#instalación-manual)
- [Descarga de datasets (ROCO y VQA-Med)](#descarga-de-datasets-roco-y-vqa-med)
- [Conversión de scripts a notebooks](#conversión-de-scripts-a-notebooks)
- [Uso rápido](#uso-rápido)
- [Configuración de ejemplo](#configuración-de-ejemplo)

---

## ⚡ Instalación Rápida (Un Solo Click)

### 🎯 Opción Recomendada: Script Automático

Si quieres configurar **todo el entorno en un solo comando**, usa el script de instalación automática:

```bash
git clone https://github.com/Alberto-97sc/mmshap_medclip.git
cd mmshap_medclip
./setup.sh
```

### ✨ ¿Qué hace el script automático?

```
┌─────────────────────────────────────────────────────────┐
│                     ./setup.sh                          │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  [Verificar Python]  [Configurar Git]  [Instalar deps]
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                   [Descargar dataset]
                            │
                            ▼
              [Convertir scripts a notebooks]
                            │
                            ▼
                     ✅ ¡LISTO!
```

El script `setup.sh` automatiza completamente la configuración del proyecto en **5 pasos**:

1. **🐍 Verifica e instala Python3** (si no está presente en el sistema)
   - Detecta automáticamente el sistema operativo (Debian/Ubuntu/RedHat/CentOS)
   - Instala Python3 y pip usando el gestor de paquetes apropiado
   - Muestra la versión de Python instalada

2. **📝 Configura Git** con las credenciales del proyecto
   - `user.name`: Alberto-97sc
   - `user.email`: alberthg.ramos@gmail.com

3. **📦 Instala todas las dependencias**
   - Actualiza pip a la última versión
   - Instala el paquete en modo editable (`pip install -e .`)
   - Incluye soporte para Jupyter notebooks (jupytext, jupyter)
   - Instala todas las librerías necesarias (SHAP, transformers, torch, etc.)

4. **📥 Descarga automática de datasets médicos**
   - Ejecuta `scripts/download_dataset.py` para obtener ROCO en `data/dataset_roco.zip`
   - Ejecuta `scripts/download_vqa_med_2019.py` para guardar `data/VQA-Med-2019.zip` (solo si cuentas con permisos de ImageCLEF)
   - Antes de reusar archivos existentes pregunta si deseas volver a descargarlos

5. **📓 Convierte scripts a notebooks** Jupyter
   - Genera archivos `.ipynb` en el directorio `experiments/`
   - Crea notebooks listos para usar en Jupyter

### 📺 Salida del script

Cuando ejecutes `./setup.sh`, verás algo similar a esto:

```
╔════════════════════════════════════════════════════════════════╗
║   Inicializando proyecto mmshap_medclip                        ║
╚════════════════════════════════════════════════════════════════╝

🐍 [1/5] Verificando instalación de Python...
   ✅ Python ya está instalado (versión 3.12.12)

📝 [2/5] Configurando Git...
   ✅ Git configurado correctamente
      Usuario: Alberto-97sc
      Email: alberthg.ramos@gmail.com

📦 [3/5] Instalando dependencias...
   → Actualizando pip...
   → Instalando mmshap_medclip con soporte para notebooks...
   ✅ Dependencias instaladas correctamente
      ✓ Paquete mmshap_medclip en modo editable
      ✓ Dependencias para notebooks (jupytext, jupyter)

📥 [4/5] Descargando datasets desde Google Drive...
   📥 [4.1/5] Dataset ROCO → data/dataset_roco.zip
      ✅ Descarga completada
   📥 [4.2/5] Dataset VQA-Med 2019 → data/VQA-Med-2019.zip
      ✅ Descarga completada (o se reutiliza el archivo existente)

📓 [5/5] Convirtiendo scripts a notebooks Jupyter...
   ✅ Notebooks creados en experiments/
      - experiments/pubmedclip_roco_isa.ipynb
      - experiments/whyxrayclip_roco_isa.ipynb
      - experiments/rclip_roco_isa.ipynb
      - experiments/biomedclip_roco_isa.ipynb

╔════════════════════════════════════════════════════════════════╗
║   ✅ INSTALACIÓN COMPLETADA EXITOSAMENTE                       ║
╚════════════════════════════════════════════════════════════════╝
```

### 🚀 Después de ejecutar el script

Una vez completada la instalación, solo necesitas:

```bash
# Ejecutar un experimento directamente
python3 experiments/pubmedclip_roco_isa.py
python3 experiments/whyxrayclip_roco_isa.py
python3 experiments/rclip_roco_isa.py
```

O usar los notebooks generados:

```bash
# Iniciar Jupyter Notebook
jupyter notebook

# Luego abrir: experiments/pubmedclip_roco_isa.ipynb
# Seleccionar cualquier kernel de Python 3.12
```

### 📋 Requisitos previos

- **Sistema operativo**: Linux (Debian/Ubuntu/RedHat/CentOS) o Mac
- **Permisos**: Puede requerir `sudo` si Python no está instalado
- **Conexión a internet**: Para descargar dependencias y dataset

### 🔧 Personalización del script

Si deseas modificar la configuración de Git, edita las siguientes líneas en `setup.sh`:

```bash
git config user.name "TuUsuario"
git config user.email "tu.email@example.com"
```

### ⚠️ Solución de problemas

**Error: Permission denied al ejecutar ./setup.sh**
```bash
# Dar permisos de ejecución al script
chmod +x setup.sh
./setup.sh
```

**Error: Python no se instaló automáticamente**
- El script requiere `sudo` para instalar Python
- Asegúrate de tener permisos de administrador
- Alternativamente, instala Python manualmente:
  ```bash
  sudo apt-get install python3 python3-pip  # Debian/Ubuntu
  # o
  sudo yum install python3 python3-pip  # RedHat/CentOS
  ```

**Error al descargar el dataset**
- Verifica tu conexión a internet
- Intenta descargar manualmente desde el [enlace de Google Drive](https://drive.google.com/file/d/1eRUC8F8PtXffa9iArJnyB8AMqlPNoSwc/view?usp=sharing)
- Coloca el archivo en `data/dataset_roco.zip`

**Error de importación en notebooks**
- Asegúrate de seleccionar un kernel de Python 3.12
- El paquete se instala directamente en el sistema Python
- No necesitas activar ningún entorno virtual

---

## 📁 Estructura del repositorio

```
mmshap_medclip/

├── src/mmshap_medclip/
│   ├── __init__.py
│   ├── comparison.py                      # compara 4 modelos ISA + batch SHAP
│   ├── comparison_vqa.py                  # comparador PubMedCLIP vs BioMedCLIP en VQA-Med
│   ├── devices.py                         # manejo de device (CUDA/CPU)
│   ├── registry.py                        # registro de modelos y datasets
│   ├── models.py                          # wrappers de CLIP (PubMedCLIP, WhyXrayCLIP, RCLIP, BioMedCLIP)
│   ├── io_utils.py                        # cargar configs YAML
│   ├── metrics.py                         # MM-score, IScore
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── base.py                        # interfaz DatasetBase
│   │   ├── roco.py                        # loader ROCO (lee ZIP local)
│   │   └── vqa_med_2019.py                # loader VQA-Med 2019 (splits Training/Val/Test)
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── isa.py                         # tarea Image-Sentence Alignment
│   │   ├── vqa.py                         # tarea VQA + SHAP + visualizaciones
│   │   ├── utils.py                       # prepare_batch, token lengths, etc.
│   │   └── whyxrayclip.py                 # utilidades específicas WhyXrayCLIP
│   ├── shap_tools/
│   │   ├── masker.py                      # build_masker (BOS/EOS safe)
│   │   ├── predictor.py                   # predictor ISA
│   │   └── vqa_predictor.py               # predictor especializado para VQA
│   └── vis/
│       └── heatmaps.py                    # mapas de calor imagen+texto
├── experiments/
│   ├── analyze_batch_results.py           # dashboard estadístico desde batch_shap_results.csv
│   ├── biomedclip_roco_isa.py
│   ├── compare_all_models.py              # comparador 4 modelos ISA (script/notebook)
│   ├── compare_vqa_models.py              # comparador VQA
│   ├── pubmedclip_roco_isa.py
│   ├── rclip_roco_isa.py
│   ├── README_compare_models.md           # guía detallada del comparador
│   └── whyxrayclip_roco_isa.py
├── outputs/
│   ├── analysis/                          # dashboards (tabla_iscore_promedio.png, etc.)
│   ├── isa_heatmaps/                      # >100 heatmaps ISA (png)
│   └── vqa_heatmaps/                      # 80+ heatmaps VQA
├── configs/
│   ├── roco_isa_pubmedclip.yaml
│   ├── roco_isa_whyxrayclip.yaml
│   ├── roco_isa_rclip.yaml
│   ├── roco_isa_biomedclip.yaml
│   └── vqa_med_2019_pubmedclip.yaml
├── scripts/
│   ├── download_dataset.py                # descarga ROCO
│   └── download_vqa_med_2019.py           # descarga VQA-Med 2019 (ZIP completo)
├── data/                                  # carpeta para datasets (no versionada)
├── documentation_tecnica.md
├── REPORTE_TECNICO_MM_SHAP_MEDCLIP.md
├── setup.sh                               # script de instalación automática (un solo click)
├── pyproject.toml                         # configuración del proyecto y dependencias
├── test_alpha_adjustment.py
└── README.md
```

---

## 🧪 Experimentos disponibles

El directorio `experiments/` contiene scripts completos listos para ejecutar localmente o convertir a notebooks:

### 📊 `pubmedclip_roco_isa.py`
- **Modelo**: PubMedCLIP (ViT-B/32)
- **Dataset**: ROCO (Radiology Objects in COntext)
- **Tarea**: Image-Sentence Alignment (ISA)
- **Configuración**: `configs/roco_isa_pubmedclip.yaml`

### 🩻 `whyxrayclip_roco_isa.py`
- **Modelo**: WhyXrayCLIP
- **Dataset**: ROCO (Radiology Objects in COntext)
- **Tarea**: Image-Sentence Alignment (ISA)
- **Configuración**: `configs/roco_isa_whyxrayclip.yaml`

### 🔬 `rclip_roco_isa.py`
- **Modelo**: Rclip (entrenado en ROCO con radiología)
- **Dataset**: ROCO (Radiology Objects in COntext)
- **Tarea**: Image-Sentence Alignment (ISA)
- **Configuración**: `configs/roco_isa_rclip.yaml`

### 🧬 `biomedclip_roco_isa.py`
- **Modelo**: BiomedCLIP (Microsoft - PubMedBERT + ViT-B/16)
- **Dataset**: ROCO (Radiology Objects in COntext)
- **Tarea**: Image-Sentence Alignment (ISA)
- **Configuración**: `configs/roco_isa_biomedclip.yaml`

### 🧠 `compare_vqa_models.py`
- **Modelos**: PubMedCLIP (ViT-B/32) y BioMedCLIP (ViT-B/16)
- **Dataset**: VQA-Med 2019 (splits *Training/Validation/Test*, categorías C1–C4). El notebook usa por defecto C1–C3 Training.
- **Tarea**: Visual Question Answering (multiple-choice) con explicación SHAP
- **Formato**: disponible en `.py` y `.ipynb` para ejecución directa o notebook
- **Incluye**:
  - Loader dedicado `vqa_med_2019` que arma candidatos por categoría
  - Resumen tabular (predicción, exactitud, TScore/IScore)
  - Visualización comparativa conjunta + heatmaps individuales por modelo
  - Control del grid de parches (PubMedCLIP mantiene 7×7, BioMedCLIP se normaliza a 7×7 para comparación justa)

### 🧮 `compare_all_models.py`
- **Objetivo**: comparar simultáneamente PubMedCLIP, BioMedCLIP, RCLIP y WhyXrayCLIP sobre la misma muestra de ROCO.
- **Basado en**: `src/mmshap_medclip/comparison.py` (usa `load_all_models`, `run_shap_on_all_models`, `plot_comparison_simple` y `save_comparison`).
- **Extras**:
  - Identifica automáticamente el modelo más balanceado según IScore.
  - Permite guardar figuras (`outputs/comparison_sample_XX.png`) y resúmenes JSON.
  - Documentación extendida en `experiments/README_compare_models.md`.

### 📈 `analyze_batch_results.py`
- **Entrada**: `outputs/batch_shap_results.csv` generado con `comparison.batch_shap_analysis`.
- **Salida**: tableros en `outputs/analysis/` (`dashboard_completo.png`, `boxplot_iscore.png`, `balance_score_comparison.png`, tablas CSV, etc.).
- **Análisis incluye**:
  - Estadísticos descriptivos para IScore/TScore/Logit por modelo.
  - Pruebas inferenciales (Shapiro-Wilk, Kruskal-Wallis, Wilcoxon) y correlaciones con la longitud del caption.
  - Visualizaciones listas para presentaciones (ranking de modelos, heatmap de correlaciones, violin/box plots).

**Todos los experimentos incluyen**:
- Carga automática del dataset desde archivo local
- Evaluación de balance multimodal con SHAP
- Generación de visualizaciones (heatmaps)
- Cálculo de métricas (TScore, IScore, MM-Score)

---

## 🧰 Herramientas de comparación y análisis

### Comparador ISA (`src/mmshap_medclip/comparison.py`)
- `load_all_models`, `run_shap_on_all_models` y `plot_comparison_simple` permiten levantar los 4 modelos médicos y generar una figura conjunta (imagen + texto coloreado) con métricas logit/TScore/IScore.
- `plot_individual_heatmaps` y `save_comparison` facilitan guardar PNG/JSON en `outputs/`.
- `analyze_multiple_samples` resume SHAP de varios índices y devuelve un `DataFrame`.
- `batch_shap_analysis` ejecuta SHAP en rangos grandes de muestras, es **idempotente** (retoma donde se quedó, detecta NaN y vuelve a procesar) y escribe `outputs/batch_shap_results.csv`, que luego usa el notebook de análisis.

```python
from mmshap_medclip.comparison import load_all_models, batch_shap_analysis
from mmshap_medclip.devices import get_device
from mmshap_medclip.io_utils import load_config
from mmshap_medclip.registry import build_dataset

device = get_device()
cfg = load_config("configs/roco_isa_pubmedclip.yaml")
dataset = build_dataset(cfg["dataset"])
models = load_all_models(device)

df = batch_shap_analysis(
    models=models,
    dataset=dataset,
    device=device,
    start_idx=0,
    end_idx=500,
    csv_path="outputs/batch_shap_results.csv",
)
```

### Comparador VQA (`src/mmshap_medclip/comparison_vqa.py`)
- `load_vqa_models` levanta PubMedCLIP y BioMedCLIP, respetando las preferencias de visualización de cada wrapper.
- `run_vqa_shap_on_models` usa `tasks.vqa.run_vqa_one` para explicar simultáneamente ambos modelos sobre la misma pregunta (usa los candidatos generados por `datasets.vqa_med_2019`).
- `plot_vqa_comparison` normaliza automáticamente los grids de parches (todos quedan en 7×7) y crea colorbars independientes para texto e imagen.
- El script `experiments/compare_vqa_models.py` y los heatmaps en `outputs/vqa_heatmaps/` se apoyan en este módulo.

### Dashboards estadísticos
- `batch_shap_analysis` + `experiments/analyze_batch_results.py` conforman el flujo de análisis masivo: primero se genera un CSV con los SHAP por modelo, luego el notebook produce tablas y gráficos listos para reportes.
- El notebook guarda tanto CSV (`estadisticas_descriptivas.csv`, `metricas_balance_multimodal.csv`, etc.) como figuras (`dashboard_completo.png`, `ranking_balance_modelos.png`, ...) directamente en `outputs/analysis/`.

---

## 📊 Resultados y dashboards incluidos

- `outputs/analysis/` contiene gráficas y tablas ya renderizadas:
  - `balance_score_comparison.png`, `ranking_balance_modelos.png` → ranking visual del modelo más balanceado.
  - `boxplot_iscore.png`, `violinplot_iscore.png`, `boxplots_iscore_tscore.png`, `violinplots_iscore_tscore.png` → distribución detallada de IScores/TScores.
  - `heatmap_correlaciones.png`, `scatter_iscore_vs_tscore.png`, `scatter_caption_length_vs_iscore.png` → correlaciones entre modelos y con la longitud del caption.
  - `tabla_iscore_promedio.png` y el `dashboard_completo.png` listos para presentaciones.
- `outputs/isa_heatmaps/` agrupa más de 100 PNG con los heatmaps ISA generados por `compare_all_models.py` (uno por modelo y muestra).
- `outputs/vqa_heatmaps/` incluye 88 visualizaciones VQA (PubMedCLIP y BioMedCLIP) con grids normalizados y palabras coloreadas.
- El notebook también escribe CSV (`estadisticas_descriptivas.csv`, `metricas_balance_multimodal.csv`, `analisis_estadisticos_inferenciales.csv`, etc.) dentro de `outputs/analysis/` cada vez que se ejecuta el flujo de análisis.

## 🚀 Instalación Manual

> 💡 **Recomendación**: Si prefieres configurar todo automáticamente, usa el [script de instalación rápida](#-instalación-rápida-un-solo-click) en su lugar.

Esta sección describe cómo instalar manualmente el proyecto paso a paso. Útil si quieres tener más control sobre cada etapa o si el script automático no funciona en tu sistema.

### 1. Clonar el repositorio

```bash
git clone https://github.com/Alberto-97sc/mmshap_medclip.git
cd mmshap_medclip
```

### 2. Instalar dependencias

**Instalación básica**:
```bash
pip install -e .
```

> 💡 Esto instala las dependencias incluyendo `gdown`, pero **NO descarga el dataset**. El dataset se descarga en el paso siguiente.

**Instalación con soporte para notebooks** (recomendado):
```bash
pip install -e ".[notebooks]"
```

**Instalación con herramientas de desarrollo** (opcional):
```bash
pip install -e ".[dev]"
```

**Instalación completa** (notebooks + dev):
```bash
pip install -e ".[notebooks,dev]"
```

> 💡 Poola opción `-e` instala el paquete en modo editable, permitiendo que cualquier cambio en `src/` se refleje inmediatamente sin necesidad de reinstalar.

---

## 📦 Descarga de datasets (ROCO y VQA-Med)

### Dataset ROCO (Image-Sentence Alignment)

### Descargar dataset ROCO desde Google Drive

El repositorio incluye scripts automáticos para descargar el dataset desde Google Drive:

#### Opción 1: Script automático (RECOMENDADA)

```bash
# Descargar dataset usando gdown (más confiable)
python scripts/download_dataset.py
```

Este script:
1. ✅ Crea el directorio `data/` si no existe
2. 📥 Descarga el dataset ROCO desde Google Drive usando `gdown`
3. 📁 Lo guarda en `data/dataset_roco.zip`
4. ✅ Verifica que la descarga sea exitosa

#### Opción 2: Descarga manual

Si el script automático no funciona, puedes descargar manualmente:

1. **Ir al enlace**: [Dataset ROCO en Google Drive](https://drive.google.com/file/d/1eRUC8F8PtXffa9iArJnyB8AMqlPNoSwc/view?usp=sharing)
2. **Hacer clic en "Descargar"**
3. **Mover el archivo** a `data/dataset_roco.zip`

#### Opción 3: Usando gdown directamente

```bash
# Instalar gdown si no está instalado
pip install gdown

# Descargar directamente
gdown 1eRUC8F8PtXffa9iArJnyB8AMqlPNoSwc -O data/dataset_roco.zip
```

### Dataset VQA-Med 2019 (Visual Question Answering)

> ⚠️ El conjunto VQA-Med 2019 requiere registro en ImageCLEF. El script y las instrucciones suponen que ya cuentas con permisos para descargar el ZIP oficial. El loader soporta los splits **Training / Validation / Test** y las categorías **C1–C4** (Modality, Plane, Organ System, Abnormality).

#### Opción 1: Script automático

```bash
python scripts/download_vqa_med_2019.py
```

El script:
1. ✅ Crea `data/` si no existe.
2. 📥 Descarga el ZIP completo `VQA-Med-2019.zip` vía `gdown` (no lo descomprime).
3. 📦 Conserva la estructura oficial; el loader abrirá internamente `ImageClef-2019-VQA-Med-Training.zip` o los sub-zips correspondientes.
4. 🔁 Si el archivo ya existe, pregunta si deseas sobrescribirlo.

#### Opción 2: Descarga manual

1. **Solicita el dataset** en la [página oficial de ImageCLEF VQA-Med](https://www.imageclef.org/VQA/2019). Descarga el archivo `ImageClef-2019-VQA-Med-Training.zip` o el paquete completo `VQA-Med-2019.zip`.
2. **Coloca el ZIP sin descomprimir** en `data/`. Se soportan ambas rutas:
   - `data/ImageClef-2019-VQA-Med-Training.zip`
   - `data/VQA-Med-2019.zip` (el loader abrirá automáticamente el ZIP interno correcto para el split seleccionado)
3. **Estructura esperada** dentro del ZIP de Training:
   ```
   ImageClef-2019-VQA-Med-Training/
     ├── QAPairsByCategory/
     │   ├── C1_Modality_train.txt
     │   ├── C2_Plane_train.txt
     │   ├── C3_Organ_train.txt
     │   └── C4_Abnormality_train.txt
     └── Train_images/
         ├── xxx.jpg
         └── ...
   ```
4. **Configura el experimento** apuntando al ZIP adecuado. Ejemplo mínimo (`configs/vqa_med_2019_pubmedclip.yaml`):
   ```yaml
   dataset:
     name: vqa_med_2019
     params:
       zip_path: data/ImageClef-2019-VQA-Med-Training.zip   # o data/VQA-Med-2019.zip
       split: Training
       images_subdir: Train_images
       n_rows: all
   ```
5. **Verificación**: al cargar el dataset verás mensajes como:
   ```
   📊 Split seleccionado: TRAINING
   📁 Archivos a leer para split TRAINING: ['C1_Modality_train.txt', ...]
   📊 Construyendo candidatos por categoría...
   ```
   Si aparece un error sobre candidatos vacíos o rutas inválidas, revisa que los archivos C1–C4 estén dentro de `QAPairsByCategory/` y que las imágenes residan en la carpeta correcta (`Train_images`, `Val_images` o `VQAMed2019_Test_Images` según el split).

---

## 📓 Conversión de scripts a notebooks

Los scripts en `experiments/` están en formato Jupytext (`.py`), lo que permite versionarlos fácilmente y convertirlos a notebooks Jupyter.

### Convertir un script a notebook

```bash
# Convertir un script específico (ejemplo con PubMedCLIP)
jupytext --to notebook experiments/pubmedclip_roco_isa.py

# Convertir otro script (ejemplo con Rclip)
jupytext --to notebook experiments/rclip_roco_isa.py

# Convertir todos los scripts
jupytext --to notebook experiments/*.py
```

Esto generará archivos `.ipynb` que puedes abrir con Jupyter Notebook o JupyterLab.

### Actualizar notebook desde script modificado

```bash
# Actualizar un notebook específico
jupytext --sync experiments/pubmedclip_roco_isa.py
jupytext --sync experiments/rclip_roco_isa.py

# O actualizar todos
jupytext --sync experiments/*.py
```

### Convertir notebook de vuelta a script

```bash
# Convertir un notebook específico de vuelta a script
jupytext --to py:percent experiments/pubmedclip_roco_isa.ipynb
jupytext --to py:percent experiments/rclip_roco_isa.ipynb
```

---

## 🎯 Uso rápido

### Opción 1: Instalación automática + ejecución (RECOMENDADA)

```bash
# 1. Clonar y configurar todo automáticamente
git clone https://github.com/Alberto-97sc/mmshap_medclip.git
cd mmshap_medclip
./setup.sh

# 2. Ejecutar cualquier experimento directamente
python3 experiments/pubmedclip_roco_isa.py
python3 experiments/whyxrayclip_roco_isa.py
python3 experiments/rclip_roco_isa.py
python3 experiments/biomedclip_roco_isa.py
```

### Opción 2: Ejecutar scripts directamente (manual)

```bash
# 1. Descargar datasets necesarios
python3 scripts/download_dataset.py
# (opcional) python3 scripts/download_vqa_med_2019.py  # solo si usarás VQA-Med

# 2. Ejecutar experimento con PubMedCLIP
python3 experiments/pubmedclip_roco_isa.py

# 3. Ejecutar experimento con WhyXrayCLIP
python3 experiments/whyxrayclip_roco_isa.py

# 4. Ejecutar experimento con Rclip
python3 experiments/rclip_roco_isa.py

# 5. Ejecutar experimento con BiomedCLIP
python3 experiments/biomedclip_roco_isa.py
```

### Opción 3: Usar notebooks

```bash
# Si usaste setup.sh, los notebooks ya están creados:
jupyter notebook
# Abrir cualquiera de los notebooks disponibles:
# - experiments/pubmedclip_roco_isa.ipynb
# - experiments/whyxrayclip_roco_isa.ipynb
# - experiments/rclip_roco_isa.ipynb
# - experiments/biomedclip_roco_isa.ipynb
# Seleccionar cualquier kernel de Python 3.12

# Si instalaste manualmente, convierte primero:
jupytext --to notebook experiments/*.py
jupyter notebook
```

### Opción 4: Comparar todos los modelos simultáneamente 🆕

**Nuevo:** Ahora puedes comparar los 4 modelos en la misma muestra con un solo script:

```bash
# Ejecutar comparación
python3 experiments/compare_all_models.py

# O como notebook
jupytext --to notebook experiments/compare_all_models.py
jupyter notebook experiments/compare_all_models.ipynb
```

**Características:**
- ✅ Carga los 4 modelos automáticamente
- ✅ Ejecuta SHAP en todos con la misma muestra
- ✅ Visualización comparativa en grid 2x2 con:
  - Heatmap de imagen (overlay SHAP)
  - Heatmap de texto (palabras coloreadas)
- ✅ Resumen de métricas (Logit, TScore, IScore)
- ✅ Identifica el modelo más balanceado

**Ejemplo de uso programático:**

```python
from mmshap_medclip.comparison import (
    load_all_models,
    run_shap_on_all_models,
    plot_comparison_simple,
    print_summary
)
from mmshap_medclip.devices import get_device
from mmshap_medclip.io_utils import load_config
from mmshap_medclip.registry import build_dataset

# Setup
device = get_device()
cfg = load_config("configs/roco_isa_pubmedclip.yaml")
dataset = build_dataset(cfg["dataset"])

# Cargar los 4 modelos
models = load_all_models(device)

# Analizar muestra 154
results, image, caption = run_shap_on_all_models(
    models, sample_idx=154, dataset=dataset, device=device
)

# Visualizar comparación
fig = plot_comparison_simple(results, image, caption, sample_idx=154)
plt.show()

# Imprimir resumen
print_summary(results)
```

Ver documentación completa en: [`experiments/README_compare_models.md`](experiments/README_compare_models.md)

---

### Opción 5: Analizar VQA-Med 2019 (PubMedCLIP vs BioMedCLIP) 🆕

```bash
# 1. Asegúrate de tener data/VQA-Med-2019.zip o ImageClef-2019-VQA-Med-Training.zip

# 2. Ejecuta el comparador VQA (script o notebook)
python3 experiments/compare_vqa_models.py
# o
jupytext --to notebook experiments/compare_vqa_models.py
jupyter notebook experiments/compare_vqa_models.ipynb
```

**Qué hace este flujo:**
- Carga el dataset `vqa_med_2019` (solo split Training, categorías C1–C3)
- Inicializa PubMedCLIP y BioMedCLIP con preferencias de visualización personalizadas
- Ejecuta SHAP para ambos modelos sobre la misma pregunta-imagen
- Muestra:
  - Tabla comparativa con predicción, exactitud y balance multimodal
  - Figura conjunta con imagen + pregunta
  - Heatmaps individuales en los que PubMedCLIP preserva su grid 7×7 y BioMedCLIP se normaliza al mismo número de parches para comparación justa
- Permite guardar resultados en `outputs/vqa/` y analizar múltiples índices en batch

> Consejo: modifica `dataset_params` y `MUESTRA_A_ANALIZAR` directamente en el notebook/script para apuntar a otra ruta de dataset o a otra muestra específica.

---

### Opción 6: Uso programático paso a paso

Ejemplo con PubMedCLIP:

```python
from mmshap_medclip.io_utils import load_config
from mmshap_medclip.devices import get_device
from mmshap_medclip.registry import build_dataset, build_model
from mmshap_medclip.tasks.isa import run_isa_one

# Cargar configuración
cfg = load_config("configs/roco_isa_pubmedclip.yaml")

# Obtener device (CUDA si está disponible)
device = get_device()

# Cargar dataset y modelo
dataset = build_dataset(cfg["dataset"])
model = build_model(cfg["model"], device=device)

print(f"Dataset cargado: {len(dataset)} muestras")
print(f"Device: {device}")

# Ejecutar evaluación en una muestra
sample = dataset[154]
image, caption = sample['image'], sample['text']

res = run_isa_one(model, image, caption, device, explain=True, plot=True)
print(f"logit={res['logit']:.4f}  TScore={res['tscore']:.2%}  IScore={res['iscore']:.2%}")
```

Ejemplo con Rclip (similar para WhyXrayCLIP):

```python
from mmshap_medclip.io_utils import load_config
from mmshap_medclip.devices import get_device
from mmshap_medclip.registry import build_dataset, build_model
from mmshap_medclip.tasks.isa import run_isa_one

# Cargar configuración de Rclip
cfg = load_config("configs/roco_isa_rclip.yaml")

# Obtener device (CUDA si está disponible)
device = get_device()

# Cargar dataset y modelo
dataset = build_dataset(cfg["dataset"])
model = build_model(cfg["model"], device=device)

print(f"Dataset cargado: {len(dataset)} muestras")
print(f"Device: {device}")

# Ejecutar evaluación en una muestra
sample = dataset[154]
image, caption = sample['image'], sample['text']

res = run_isa_one(model, image, caption, device, explain=True, plot=True)
print(f"logit={res['logit']:.4f}  TScore={res['tscore']:.2%}  IScore={res['iscore']:.2%}")
```

---

## ⚙️ Configuración de ejemplo

### `configs/roco_isa_pubmedclip.yaml`

```yaml
experiment_name: demo_roco_sample
device: auto

dataset:
  name: roco
  params:
    zip_path: data/dataset_roco.zip
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

### `configs/roco_isa_whyxrayclip.yaml`

```yaml
experiment_name: demo_roco_whyxrayclip
device: auto

dataset:
  name: roco
  params:
    zip_path: data/dataset_roco.zip
    split: validation
    n_rows: all
    columns:
      image_key: name
      caption_key: caption
      images_subdir: all_data/validation/radiology/images

model:
  name: whyxrayclip
  params:
    model_name: hf-hub:yyupenn/whyxrayclip
    tokenizer_name: ViT-L-14
```

### `configs/roco_isa_rclip.yaml`

```yaml
experiment_name: demo_roco_rclip
device: auto

dataset:
  name: roco
  params:
    zip_path: data/dataset_roco.zip
    split: validation
    n_rows: all
    columns:
      image_key: name
      caption_key: caption
      images_subdir: all_data/validation/radiology/images

model:
  name: rclip
  params:
    model_name: kaveh/rclip
```

### `configs/roco_isa_biomedclip.yaml`

```yaml
experiment_name: demo_roco_biomedclip
device: auto

dataset:
  name: roco
  params:
    zip_path: data/dataset_roco.zip
    split: validation
    n_rows: all
    columns:
      image_key: name
      caption_key: caption
      images_subdir: all_data/validation/radiology/images

model:
  name: biomedclip
  params:
    model_name: hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
```

### `configs/vqa_med_2019_pubmedclip.yaml`

```yaml
experiment_name: vqa_med_2019_pubmedclip
device: auto

dataset:
  name: vqa_med_2019
  params:
    zip_path: data/ImageClef-2019-VQA-Med-Training.zip   # o data/VQA-Med-2019.zip
    split: Training
    images_subdir: Train_images
    n_rows: all

model:
  name: pubmedclip-vit-b32
  params: {}
```

---

## 📄 Licencia

MIT

## 👨‍💻 Autor

Proyecto de tesis: **Medición del balance multimodal con SHAP en CLIP médico**
