# mmshap_medclip

Pipeline modular para medir el **balance multimodal** con **SHAP** en modelos tipo **CLIP** (incluye PubMedCLIP, WhyXrayCLIP, Rclip y BiomedCLIP) sobre datasets médicos (p. ej., ROCO). Diseñado para **ejecución local** con datasets descargados desde **Google Drive**.

> 🚀 **Instalación en un solo click**: Ejecuta `./setup.sh` y tendrás todo listo automáticamente. Ver [Instalación Rápida](#-instalación-rápida-un-solo-click).

> Esta versión utiliza **instalación con `pyproject.toml`** y uso de **`pip install -e .`**.

---

## 📋 Tabla de Contenidos

- [Instalación Rápida (Un Solo Click)](#-instalación-rápida-un-solo-click)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Experimentos disponibles](#experimentos-disponibles)
- [Instalación Manual](#instalación-manual)
- [Descarga del dataset](#descarga-del-dataset)
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
                            ▼
                  [Instalar dependencias]
                            │
                            ▼
                   [Descargar dataset]
                            │
                            ▼
                  [Convertir a notebooks]
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

4. **📥 Descarga el dataset ROCO** desde Google Drive
   - Descarga automáticamente usando `gdown`
   - Lo guarda en `data/dataset_roco.zip`
   - Si ya existe, pregunta si deseas volver a descargarlo

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

📥 [4/5] Descargando dataset ROCO desde Google Drive...
   ✅ Dataset descargado correctamente

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
│   ├── devices.py                          # manejo de device (CUDA/CPU)
│   ├── registry.py                         # registro de modelos y datasets
│   ├── models.py                           # wrappers de CLIP (PubMedCLIP, WhyXrayCLIP, Rclip)
│   ├── io_utils.py                         # cargar configs YAML
│   ├── metrics.py                          # MM-score, IScore
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── base.py                         # interfaz DatasetBase
│   │   └── roco.py                         # loader ROCO (lee ZIP local)
│   ├── tasks/
│   │   ├── __init__.py
│   │   ├── isa.py                          # tarea Image-Sentence Alignment
│   │   ├── utils.py                        # prepare_batch, token lengths, etc.
│   │   └── whyxrayclip.py                  # utilidades específicas WhyXrayCLIP
│   ├── shap_tools/
│   │   ├── masker.py                       # build_masker (BOS/EOS safe)
│   │   └── predictor.py                    # Predictor callable para SHAP
│   └── vis/
│       └── heatmaps.py                     # mapas de calor imagen+texto
├── experiments/
│   ├── pubmedclip_roco_isa.py              # experimento PubMedCLIP + ROCO
│   ├── whyxrayclip_roco_isa.py             # experimento WhyXrayCLIP + ROCO
│   ├── rclip_roco_isa.py                   # experimento Rclip + ROCO
│   └── biomedclip_roco_isa.py              # experimento BiomedCLIP + ROCO
├── configs/
│   ├── roco_isa_pubmedclip.yaml            # config ISA para PubMedCLIP
│   ├── roco_isa_whyxrayclip.yaml           # config ISA para WhyXrayCLIP
│   ├── roco_isa_rclip.yaml                 # config ISA para Rclip
│   └── roco_isa_biomedclip.yaml            # config ISA para BiomedCLIP
├── scripts/
│   └── download_dataset.py                 # script para descargar dataset
├── data/                                    # carpeta para datasets (no versionada)
├── setup.sh                                 # script de instalación automática (un solo click)
├── pyproject.toml                          # configuración del proyecto y dependencias
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

**Todos los experimentos incluyen**:
- Carga automática del dataset desde archivo local
- Evaluación de balance multimodal con SHAP
- Generación de visualizaciones (heatmaps)
- Cálculo de métricas (TScore, IScore, MM-Score)

---

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

## 📦 Descarga del dataset

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
# 1. Descargar dataset
python3 scripts/download_dataset.py

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
- ✅ Visualización comparativa en grid 2x2
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

### Opción 5: Uso programático paso a paso

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

---

## 📄 Licencia

MIT

## 👨‍💻 Autor

Proyecto de tesis: **Medición del balance multimodal con SHAP en CLIP médico**
