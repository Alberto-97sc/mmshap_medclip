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

# %% [markdown]
# # 🔬 Análisis SHAP de los 4 Modelos CLIP Médicos
#
# Este notebook permite ejecutar SHAP en una misma muestra con los 4 modelos y generar
# heatmaps individuales detallados para cada uno, permitiendo comparar su comportamiento.
#
# **Modelos:**
# - PubMedCLIP
# - BioMedCLIP
# - RCLIP
# - WhyXRayCLIP
#
# **Dataset:** ROCO (Radiology Objects in COntext)
#
# **Tarea:** ISA (Image-Sentence Alignment)
#
# **Visualización:** Heatmaps individuales con imagen y texto por cada modelo
#
# ---

# %% [markdown]
# ## 📦 Configuración inicial

# %%
import os
from pathlib import Path

# 📌 Configuración - Asegurar que estamos en el directorio correcto
try:
    # En scripts Python
    PROJECT_ROOT = Path(__file__).parent.parent
except NameError:
    # En notebooks de Jupyter
    PROJECT_ROOT = Path.cwd()
    # Si estamos en experiments/, subir un nivel
    if PROJECT_ROOT.name == "experiments":
        PROJECT_ROOT = PROJECT_ROOT.parent

os.chdir(PROJECT_ROOT)
print(f"📂 Directorio de trabajo: {PROJECT_ROOT}")

# %% [markdown]
# ## 🎯 Cargar dataset y dispositivo

# %%
from mmshap_medclip.io_utils import load_config
from mmshap_medclip.devices import get_device
from mmshap_medclip.registry import build_dataset

print("🔄 Cargando configuración y dataset...")
cfg = load_config("configs/roco_isa_pubmedclip.yaml")

ROCO_SPLIT_MANUAL = None  # "train", "validation" o "test". Usa None para valor por defecto/entorno.

roco_split_aliases = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}

default_roco_split = (
    cfg.get("dataset", {})
       .get("params", {})
       .get("split", "validation")
       .lower()
)
roco_split_env = os.environ.get("ROCO_SPLIT")
roco_split_source = ROCO_SPLIT_MANUAL or roco_split_env or default_roco_split
roco_split_raw = roco_split_source.strip().lower()
if roco_split_raw not in roco_split_aliases:
    raise ValueError(f"Split ROCO '{roco_split_raw}' no soportado. Usa train, validation o test.")

roco_split = roco_split_aliases[roco_split_raw]
roco_image_subdirs = {
    "train": "all_data/training/radiology/images",
    "validation": "all_data/validation/radiology/images",
    "test": "all_data/test/radiology/images",
}

cfg["dataset"]["params"]["split"] = roco_split
cfg["dataset"]["params"]["images_subdir"] = roco_image_subdirs.get(roco_split)
cfg["dataset"]["params"].setdefault("columns", {}).pop("images_subdir", None)
print(f"📁 ROCO split seleccionado: {roco_split.upper()}")

device = get_device()
dataset = build_dataset(cfg["dataset"])

print(f"✅ Dataset cargado: {len(dataset)} muestras")
print(f"💻 Dispositivo: {device}")

# %% [markdown]
# ## 🤖 Cargar los 4 modelos

# %%
from mmshap_medclip.comparison import load_all_models

models = load_all_models(device)

# Filtrar solo los modelos que se cargaron correctamente
loaded_models = {k: v for k, v in models.items() if v is not None}
print(f"\n📊 Modelos cargados: {len(loaded_models)}/{len(models)}")

# %% [markdown]
# ## 🚀 Ejecutar SHAP y visualizar resultados
#
# Este bloque ejecuta SHAP en todos los modelos y muestra los heatmaps.

# %%
from mmshap_medclip.comparison import (
    run_shap_on_all_models,
    print_summary,
    plot_individual_heatmaps
)

# 🎯 CONFIGURACIÓN: Cambiar este número para probar diferentes muestras
MUESTRA_A_ANALIZAR = 154

# Ejecutar SHAP en todos los modelos
print("="*80)
print("🚀 INICIANDO ANÁLISIS COMPARATIVO")
print("="*80)

results, image, caption = run_shap_on_all_models(
    models=loaded_models,
    sample_idx=MUESTRA_A_ANALIZAR,
    dataset=dataset,
    device=device,
    verbose=True
)

# Imprimir caption completo antes del resumen
print("\n" + "="*80)
print("📝 Caption original (completo):")
print(caption)
print("="*80 + "\n")

# Mostrar resumen en tabla
print_summary(results)

# Visualizar heatmaps individuales detallados
print("\n" + "="*80)
print("🔍 GENERANDO HEATMAPS INDIVIDUALES DETALLADOS")
print("="*80 + "\n")

plot_individual_heatmaps(results, image, caption)

# %% [markdown]
# ## 🖼️ Exportar heatmaps ISA en lote
#
# Esta sección permite recorrer automáticamente un rango de muestras y guardar los
# 4 heatmaps (uno por modelo) en la carpeta indicada dentro de `outputs/`.
# Puedes relanzar el proceso cuantas veces necesites; opcionalmente puedes
# sobrescribir archivos existentes.

# %%
import re
from typing import Optional

import matplotlib.pyplot as plt
from mmshap_medclip.tasks.isa import plot_isa

# 🎯 CONFIGURACIÓN: ajusta el rango y carpeta destino
HEATMAPS_START_IDX = 150          # Índice inicial (inclusive)
HEATMAPS_END_IDX = 155            # Índice final (inclusive). Usa None para llegar al final del dataset
HEATMAPS_OUTPUT_DIR = "outputs/isa_heatmaps"
HEATMAPS_OVERWRITE = False        # Cambia a True para reemplazar archivos existentes
HEATMAPS_DPI = 200

_slug_pattern = re.compile(r"[^a-z0-9]+")


def _slugify(label: str) -> str:
    slug = _slug_pattern.sub("_", label.lower()).strip("_")
    return slug or "modelo"


def export_isa_heatmaps_batch(
    start_idx: int,
    end_idx: Optional[int],
    output_dir: str,
    overwrite: bool = False,
    dpi: int = 200,
):
    if not loaded_models:
        raise RuntimeError("No hay modelos cargados para generar heatmaps.")

    total_samples = len(dataset)
    if total_samples == 0:
        raise RuntimeError("El dataset está vacío; no hay muestras para procesar.")

    max_idx = total_samples - 1
    if start_idx < 0 or start_idx > max_idx:
        raise ValueError(f"start_idx debe estar entre 0 y {max_idx}.")

    if end_idx is None:
        end_idx = max_idx
    end_idx = min(end_idx, max_idx)
    if end_idx < start_idx:
        raise ValueError("end_idx no puede ser menor que start_idx.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n🖼️ Exportando heatmaps ISA de {start_idx} a {end_idx} (total {end_idx - start_idx + 1} muestras)")

    for sample_idx in range(start_idx, end_idx + 1):
        print(f"\n{'='*80}")
        print(f"📸 Procesando muestra {sample_idx}")
        print(f"{'='*80}")

        try:
            results_batch, image, caption = run_shap_on_all_models(
                models=loaded_models,
                sample_idx=sample_idx,
                dataset=dataset,
                device=device,
                verbose=False
            )
        except Exception as exc:
            print(f"❌ Error obteniendo SHAP para la muestra {sample_idx}: {exc}")
            continue

        for model_name, result in results_batch.items():
            if result is None:
                print(f"⚠️  {model_name} no devolvió resultados; se omite.")
                continue

            fig = None
            try:
                fig = plot_isa(
                    image=image,
                    caption=caption,
                    isa_output=result,
                    display_plot=False
                )

                filename = f"{sample_idx}_{_slugify(model_name)}_isa.png"
                filepath = output_path / filename

                if filepath.exists() and not overwrite:
                    print(f"⏭️  {filepath.name} ya existe. Usa HEATMAPS_OVERWRITE=True para reemplazarlo.")
                    continue

                fig.savefig(filepath, bbox_inches="tight", dpi=dpi)
                print(f"✅ Heatmap guardado: {filepath}")
            except Exception as exc:
                print(f"❌ Error guardando heatmap de {model_name} (muestra {sample_idx}): {exc}")
            finally:
                if fig is not None:
                    plt.close(fig)

    print("\n🎉 Exportación de heatmaps ISA finalizada.")


export_isa_heatmaps_batch(
    start_idx=HEATMAPS_START_IDX,
    end_idx=HEATMAPS_END_IDX,
    output_dir=HEATMAPS_OUTPUT_DIR,
    overwrite=HEATMAPS_OVERWRITE,
    dpi=HEATMAPS_DPI,
)

# %% [markdown]
# ## 💾 Guardar resultados
#
# Descomentar para guardar los resultados en disco.

# %%
from mmshap_medclip.comparison import save_comparison

# Descomentar para guardar
# save_comparison(results, image, caption, MUESTRA_A_ANALIZAR, output_dir="outputs")

# %% [markdown]
# ## 🔬 Análisis de múltiples muestras
#
# Para analizar múltiples muestras y obtener estadísticas agregadas,
# descomentar y ejecutar la siguiente celda.

# %%
from mmshap_medclip.comparison import analyze_multiple_samples

# Ejemplo: analizar 5 muestras
# sample_indices = [10, 50, 100, 154, 200]
# df_results = analyze_multiple_samples(loaded_models, dataset, device, sample_indices)
# print(df_results.head(10))

# %% [markdown]
# ## 🚀 Análisis Batch de SHAP (Sin Heatmaps)
#
# Esta sección permite ejecutar SHAP en múltiples muestras sin generar heatmaps,
# guardando automáticamente los resultados en un CSV. La función está blindada ante
# interrupciones: si se interrumpe la ejecución, puede continuar desde donde se quedó.
#
# **Características:**
# - ✅ Guarda automáticamente después de cada muestra
# - ✅ Salta muestras ya procesadas
# - ✅ Continúa automáticamente desde donde se quedó
# - ✅ Guarda: sample_idx, Iscore_[modelo], Tscore_[modelo], Logit_[modelo] para cada modelo
# - ✅ Incluye variables adicionales útiles (caption_length, timestamp)
# - ✅ Imprime estado de ejecución en tiempo real

# %%
from mmshap_medclip.comparison import batch_shap_analysis

# 🎯 CONFIGURACIÓN: Ajustar estos valores según necesites
START_IDX = 0          # Índice inicial de la muestra (inclusive)
END_IDX = None          # Índice final de la muestra (exclusive). None = hasta el final del dataset
CSV_PATH = "outputs/batch_shap_results.csv"  # Ruta donde guardar los resultados

# Ejecutar análisis batch
df_batch_results = batch_shap_analysis(
    models=loaded_models,
    dataset=dataset,
    device=device,
    start_idx=START_IDX,
    end_idx=END_IDX,
    csv_path=CSV_PATH,
    verbose=True,
    show_dataframe=True  # Mostrar DataFrame en tiempo real después de cada muestra
)

# Mostrar primeras filas del DataFrame
print("\n📊 Primeras filas del DataFrame de resultados:")
print(df_batch_results.head(10))

# Mostrar estadísticas resumidas
if not df_batch_results.empty:
    print("\n📈 Estadísticas resumidas:")
    print(f"   Total de muestras procesadas: {len(df_batch_results)}")

    # Calcular promedios de IScore por modelo
    model_names = [name for name in loaded_models.keys() if loaded_models[name] is not None]
    print("\n📊 IScore promedio por modelo:")
    for model_name in model_names:
        col_name = f'Iscore_{model_name}'
        if col_name in df_batch_results.columns:
            avg_iscore = df_batch_results[col_name].mean()
            print(f"   {model_name}: {avg_iscore:.2%}")

# %% [markdown]
# ---
#
# ## 📝 Notas de Uso
#
# ### 🎯 Uso Básico
#
# 1. **Cambiar la muestra a analizar:**
#    Modifica la variable `MUESTRA_A_ANALIZAR` en la celda correspondiente.
#
# 2. **Re-ejecutar el análisis:**
#    Simplemente ejecuta las celdas de nuevo con el nuevo número de muestra.
#
# 3. **Ver resultados:**
#    - Heatmaps individuales detallados para cada modelo
#    - Resumen de métricas en tabla
#    - Análisis de balance multimodal
#
# ### 📊 Métricas Explicadas
#
# - **Logit**: Score de similitud imagen-texto del modelo
# - **TScore**: Proporción de importancia asignada al texto (0-100%)
# - **IScore**: Proporción de importancia asignada a la imagen (0-100%)
# - **Balance ideal**: TScore ≈ IScore ≈ 50%
#
# ### 🔬 Funciones Disponibles
#
# - `load_all_models()`: Carga los 4 modelos CLIP médicos
# - `run_shap_on_all_models()`: Ejecuta SHAP en todos los modelos
# - `plot_individual_heatmaps()`: Muestra heatmaps detallados individuales para cada modelo
# - `print_summary()`: Imprime resumen comparativo en tabla
# - `save_comparison()`: Guarda resultados en disco
# - `analyze_multiple_samples()`: Análisis batch de múltiples muestras
#
# ---
#
# **Proyecto de tesis sobre balance multimodal en modelos CLIP médicos**
