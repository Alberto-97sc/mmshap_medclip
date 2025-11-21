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
