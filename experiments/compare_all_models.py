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
# # 🔬 Comparación de los 4 Modelos CLIP Médicos
#
# Este notebook permite ejecutar SHAP en una misma muestra con los 4 modelos y visualizar
# los resultados lado a lado para comparar su comportamiento.
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
# ## 🚀 Ejecutar comparación en una muestra

# %%
from mmshap_medclip.comparison import run_shap_on_all_models, plot_comparison_simple

# 🎯 CONFIGURACIÓN: Cambiar este número para probar diferentes muestras
MUESTRA_A_ANALIZAR = 154

print("="*80)
print("🚀 INICIANDO ANÁLISIS COMPARATIVO")
print("="*80)

# Ejecutar SHAP en todos los modelos
results, image, caption = run_shap_on_all_models(
    models=loaded_models,
    sample_idx=MUESTRA_A_ANALIZAR,
    dataset=dataset,
    device=device,
    verbose=True
)

# Visualizar comparación
print("\n📊 Generando visualización comparativa...")
fig = plot_comparison_simple(results, image, caption, MUESTRA_A_ANALIZAR)
if fig is not None:
    import matplotlib.pyplot as plt
    plt.show()
    print("✅ Visualización completada")
else:
    print("❌ No se pudo generar la visualización")

# %% [markdown]
# ## 📈 Resumen de resultados

# %%
from mmshap_medclip.comparison import print_summary

print_summary(results)

# %% [markdown]
# ## 🔍 Visualizar heatmaps individuales detallados
#
# Si deseas ver los heatmaps completos con las palabras coloreadas para cada modelo,
# descomenta y ejecuta la siguiente celda.

# %%
from mmshap_medclip.comparison import plot_individual_heatmaps

# Descomentar para ver heatmaps individuales detallados
# plot_individual_heatmaps(results, image, caption)

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
#    Simplemente ejecuta la celda de análisis de nuevo con el nuevo número de muestra.
#
# 3. **Ver resultados:**
#    - Visualización comparativa en grid 2x2
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
# - `plot_comparison_simple()`: Visualización comparativa rápida
# - `plot_individual_heatmaps()`: Heatmaps detallados individuales
# - `print_summary()`: Imprime resumen comparativo
# - `save_comparison()`: Guarda resultados en disco
# - `analyze_multiple_samples()`: Análisis batch de múltiples muestras
#
# ---
#
# **Proyecto de tesis sobre balance multimodal en modelos CLIP médicos**

