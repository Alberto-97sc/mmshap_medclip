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
# # 🔬 Análisis SHAP de Modelos CLIP Médicos en VQA-Med 2019
#
# Este notebook permite ejecutar SHAP en una misma muestra VQA con PubMedCLIP y BiomedCLIP
# y generar heatmaps individuales detallados para cada uno, permitiendo comparar su comportamiento.
#
# **Modelos:**
# - PubMedCLIP
# - BioMedCLIP
#
# **Dataset:** VQA-Med 2019
#
# **Tarea:** VQA (Visual Question Answering)
#
# **Visualización:** Heatmaps individuales con imagen y pregunta por cada modelo
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
# Nota: Necesitarás crear un archivo de configuración para VQA-Med 2019
# Por ahora, cargamos directamente el dataset
device = get_device()

# Construir dataset VQA-Med 2019 directamente
# Ajusta estos parámetros según tu configuración
dataset_params = {
    "zip_path": "data/VQA-Med-2019.zip",  # Ruta al ZIP padre que contiene Training.zip
    "split": "Training",  # SOLO se soporta "Training" o "train"
    "images_subdir": "Train_images",  # SOLO se soporta "Train_images" para el split Training
    "n_rows": "all"  # o un número para limitar muestras (ej: 100)
}

from mmshap_medclip.registry import build_dataset
dataset = build_dataset({"name": "vqa_med_2019", "params": dataset_params})

print(f"✅ Dataset cargado: {len(dataset)} muestras")
print(f"💻 Dispositivo: {device}")

# %% [markdown]
# ## 🤖 Cargar los modelos VQA (PubMedCLIP y BiomedCLIP)

# %%
from mmshap_medclip.comparison_vqa import load_vqa_models

models = load_vqa_models(device)

# Filtrar solo los modelos que se cargaron correctamente
loaded_models = {k: v for k, v in models.items() if v is not None}
print(f"\n📊 Modelos cargados: {len(loaded_models)}/{len(models)}")

# %% [markdown]
# ## 🚀 Ejecutar SHAP y visualizar resultados
#
# Este bloque ejecuta SHAP en todos los modelos y muestra los heatmaps.

# %%
from mmshap_medclip.comparison_vqa import (
    run_vqa_shap_on_models,
    print_vqa_summary,
    plot_vqa_comparison
)

# 🎯 CONFIGURACIÓN: Cambiar este número para probar diferentes muestras
MUESTRA_A_ANALIZAR = 0

# Ejecutar SHAP en todos los modelos
print("="*80)
print("🚀 INICIANDO ANÁLISIS COMPARATIVO VQA")
print("="*80)

results, image, question, answer, candidates, category = run_vqa_shap_on_models(
    models=loaded_models,
    sample_idx=MUESTRA_A_ANALIZAR,
    dataset=dataset,
    device=device,
    target_logit="correct",  # o "predicted"
    verbose=True
)

# Imprimir información de la muestra
print("\n" + "="*80)
print("📝 Información de la muestra:")
print(f"   Pregunta: {question}")
print(f"   Categoría: {category}")
print(f"   Respuesta correcta: {answer}")
print(f"   Candidatos: {len(candidates)} opciones")
print("="*80 + "\n")

# Mostrar resumen en tabla
print_vqa_summary(results)

# Visualizar comparación
print("\n" + "="*80)
print("🔍 GENERANDO VISUALIZACIÓN COMPARATIVA")
print("="*80 + "\n")

fig = plot_vqa_comparison(
    results, image, question, answer, candidates, MUESTRA_A_ANALIZAR
)
if fig is not None:
    fig.show()

# %% [markdown]
# ## 🔍 Heatmaps individuales detallados

# %%
from mmshap_medclip.tasks.vqa import plot_vqa

# Generar heatmaps individuales para cada modelo
for model_name, result in results.items():
    if result is None:
        continue
    
    print(f"\n{'='*60}")
    print(f"🔍 Heatmap detallado: {model_name}")
    print(f"{'='*60}\n")
    
    try:
        fig = plot_vqa(
            image=image,
            question=question,
            vqa_output=result,
            model_wrapper=result.get("model_wrapper"),
            display_plot=True
        )
        
        # Imprimir métricas
        prediction = result.get('prediction', 'N/A')
        correct = result.get('correct', None)
        tscore = result.get('tscore', 0.0)
        iscore = result.get('iscore', 0.0)
        correct_str = "✅" if correct else "❌" if correct is False else "?"
        print(f"📊 {model_name} - Predicción: {prediction} {correct_str} | TScore: {tscore:.2%} | IScore: {iscore:.2%}\n")
        
    except Exception as e:
        print(f"❌ Error generando heatmap para {model_name}: {e}\n")

# %% [markdown]
# ## 💾 Guardar resultados
#
# Descomentar para guardar los resultados en disco.

# %%
from pathlib import Path
import json

# Descomentar para guardar
# output_dir = Path("outputs/vqa")
# output_dir.mkdir(parents=True, exist_ok=True)
# 
# # Guardar figura
# if fig is not None:
#     fig_path = output_dir / f"vqa_comparison_sample_{MUESTRA_A_ANALIZAR}.png"
#     fig.savefig(fig_path, dpi=150, bbox_inches='tight')
#     print(f"💾 Figura guardada en: {fig_path}")
#     plt.close(fig)
# 
# # Guardar resultados numéricos
# summary = {}
# for model_name, result in results.items():
#     if result is not None:
#         summary[model_name] = {
#             "prediction": result.get('prediction', 'N/A'),
#             "correct": result.get('correct', None),
#             "tscore": float(result.get('tscore', 0.0)),
#             "iscore": float(result.get('iscore', 0.0)),
#         }
# 
# json_path = output_dir / f"vqa_comparison_sample_{MUESTRA_A_ANALIZAR}.json"
# with open(json_path, 'w') as f:
#     json.dump({
#         "sample_idx": MUESTRA_A_ANALIZAR,
#         "question": question,
#         "answer": answer,
#         "category": category,
#         "candidates": candidates,
#         "results": summary
#     }, f, indent=2)
# 
# print(f"💾 Resultados guardados en: {json_path}")

# %% [markdown]
# ## 🔬 Análisis de múltiples muestras
#
# Para analizar múltiples muestras y obtener estadísticas agregadas,
# descomentar y ejecutar la siguiente celda.

# %%
# from mmshap_medclip.comparison_vqa import run_vqa_shap_on_models
# import pandas as pd
# 
# # Ejemplo: analizar 5 muestras
# sample_indices = [0, 10, 20, 30, 40]
# all_results = []
# 
# for idx in sample_indices:
#     print(f"\n📍 Procesando muestra {idx}...")
#     results, _, question, answer, candidates, category = run_vqa_shap_on_models(
#         models=loaded_models,
#         sample_idx=idx,
#         dataset=dataset,
#         device=device,
#         target_logit="correct",
#         verbose=False
#     )
#     
#     for model_name, result in results.items():
#         if result is not None:
#             all_results.append({
#                 "sample_idx": idx,
#                 "model": model_name,
#                 "prediction": result.get('prediction', 'N/A'),
#                 "correct": result.get('correct', None),
#                 "tscore": result.get('tscore', 0.0),
#                 "iscore": result.get('iscore', 0.0),
#                 "category": category,
#                 "question": question[:50] + "..."
#             })
# 
# df_results = pd.DataFrame(all_results)
# print("\n📊 Primeras filas del DataFrame de resultados:")
# print(df_results.head(10))
# 
# # Estadísticas por modelo
# if not df_results.empty:
#     print("\n📈 Estadísticas por modelo:")
#     print(df_results.groupby('model')[['tscore', 'iscore']].mean().round(4))
#     
#     # Precisión por modelo
#     print("\n🎯 Precisión por modelo:")
#     for model_name in df_results['model'].unique():
#         model_df = df_results[df_results['model'] == model_name]
#         correct_count = model_df['correct'].sum()
#         total = len(model_df)
#         accuracy = correct_count / total if total > 0 else 0.0
#         print(f"   {model_name}: {accuracy:.2%} ({correct_count}/{total})")

# %% [markdown]
# ---
#
# ## 📝 Notas de Uso
#
# ### 🎯 Uso Básico
#
# 1. **Configurar ruta del dataset:**
#    Modifica `dataset_params["zip_path"]` con la ruta correcta al archivo ZIP de VQA-Med 2019.
#
# 2. **Cambiar la muestra a analizar:**
#    Modifica la variable `MUESTRA_A_ANALIZAR` en la celda correspondiente.
#
# 3. **Re-ejecutar el análisis:**
#    Simplemente ejecuta las celdas de nuevo con el nuevo número de muestra.
#
# 4. **Ver resultados:**
#    - Heatmaps individuales detallados para cada modelo
#    - Resumen de métricas en tabla
#    - Análisis de balance multimodal
#
# ### 📊 Métricas Explicadas
#
# - **Predicción**: Candidato predicho por el modelo
# - **Correcto**: Si la predicción coincide con la respuesta correcta
# - **TScore**: Proporción de importancia asignada al texto (0-100%)
# - **IScore**: Proporción de importancia asignada a la imagen (0-100%)
# - **Balance ideal**: TScore ≈ IScore ≈ 50%
#
# ### 🔬 Funciones Disponibles
#
# - `load_vqa_models()`: Carga PubMedCLIP y BiomedCLIP
# - `run_vqa_shap_on_models()`: Ejecuta VQA+SHAP en todos los modelos
# - `plot_vqa_comparison()`: Muestra comparación visual de modelos
# - `print_vqa_summary()`: Imprime resumen comparativo en tabla
# - `plot_vqa()`: Genera heatmap individual detallado
#
# ### ⚙️ Parámetros Importantes
#
# - `target_logit`: "correct" (explicar logit del candidato correcto) o "predicted" (explicar logit del predicho)
# - SHAP solo se aplica a imagen y pregunta, NO a los candidatos
#
# ---
#
# **Proyecto de tesis sobre balance multimodal en modelos CLIP médicos aplicados a VQA**

