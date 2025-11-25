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
# ## 🚀 Análisis Batch de SHAP en VQA-Med 2019 (Sin Heatmaps)
#
# Esta sección replica el pipeline blindado del notebook de ISA pero adaptado a VQA.
# Permite recorrer todo el split, almacenar las métricas clave por modelo y retomar
# la ejecución automáticamente si se interrumpe.
#
# **Características:**
# - ✅ Guarda automáticamente después de cada muestra
# - ✅ Salta muestras completas y re-procesa las que tengan NaN
# - ✅ Continua desde el último índice pendiente
# - ✅ Registra: `Iscore`, `Tscore`, `Logit`, `Correct` por modelo
# - ✅ Añade metadatos útiles (`question_length`, `answer_length`, `candidate_count`, `category`, `timestamp`)
# - 🕒 Si el contador supera los 60 minutos, el ETA se muestra como `3horas y 45min`

# %%
from mmshap_medclip.comparison_vqa import batch_vqa_shap_analysis

# 🎯 CONFIGURACIÓN: Ajustar según tus necesidades
target_logit = "correct"  # "correct" explica la respuesta correcta; "predicted" explica la predicción
START_IDX = 0
END_IDX = None  # None = recorre todo el dataset
CSV_PATH = "outputs/vqa_batch_shap_results.csv"

# Ejecutar análisis batch (sin heatmaps)
df_vqa_batch = batch_vqa_shap_analysis(
    models=loaded_models,
    dataset=dataset,
    device=device,
    start_idx=START_IDX,
    end_idx=END_IDX,
    csv_path=CSV_PATH,
    target_logit=target_logit,
    verbose=True,
    show_dataframe=True
)

print("\n📊 Primeras filas del DataFrame de resultados:")
print(df_vqa_batch.head(10))

if not df_vqa_batch.empty:
    print("\n📈 Estadísticas resumidas:")
    print(f"   Total de muestras procesadas: {len(df_vqa_batch)}")

    for model_name in loaded_models.keys():
        if loaded_models[model_name] is None:
            continue
        iscore_col = f'Iscore_{model_name}'
        if iscore_col in df_vqa_batch.columns:
            serie = df_vqa_batch[iscore_col].dropna()
            if not serie.empty:
                avg_iscore = serie.mean()
                print(f"   {model_name} - IScore promedio: {avg_iscore:.2%}")

    print("\n🎯 Precisión por modelo:")
    for model_name in loaded_models.keys():
        if loaded_models[model_name] is None:
            continue
        correct_col = f'Correct_{model_name}'
        if correct_col in df_vqa_batch.columns:
            serie = df_vqa_batch[correct_col].dropna()
            if not serie.empty:
                accuracy = serie.mean()
                total = len(serie)
                correct = int((serie == True).sum())
                print(f"   {model_name}: {accuracy:.2%} ({correct}/{total})")

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
