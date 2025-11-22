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
# # 📊 Análisis de Resultados Batch SHAP
#
# Este notebook analiza los resultados del análisis batch de SHAP para los 4 modelos CLIP médicos.
# Genera estadísticas descriptivas, gráficas comparativas, métricas de balance multimodal,
# análisis estadísticos inferenciales y visualizaciones para presentación.
#
# **Modelos analizados:**
# - PubMedCLIP
# - BioMedCLIP
# - RCLIP
# - WhyXRayCLIP
#
# ---

# %% [markdown]
# ## 📦 Configuración inicial

# %%
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, wilcoxon, shapiro
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficas
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

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
# ## 📁 Configuración de rutas
#
# **⚠️ IMPORTANTE:** Cambia la ruta del CSV aquí según necesites

# %%
# 🎯 CONFIGURACIÓN: Cambia esta ruta según el CSV que quieras analizar
CSV_PATH = "outputs/batch_shap_results.csv"  # Ruta relativa al directorio raíz
OUTPUT_DIR = "outputs/analysis"  # Directorio donde guardar gráficas y reportes

# Crear directorio de salida si no existe
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print(f"📂 CSV a analizar: {CSV_PATH}")
print(f"📂 Directorio de salida: {OUTPUT_DIR}")

# %% [markdown]
# ## 📊 Carga de datos

# %%
# Cargar CSV
if not Path(CSV_PATH).exists():
    raise FileNotFoundError(f"❌ No se encontró el archivo: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"✅ CSV cargado exitosamente")
print(f"📊 Total de muestras: {len(df)}")
print(f"📋 Columnas: {list(df.columns)}")

# Mostrar primeras filas
print("\n📋 Primeras 5 filas del dataset:")
print(df.head())

# %% [markdown]
# ## 🔧 Preparación de datos

# %%
# Extraer nombres de modelos de las columnas
model_names = []
for col in df.columns:
    if col.startswith('Iscore_'):
        model_name = col.replace('Iscore_', '')
        model_names.append(model_name)

print(f"🤖 Modelos encontrados: {model_names}")

# Crear diccionario con datos por modelo (filtrando NaN)
models_data = {}
for model_name in model_names:
    iscore_raw = df[f'Iscore_{model_name}'].values
    tscore_raw = df[f'Tscore_{model_name}'].values
    logit_raw = df[f'Logit_{model_name}'].values

    # Filtrar NaN y crear máscara de valores válidos
    valid_mask = ~(np.isnan(iscore_raw) | np.isnan(tscore_raw) | np.isnan(logit_raw))

    models_data[model_name] = {
        'iscore': iscore_raw[valid_mask],
        'tscore': tscore_raw[valid_mask],
        'logit': logit_raw[valid_mask],
        'valid_count': np.sum(valid_mask),
        'total_count': len(iscore_raw)
    }

    if np.sum(valid_mask) < len(iscore_raw):
        print(f"⚠️  {model_name}: {len(iscore_raw) - np.sum(valid_mask)} valores NaN encontrados y filtrados")

# %% [markdown]
# ## 1️⃣ Estadísticas Descriptivas por Modelo

# %%
print("="*80)
print("📊 ESTADÍSTICAS DESCRIPTIVAS POR MODELO")
print("="*80)

# Crear DataFrame con estadísticas
stats_list = []

for model_name in model_names:
    iscore = models_data[model_name]['iscore']
    tscore = models_data[model_name]['tscore']
    logit = models_data[model_name]['logit']
    valid_count = models_data[model_name]['valid_count']

    # Usar funciones que manejan NaN correctamente
    if len(iscore) > 0:
        stats_list.append({
            'Modelo': model_name,
            'Métrica': 'IScore',
            'Media': np.nanmean(iscore),
            'Mediana': np.nanmedian(iscore),
            'Desv. Est.': np.nanstd(iscore),
            'Mínimo': np.nanmin(iscore),
            'Máximo': np.nanmax(iscore),
            'Q25': np.nanpercentile(iscore, 25),
            'Q75': np.nanpercentile(iscore, 75),
            'CV (%)': (np.nanstd(iscore) / np.nanmean(iscore)) * 100 if np.nanmean(iscore) > 0 else 0,
            'N válidos': valid_count
        })

        stats_list.append({
            'Modelo': model_name,
            'Métrica': 'TScore',
            'Media': np.nanmean(tscore),
            'Mediana': np.nanmedian(tscore),
            'Desv. Est.': np.nanstd(tscore),
            'Mínimo': np.nanmin(tscore),
            'Máximo': np.nanmax(tscore),
            'Q25': np.nanpercentile(tscore, 25),
            'Q75': np.nanpercentile(tscore, 75),
            'CV (%)': (np.nanstd(tscore) / np.nanmean(tscore)) * 100 if np.nanmean(tscore) > 0 else 0,
            'N válidos': valid_count
        })

        stats_list.append({
            'Modelo': model_name,
            'Métrica': 'Logit',
            'Media': np.nanmean(logit),
            'Mediana': np.nanmedian(logit),
            'Desv. Est.': np.nanstd(logit),
            'Mínimo': np.nanmin(logit),
            'Máximo': np.nanmax(logit),
            'Q25': np.nanpercentile(logit, 25),
            'Q75': np.nanpercentile(logit, 75),
            'CV (%)': (np.nanstd(logit) / np.nanmean(logit)) * 100 if np.nanmean(logit) > 0 else 0,
            'N válidos': valid_count
        })
    else:
        print(f"⚠️  {model_name}: No hay datos válidos para calcular estadísticas")

df_stats = pd.DataFrame(stats_list)

# Mostrar tabla formateada
print("\n📋 Tabla de estadísticas descriptivas:")
print(df_stats.to_string(index=False))

# Guardar en CSV
stats_path = Path(OUTPUT_DIR) / "estadisticas_descriptivas.csv"
df_stats.to_csv(stats_path, index=False)
print(f"\n💾 Estadísticas guardadas en: {stats_path}")

# %% [markdown]
# ### 📊 Tabla Resumen: Promedio de IScore por Modelo

# %%
# Crear tabla resumen con promedio de IScore por modelo
iscore_summary = []
for model_name in model_names:
    iscore = models_data[model_name]['iscore']
    iscore_summary.append({
        'Modelo': model_name,
        'IScore Promedio': np.nanmean(iscore),
        'IScore Mediana': np.nanmedian(iscore),
        'IScore Desv. Est.': np.nanstd(iscore),
        'IScore Mínimo': np.nanmin(iscore),
        'IScore Máximo': np.nanmax(iscore),
        'N Muestras Válidas': models_data[model_name]['valid_count']
    })

df_iscore_summary = pd.DataFrame(iscore_summary)
# Ordenar por IScore promedio (de mayor a menor)
df_iscore_summary = df_iscore_summary.sort_values('IScore Promedio', ascending=False)

print("="*80)
print("📊 RESUMEN: PROMEDIO DE ISCORE POR MODELO")
print("="*80)
print("\n" + df_iscore_summary.to_string(index=False))
print("\n" + "="*80)

# Guardar tabla
iscore_summary_path = Path(OUTPUT_DIR) / "resumen_iscore_promedio.csv"
df_iscore_summary.to_csv(iscore_summary_path, index=False)
print(f"💾 Tabla guardada en: {iscore_summary_path}")

# Visualización de la tabla
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Formatear valores para mostrar
table_data = df_iscore_summary.copy()
table_data['IScore Promedio'] = table_data['IScore Promedio'].apply(lambda x: f'{x:.4f}')
table_data['IScore Mediana'] = table_data['IScore Mediana'].apply(lambda x: f'{x:.4f}')
table_data['IScore Desv. Est.'] = table_data['IScore Desv. Est.'].apply(lambda x: f'{x:.4f}')
table_data['IScore Mínimo'] = table_data['IScore Mínimo'].apply(lambda x: f'{x:.4f}')
table_data['IScore Máximo'] = table_data['IScore Máximo'].apply(lambda x: f'{x:.4f}')

table = ax.table(
    cellText=table_data.values,
    colLabels=table_data.columns,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Colorear la columna de IScore Promedio
for i in range(len(table_data) + 1):
    if i == 0:  # Header
        table[(i, 1)].set_facecolor('#4CAF50')
        table[(i, 1)].set_text_props(weight='bold', color='white')
    else:  # Data rows
        table[(i, 1)].set_facecolor('#E8F5E9')

ax.set_title('Resumen: Promedio de IScore por Modelo', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
iscore_table_path = Path(OUTPUT_DIR) / "tabla_iscore_promedio.png"
plt.savefig(iscore_table_path, dpi=300, bbox_inches='tight')
print(f"💾 Gráfica de tabla guardada en: {iscore_table_path}")
plt.show()

# %% [markdown]
# ## 2️⃣ Gráficas Comparativas

# %% [markdown]
# ### 2.1 Boxplot de IScore por Modelo

# %%
fig, ax = plt.subplots(figsize=(12, 6))

# Preparar datos para boxplot
iscore_data = [models_data[model]['iscore'] for model in model_names]

# Boxplot IScore
bp = ax.boxplot(iscore_data, labels=model_names, patch_artist=True)
ax.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Balance ideal (50%)')
ax.set_title('Distribución de IScore por Modelo', fontsize=14, fontweight='bold')
ax.set_ylabel('IScore', fontsize=12)
ax.set_xlabel('Modelo', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(0, 1)

# Colorear boxes
colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

plt.tight_layout()
boxplot_path = Path(OUTPUT_DIR) / "boxplot_iscore.png"
plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
print(f"💾 Gráfica guardada en: {boxplot_path}")
plt.show()

# %% [markdown]
# ### 2.2 Violin Plot de IScore por Modelo

# %%
fig, ax = plt.subplots(figsize=(12, 6))

# Preparar datos en formato largo para seaborn
iscore_df = pd.DataFrame({
    'Modelo': np.repeat(model_names, [len(d) for d in iscore_data]),
    'IScore': np.concatenate(iscore_data)
})

# Violin plot IScore
sns.violinplot(data=iscore_df, x='Modelo', y='IScore', ax=ax, inner='box')
ax.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Balance ideal (50%)')
ax.set_title('Distribución de IScore por Modelo (Violin Plot)', fontsize=14, fontweight='bold')
ax.set_ylabel('IScore', fontsize=12)
ax.set_xlabel('Modelo', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_ylim(0, 1)

plt.tight_layout()
violin_path = Path(OUTPUT_DIR) / "violinplot_iscore.png"
plt.savefig(violin_path, dpi=300, bbox_inches='tight')
print(f"💾 Gráfica guardada en: {violin_path}")
plt.show()

# %% [markdown]
# ### 2.3 Scatter Plot: IScore vs TScore

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, model_name in enumerate(model_names):
    iscore = models_data[model_name]['iscore']
    tscore = models_data[model_name]['tscore']

    axes[idx].scatter(iscore, tscore, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    axes[idx].plot([0, 1], [1, 0], 'r--', linewidth=2, label='Balance ideal (IScore + TScore = 1)')
    axes[idx].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    axes[idx].axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    axes[idx].set_xlabel('IScore', fontsize=12)
    axes[idx].set_ylabel('TScore', fontsize=12)
    axes[idx].set_title(f'{model_name}\nIScore vs TScore', fontsize=13, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend()
    axes[idx].set_xlim(0, 1)
    axes[idx].set_ylim(0, 1)
    axes[idx].set_aspect('equal')

plt.suptitle('Relación IScore vs TScore por Modelo', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
scatter_path = Path(OUTPUT_DIR) / "scatter_iscore_vs_tscore.png"
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
print(f"💾 Gráfica guardada en: {scatter_path}")
plt.show()

# %% [markdown]
# ### 2.4 Heatmap de Correlaciones entre Modelos

# %%
# Calcular matriz de correlación de IScores entre modelos
# Usar solo las muestras que tienen datos válidos en TODOS los modelos
# Crear DataFrame con IScores de todos los modelos
iscore_df_corr = pd.DataFrame({
    model: df[f'Iscore_{model}'] for model in model_names
})

# Calcular correlación usando pandas (maneja NaN automáticamente)
correlation_matrix = iscore_df_corr.corr().values

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt='.3f',
    cmap='coolwarm',
    center=0,
    square=True,
    xticklabels=model_names,
    yticklabels=model_names,
    cbar_kws={'label': 'Coeficiente de Correlación'},
    linewidths=0.5,
    linecolor='black'
)
ax.set_title('Matriz de Correlación de IScores entre Modelos', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
corr_path = Path(OUTPUT_DIR) / "heatmap_correlaciones.png"
plt.savefig(corr_path, dpi=300, bbox_inches='tight')
print(f"💾 Gráfica guardada en: {corr_path}")
plt.show()

# %% [markdown]
# ## 3️⃣ Métricas de Balance Multimodal

# %%
print("="*80)
print("🎯 MÉTRICAS DE BALANCE MULTIMODAL")
print("="*80)

# Calcular métricas de balance para cada modelo
balance_metrics = []

for model_name in model_names:
    iscore = models_data[model_name]['iscore']
    tscore = models_data[model_name]['tscore']

    # Balance Score: 1 - |IScore - 0.5| (más cercano a 1 = más balanceado)
    balance_score = 1 - np.abs(iscore - 0.5)

    # Desviación del balance: |IScore - TScore|
    balance_deviation = np.abs(iscore - tscore)

    # Ratio IScore/TScore (ideal = 1.0)
    ratio = iscore / (tscore + 1e-10)  # Evitar división por cero

    # Porcentaje de muestras "balanceadas" (dentro de ±10% de 50/50)
    balanced_samples = np.sum(np.abs(iscore - 0.5) <= 0.1)
    balanced_percentage = (balanced_samples / len(iscore)) * 100

    balance_metrics.append({
        'Modelo': model_name,
        'Balance Score (Media)': np.mean(balance_score),
        'Balance Score (Mediana)': np.median(balance_score),
        'Desviación Balance (Media)': np.mean(balance_deviation),
        'Ratio IScore/TScore (Media)': np.mean(ratio),
        'Muestras Balanceadas (%)': balanced_percentage,
        'Muestras Balanceadas (N)': balanced_samples,
        'Total Muestras': len(iscore)
    })

df_balance = pd.DataFrame(balance_metrics)

# Ordenar por Balance Score (mayor a menor)
df_balance = df_balance.sort_values('Balance Score (Media)', ascending=False)

print("\n📋 Métricas de Balance Multimodal:")
print(df_balance.to_string(index=False))

# Guardar en CSV
balance_path = Path(OUTPUT_DIR) / "metricas_balance_multimodal.csv"
df_balance.to_csv(balance_path, index=False)
print(f"\n💾 Métricas guardadas en: {balance_path}")

# Visualización de Balance Score
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(model_names))
bars = ax.barh(x_pos, df_balance['Balance Score (Media)'], color=colors[:len(model_names)])
ax.set_yticks(x_pos)
ax.set_yticklabels(df_balance['Modelo'])
ax.set_xlabel('Balance Score (Media)', fontsize=12)
ax.set_title('Balance Multimodal por Modelo\n(Mayor = Más Balanceado, Ideal = 1.0)',
              fontsize=14, fontweight='bold')
ax.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='Balance perfecto')
ax.grid(True, alpha=0.3, axis='x')
ax.legend()

# Agregar valores en las barras
for i, (bar, value) in enumerate(zip(bars, df_balance['Balance Score (Media)'])):
    ax.text(value + 0.01, i, f'{value:.3f}', va='center', fontweight='bold')

plt.tight_layout()
balance_viz_path = Path(OUTPUT_DIR) / "balance_score_comparison.png"
plt.savefig(balance_viz_path, dpi=300, bbox_inches='tight')
print(f"💾 Gráfica guardada en: {balance_viz_path}")
plt.show()

# %% [markdown]
# ## 4️⃣ Análisis Estadísticos Inferenciales

# %%
print("="*80)
print("📈 ANÁLISIS ESTADÍSTICOS INFERENCIALES")
print("="*80)

# 4.1 Test de Normalidad (Shapiro-Wilk)
print("\n🔍 4.1 Test de Normalidad (Shapiro-Wilk):")
print("-" * 80)
normality_results = []

for model_name in model_names:
    iscore = models_data[model_name]['iscore']
    tscore = models_data[model_name]['tscore']

    # Test para IScore
    stat_iscore, p_iscore = shapiro(iscore)
    # Test para TScore
    stat_tscore, p_tscore = shapiro(tscore)

    normality_results.append({
        'Modelo': model_name,
        'IScore - Estadístico': stat_iscore,
        'IScore - p-value': p_iscore,
        'IScore - Normal?': 'Sí' if p_iscore > 0.05 else 'No',
        'TScore - Estadístico': stat_tscore,
        'TScore - p-value': p_tscore,
        'TScore - Normal?': 'Sí' if p_tscore > 0.05 else 'No'
    })

df_normality = pd.DataFrame(normality_results)
print(df_normality.to_string(index=False))

# 4.2 Test de Kruskal-Wallis (comparación entre modelos)
print("\n🔍 4.2 Test de Kruskal-Wallis (Comparación de IScores entre modelos):")
print("-" * 80)
# Usar solo muestras que tienen datos válidos en TODOS los modelos
iscore_df_for_test = pd.DataFrame({
    model: df[f'Iscore_{model}'] for model in model_names
})
# Filtrar filas donde TODOS los modelos tienen datos válidos
valid_rows = iscore_df_for_test.dropna()
iscore_groups = [valid_rows[model].values for model in model_names]
if len(valid_rows) > 0:
    h_stat, p_value = kruskal(*iscore_groups)
    print(f"Estadístico H: {h_stat:.4f}")
    print(f"p-value: {p_value:.6f}")
    print(f"¿Hay diferencias significativas? {'Sí' if p_value < 0.05 else 'No'} (α=0.05)")
    print(f"Muestras usadas: {len(valid_rows)} (de {len(df)} totales)")
else:
    print("⚠️  No hay suficientes datos válidos para realizar el test")

# 4.3 Comparaciones pareadas (Wilcoxon)
print("\n🔍 4.3 Comparaciones Pareadas (Wilcoxon):")
print("-" * 80)
pairwise_results = []

for i, model1 in enumerate(model_names):
    for j, model2 in enumerate(model_names):
        if i < j:
            # Usar datos del DataFrame original para tener la misma longitud
            iscore1 = df[f'Iscore_{model1}'].values
            iscore2 = df[f'Iscore_{model2}'].values

            # Filtrar solo muestras válidas en ambos modelos
            valid_mask = ~(np.isnan(iscore1) | np.isnan(iscore2))
            iscore1_valid = iscore1[valid_mask]
            iscore2_valid = iscore2[valid_mask]

            if len(iscore1_valid) > 0 and len(iscore2_valid) > 0:
                stat, p_val = wilcoxon(iscore1_valid, iscore2_valid)

                pairwise_results.append({
                    'Modelo 1': model1,
                    'Modelo 2': model2,
                    'Estadístico': stat,
                    'p-value': p_val,
                    'Significativo?': 'Sí' if p_val < 0.05 else 'No',
                    'N muestras válidas': len(iscore1_valid)
                })
            else:
                pairwise_results.append({
                    'Modelo 1': model1,
                    'Modelo 2': model2,
                    'Estadístico': np.nan,
                    'p-value': np.nan,
                    'Significativo?': 'N/A',
                    'N muestras válidas': 0
                })

df_pairwise = pd.DataFrame(pairwise_results)
print(df_pairwise.to_string(index=False))

# Guardar resultados
stats_inf_path = Path(OUTPUT_DIR) / "analisis_estadisticos_inferenciales.csv"
df_pairwise.to_csv(stats_inf_path, index=False)
print(f"\n💾 Resultados guardados en: {stats_inf_path}")

# %% [markdown]
# ## 5️⃣ Análisis de Relación con Caption Length

# %%
print("="*80)
print("📏 ANÁLISIS DE RELACIÓN CON CAPTION LENGTH")
print("="*80)

caption_lengths = df['caption_length'].values

# Calcular correlaciones
# Usar datos originales del DataFrame para mantener la misma longitud
correlations = []
for model_name in model_names:
    iscore_raw = df[f'Iscore_{model_name}'].values
    tscore_raw = df[f'Tscore_{model_name}'].values

    # Filtrar NaN para tener arrays de la misma longitud
    valid_mask_iscore = ~np.isnan(iscore_raw)
    valid_mask_tscore = ~np.isnan(tscore_raw)

    # Correlación de Pearson (solo con valores válidos)
    if np.sum(valid_mask_iscore) > 0:
        corr_iscore, p_iscore = stats.pearsonr(
            caption_lengths[valid_mask_iscore],
            iscore_raw[valid_mask_iscore]
        )
    else:
        corr_iscore, p_iscore = np.nan, np.nan

    if np.sum(valid_mask_tscore) > 0:
        corr_tscore, p_tscore = stats.pearsonr(
            caption_lengths[valid_mask_tscore],
            tscore_raw[valid_mask_tscore]
        )
    else:
        corr_tscore, p_tscore = np.nan, np.nan

    correlations.append({
        'Modelo': model_name,
        'Correlación IScore-Caption': corr_iscore,
        'p-value IScore': p_iscore,
        'Significativa IScore?': 'Sí' if p_iscore < 0.05 else 'No',
        'Correlación TScore-Caption': corr_tscore,
        'p-value TScore': p_tscore,
        'Significativa TScore?': 'Sí' if p_tscore < 0.05 else 'No'
    })

df_correlations = pd.DataFrame(correlations)
print("\n📋 Correlaciones con Caption Length:")
print(df_correlations.to_string(index=False))

# Guardar
corr_caption_path = Path(OUTPUT_DIR) / "correlaciones_caption_length.csv"
df_correlations.to_csv(corr_caption_path, index=False)
print(f"\n💾 Correlaciones guardadas en: {corr_caption_path}")

# Visualización: Scatter plots
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, model_name in enumerate(model_names):
    # Usar datos originales y filtrar NaN
    iscore_raw = df[f'Iscore_{model_name}'].values
    valid_mask = ~np.isnan(iscore_raw)
    iscore_valid = iscore_raw[valid_mask]
    caption_valid = caption_lengths[valid_mask]

    axes[idx].scatter(caption_valid, iscore_valid, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # Línea de regresión (solo con valores válidos)
    if len(iscore_valid) > 1:
        z = np.polyfit(caption_valid, iscore_valid, 1)
        p = np.poly1d(z)
        x_line = np.linspace(caption_valid.min(), caption_valid.max(), 100)
        axes[idx].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Regresión lineal')

    corr = df_correlations[df_correlations['Modelo'] == model_name]['Correlación IScore-Caption'].values[0]
    axes[idx].set_xlabel('Caption Length (caracteres)', fontsize=12)
    axes[idx].set_ylabel('IScore', fontsize=12)
    axes[idx].set_title(f'{model_name}\nIScore vs Caption Length\n(r={corr:.3f})',
                        fontsize=13, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend()

plt.suptitle('Relación entre IScore y Longitud del Caption', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
caption_corr_path = Path(OUTPUT_DIR) / "scatter_caption_length_vs_iscore.png"
plt.savefig(caption_corr_path, dpi=300, bbox_inches='tight')
print(f"💾 Gráfica guardada en: {caption_corr_path}")
plt.show()

# %% [markdown]
# ## 6️⃣ Visualizaciones para Presentación

# %% [markdown]
# ### 6.1 Dashboard Comparativo Completo

# %%
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Boxplot IScore (arriba izquierda)
ax1 = fig.add_subplot(gs[0, 0])
bp = ax1.boxplot(iscore_data, labels=model_names, patch_artist=True)
ax1.axhline(y=0.5, color='r', linestyle='--', linewidth=2)
ax1.set_title('Distribución IScore', fontweight='bold')
ax1.set_ylabel('IScore')
ax1.grid(True, alpha=0.3)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 2. Balance Score (arriba centro)
ax2 = fig.add_subplot(gs[0, 1])
x_pos = np.arange(len(model_names))
bars = ax2.barh(x_pos, df_balance['Balance Score (Media)'], color=colors[:len(model_names)])
ax2.set_yticks(x_pos)
ax2.set_yticklabels(df_balance['Modelo'])
ax2.set_xlabel('Balance Score')
ax2.set_title('Balance Multimodal', fontweight='bold')
ax2.axvline(x=1.0, color='r', linestyle='--', linewidth=2)
ax2.grid(True, alpha=0.3, axis='x')

# 3. Scatter IScore vs TScore (arriba derecha)
ax3 = fig.add_subplot(gs[0, 2])
x_pos = np.arange(len(model_names))
bars = ax3.barh(x_pos, df_balance['Balance Score (Media)'], color=colors[:len(model_names)])
ax3.set_yticks(x_pos)
ax3.set_yticklabels(df_balance['Modelo'])
ax3.set_xlabel('Balance Score')
ax3.set_title('Balance Multimodal', fontweight='bold')
ax3.axvline(x=1.0, color='r', linestyle='--', linewidth=2)
ax3.grid(True, alpha=0.3, axis='x')

# 4. Scatter IScore vs TScore combinado (centro izquierda)
ax4 = fig.add_subplot(gs[1, 0])
for model_name, color in zip(model_names, colors):
    iscore = models_data[model_name]['iscore']
    tscore = models_data[model_name]['tscore']
    ax4.scatter(iscore, tscore, alpha=0.5, s=30, label=model_name, color=color)
ax4.plot([0, 1], [1, 0], 'r--', linewidth=2, label='Balance ideal')
ax4.set_xlabel('IScore')
ax4.set_ylabel('TScore')
ax4.set_title('IScore vs TScore (Todos los modelos)', fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)

# 5. Heatmap de correlaciones (centro)
ax5 = fig.add_subplot(gs[1, 1])
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True,
    xticklabels=model_names,
    yticklabels=model_names,
    cbar_kws={'label': 'Correlación'},
    ax=ax5,
    linewidths=0.5
)
ax5.set_title('Correlación entre Modelos', fontweight='bold')

# 6. Caption Length vs IScore (centro derecha)
ax6 = fig.add_subplot(gs[1, 2])
for model_name, color in zip(model_names, colors):
    # Usar datos originales y filtrar NaN
    iscore_raw = df[f'Iscore_{model_name}'].values
    valid_mask = ~np.isnan(iscore_raw)
    ax6.scatter(caption_lengths[valid_mask], iscore_raw[valid_mask],
                alpha=0.4, s=20, label=model_name, color=color)
ax6.set_xlabel('Caption Length')
ax6.set_ylabel('IScore')
ax6.set_title('IScore vs Caption Length', fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# 7. Tabla de estadísticas (abajo izquierda)
ax7 = fig.add_subplot(gs[2, :])
ax7.axis('tight')
ax7.axis('off')
table_data = df_balance[['Modelo', 'Balance Score (Media)', 'Muestras Balanceadas (%)']].values
table = ax7.table(cellText=table_data, colLabels=['Modelo', 'Balance Score', 'Muestras Balanceadas (%)'],
                  cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
ax7.set_title('Resumen de Métricas de Balance', fontweight='bold', pad=20)

plt.suptitle('Dashboard Comparativo: Análisis de Balance Multimodal en Modelos CLIP Médicos',
             fontsize=18, fontweight='bold', y=0.98)

dashboard_path = Path(OUTPUT_DIR) / "dashboard_completo.png"
plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
print(f"💾 Dashboard guardado en: {dashboard_path}")
plt.show()

# %% [markdown]
# ### 6.2 Ranking de Modelos por Balance

# %%
fig, ax = plt.subplots(figsize=(12, 8))

# Ordenar por Balance Score
df_sorted = df_balance.sort_values('Balance Score (Media)', ascending=True)

y_pos = np.arange(len(model_names))
bars = ax.barh(y_pos, df_sorted['Balance Score (Media)'],
               color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))

ax.set_yticks(y_pos)
ax.set_yticklabels(df_sorted['Modelo'])
ax.set_xlabel('Balance Score (Media)', fontsize=14, fontweight='bold')
ax.set_title('Ranking de Modelos por Balance Multimodal\n(Mayor = Más Balanceado)',
             fontsize=16, fontweight='bold', pad=20)
ax.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='Balance perfecto (1.0)')
ax.grid(True, alpha=0.3, axis='x')
ax.legend(fontsize=12)

# Agregar valores y porcentaje de muestras balanceadas
for i, (bar, value, pct) in enumerate(zip(bars,
                                          df_sorted['Balance Score (Media)'],
                                          df_sorted['Muestras Balanceadas (%)'])):
    ax.text(value + 0.01, i, f'{value:.3f}\n({pct:.1f}% balanceadas)',
            va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
ranking_path = Path(OUTPUT_DIR) / "ranking_balance_modelos.png"
plt.savefig(ranking_path, dpi=300, bbox_inches='tight')
print(f"💾 Gráfica guardada en: {ranking_path}")
plt.show()

# %% [markdown]
# ## 📋 Resumen Final

# %%
print("="*80)
print("✅ ANÁLISIS COMPLETADO")
print("="*80)
print(f"\n📊 Total de muestras analizadas: {len(df)}")
print(f"🤖 Modelos analizados: {len(model_names)}")
print(f"📁 Archivos generados en: {OUTPUT_DIR}")
print("\n📋 Archivos creados:")
print(f"   • estadisticas_descriptivas.csv")
print(f"   • metricas_balance_multimodal.csv")
print(f"   • analisis_estadisticos_inferenciales.csv")
print(f"   • correlaciones_caption_length.csv")
print(f"   • boxplots_iscore_tscore.png")
print(f"   • violinplots_iscore_tscore.png")
print(f"   • scatter_iscore_vs_tscore.png")
print(f"   • heatmap_correlaciones.png")
print(f"   • balance_score_comparison.png")
print(f"   • scatter_caption_length_vs_iscore.png")
print(f"   • dashboard_completo.png")
print(f"   • ranking_balance_modelos.png")
print("\n🎉 ¡Análisis completo! Revisa los archivos generados.")

# %% [markdown]
# ---
#
# **Proyecto de tesis: Medición del balance multimodal con SHAP en modelos CLIP médicos**
