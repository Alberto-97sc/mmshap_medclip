# 🔬 Comparación de Modelos CLIP Médicos

Este directorio contiene el script `compare_all_models.py` que permite comparar el funcionamiento de los 4 modelos CLIP médicos en la misma muestra.

## 📋 Modelos Incluidos

1. **PubMedCLIP** - Modelo entrenado en literatura biomédica de PubMed
2. **BioMedCLIP** - Modelo de Microsoft entrenado en datos biomédicos
3. **RCLIP** - Modelo especializado en radiología
4. **WhyXRayCLIP** - Modelo enfocado en rayos X con explicaciones

## 🚀 Uso Rápido

### Como Script Python

```bash
cd /root/mmshap_medclip
python experiments/compare_all_models.py
```

### Como Notebook Jupyter

1. **Convertir a notebook:**
   ```bash
   jupytext --to notebook experiments/compare_all_models.py
   ```

2. **Abrir el notebook:**
   ```bash
   jupyter notebook experiments/compare_all_models.ipynb
   ```

3. **Modificar la muestra:**
   En la celda correspondiente, cambia:
   ```python
   MUESTRA_A_ANALIZAR = 154  # Cambia este número
   ```

## 📊 Salida del Script

El script ejecuta SHAP en todos los modelos y genera:

1. **Visualización comparativa** - Grid 2x2 con los 4 modelos
   - Heatmap de imagen con overlay SHAP
   - Heatmap de texto con palabras coloreadas según importancia
2. **Tabla de métricas** - Logit, TScore, IScore para cada modelo
3. **Análisis de balance** - Identificación del modelo más balanceado

### Ejemplo de salida:

```
🔄 Cargando modelo PubMedCLIP...
✅ PubMedCLIP cargado exitosamente
🔄 Cargando modelo BioMedCLIP...
✅ BioMedCLIP cargado exitosamente
...

🔄 Ejecutando SHAP en PubMedCLIP...
✅ PubMedCLIP: logit=0.2345 | TScore=45.67% | IScore=54.33%
...

📊 RESUMEN COMPARATIVO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Modelo                  Logit    TScore    IScore
────────────────────────────────────────────────────
PubMedCLIP             0.2345    45.67%    54.33%
BioMedCLIP             0.3456    42.10%    57.90%
RCLIP                  0.1234    48.20%    51.80%
WhyXRayCLIP            0.2890    46.50%    53.50%
```

## 🔧 Funciones Disponibles

Todas las funciones están en el módulo `mmshap_medclip.comparison`:

### Funciones Principales

- **`load_all_models(device)`** - Carga los 4 modelos
- **`run_shap_on_all_models(models, sample_idx, dataset, device)`** - Ejecuta SHAP
- **`plot_comparison_simple(results, image, caption, sample_idx)`** - Visualización rápida
- **`print_summary(results)`** - Resumen en tabla

### Funciones Avanzadas

- **`plot_individual_heatmaps(results, image, caption)`** - Heatmaps detallados individuales
- **`save_comparison(results, image, caption, sample_idx, output_dir)`** - Guarda en disco
- **`analyze_multiple_samples(models, dataset, device, sample_indices)`** - Análisis batch

## 📖 Métricas Explicadas

- **Logit**: Score de similitud imagen-texto (cuanto más alto, más similar)
- **TScore**: % de importancia asignada al texto (0-100%)
- **IScore**: % de importancia asignada a la imagen (0-100%)
- **Balance ideal**: TScore ≈ IScore ≈ 50% (ambas modalidades igualmente importantes)

## 💡 Ejemplos de Uso

### Analizar una muestra específica

```python
from mmshap_medclip.comparison import load_all_models, run_shap_on_all_models, plot_comparison_simple
from mmshap_medclip.devices import get_device
from mmshap_medclip.registry import build_dataset
from mmshap_medclip.io_utils import load_config

# Setup
device = get_device()
cfg = load_config("configs/roco_isa_pubmedclip.yaml")
dataset = build_dataset(cfg["dataset"])

# Cargar modelos
models = load_all_models(device)
loaded_models = {k: v for k, v in models.items() if v is not None}

# Analizar muestra
results, image, caption = run_shap_on_all_models(
    loaded_models, sample_idx=154, dataset=dataset, device=device
)

# Visualizar
fig = plot_comparison_simple(results, image, caption, sample_idx=154)
plt.show()
```

### Analizar múltiples muestras

```python
from mmshap_medclip.comparison import analyze_multiple_samples

sample_indices = [10, 50, 100, 154, 200]
df = analyze_multiple_samples(loaded_models, dataset, device, sample_indices)

# Ver estadísticas
print(df.groupby('model')[['logit', 'tscore', 'iscore']].mean())
```

### Guardar resultados

```python
from mmshap_medclip.comparison import save_comparison

save_comparison(results, image, caption, sample_idx=154, output_dir="outputs")
# Guarda: outputs/comparison_sample_154.png
#         outputs/comparison_sample_154.json
```

## 🏗️ Arquitectura

```
mmshap_medclip/
├── src/mmshap_medclip/
│   └── comparison.py          # Módulo con funciones robustas
└── experiments/
    └── compare_all_models.py  # Script ligero para pruebas
```

El código está modularizado:
- **`comparison.py`**: Funciones reutilizables y robustas
- **`compare_all_models.py`**: Script minimalista solo para ejecutar pruebas

## 🎨 Personalización

Para modificar el comportamiento, edita el módulo `comparison.py`:

```python
# src/mmshap_medclip/comparison.py

# Cambiar tamaño de figuras
fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows))

# Cambiar transparencia de overlays
ax.imshow(heat_up, cmap='coolwarm', norm=norm, alpha=0.4)  # Cambiar 0.4

# Agregar más modelos
model_configs = {
    "NuevoModelo": {
        "name": "nuevo-modelo",
        "params": {...}
    }
}
```

## 📝 Notas

- Los modelos se cargan una sola vez al inicio
- SHAP puede tardar varios segundos por modelo
- Si un modelo falla, el script continúa con los demás
- Los heatmaps usan la misma normalización para facilitar comparación

## 🐛 Troubleshooting

**Error: "CUDA out of memory"**
```python
# Usar CPU en lugar de GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

**Error: "Model not found"**
- Verifica que tienes conexión a internet
- Los modelos se descargan de HuggingFace automáticamente

**El notebook no se genera correctamente**
```bash
# Reinstalar jupytext
pip install --upgrade jupytext

# Verificar formato
jupytext --test experiments/compare_all_models.py
```

---

**Creado para el proyecto de tesis sobre balance multimodal en modelos CLIP médicos**

