# Documentación técnica de mmshap_medclip

## Resumen del propósito
El repositorio implementa una canalización modular para medir el **balance multimodal** en modelos tipo CLIP aplicados a dominios médicos mediante explicaciones SHAP. Está diseñado para ejecutarse en entornos como Google Colab, con datasets almacenados en Google Drive, y ya incluye scripts listos para ejecutar experimentos completos sobre ROCO con PubMedCLIP o WhyXrayCLIP.【F:README.md†L1-L69】

## Visión general de la arquitectura
Al instalar el paquete en modo editable (`pip install -e .`), el paquete `mmshap_medclip` se registra automáticamente importando definiciones de modelos y datasets al cargar `mmshap_medclip/__init__.py`, lo que activa los decoradores de registro y permite construir componentes desde configuraciones YAML.【F:README.md†L73-L196】【F:src/mmshap_medclip/__init__.py†L1-L6】

Los puntos clave de la arquitectura son:

- **Carga de configuración**: `load_config` lee archivos YAML desde `configs/` y devuelve diccionarios que describen dataset y modelo.【F:src/mmshap_medclip/io_utils.py†L1-L4】【F:configs/roco_isa_pubmedclip.yaml†L1-L17】
- **Registro de fábricas**: `registry.py` mantiene mapas internos para modelos y datasets. Decoradores `@register_model` y `@register_dataset` exponen factorías que pueden instanciarse con `build_model` y `build_dataset`, asegurando que los submódulos correctos se importen bajo demanda.【F:src/mmshap_medclip/registry.py†L1-L63】
- **Gestión de dispositivos**: `devices.get_device` selecciona CUDA cuando está disponible y `move_to_device` traslada tensores o diccionarios de tensores al dispositivo solicitado.【F:src/mmshap_medclip/devices.py†L1-L11】
- **Wrappers de modelos CLIP**: `models.py` encapsula modelos de Hugging Face y OpenCLIP, exponiendo una API consistente (`processor`, `tokenizer`, `model`, tamaños de parches) y registrando variantes específicas como PubMedCLIP, OpenAI CLIP y WhyXrayCLIP.【F:src/mmshap_medclip/models.py†L10-L198】
- **Datasets**: `datasets/roco.py` implementa `RocoDataset`, que lee el ZIP de ROCO, indexa las rutas de imágenes y devuelve pares imagen-texto listos para usar.【F:src/mmshap_medclip/datasets/roco.py†L8-L70】
- **Utilidades de lotes y tokens**: `tasks/utils.py` contiene funciones para preparar lotes, contar tokens, generar IDs de parches de imagen y concatenar la secuencia texto-imagen utilizada por SHAP.【F:src/mmshap_medclip/tasks/utils.py†L1-L168】
- **Pipeline ISA**: `tasks/isa.py` implementa la tarea de Image-Sentence Alignment (ISA), integrando preparación de lotes, explicación con SHAP, cálculo de métricas y visualización opcional.【F:src/mmshap_medclip/tasks/isa.py†L1-L248】
- **Herramientas SHAP**: `shap_tools/masker.py` y `shap_tools/predictor.py` definen el “masker” personalizado que preserva tokens especiales y el `Predictor` que aplica máscaras de parches antes de evaluar el modelo.【F:src/mmshap_medclip/shap_tools/masker.py†L1-L86】【F:src/mmshap_medclip/shap_tools/predictor.py†L1-L117】
- **Visualización**: `vis/heatmaps.py` convierte los valores SHAP en mapas de calor combinados para texto e imagen, ajustando automáticamente escalas y rejillas de parches.【F:src/mmshap_medclip/vis/heatmaps.py†L1-L562】
- **Extensiones WhyXrayCLIP**: `tasks/whyxrayclip.py` añade herramientas para filtrar ROCO por palabras clave, muestrear ejemplos negativos y puntuar alineación con logits normalizados.【F:src/mmshap_medclip/tasks/whyxrayclip.py†L1-L242】

## Flujo end-to-end del experimento ISA
1. **Lectura de configuración**: se carga un YAML (p. ej. `configs/roco_isa_pubmedclip.yaml`) que define el dataset ROCO (ruta ZIP, split, columnas) y el modelo objetivo.【F:configs/roco_isa_pubmedclip.yaml†L1-L17】
2. **Construcción de dataset y modelo**: `build_dataset` crea una instancia de `RocoDataset`, que abre el ZIP, lee el CSV del split requerido y resuelve la ruta de cada imagen; `build_model` invoca la fábrica registrada para PubMedCLIP/WhyXrayCLIP y devuelve un wrapper con `processor` y `tokenizer`.【F:src/mmshap_medclip/registry.py†L47-L63】【F:src/mmshap_medclip/datasets/roco.py†L8-L70】【F:src/mmshap_medclip/models.py†L146-L198】
3. **Preparación de lote**: `prepare_batch` estandariza listas de imágenes y textos, asegura formato RGB, ejecuta el `processor` asociado al modelo y traslada los tensores al dispositivo detectado.【F:src/mmshap_medclip/tasks/utils.py†L8-L55】
4. **Cálculo de logits**: el wrapper produce `logits_per_image` normalizados para CLIP, que sirven como punto de partida para evaluar alineación.【F:src/mmshap_medclip/models.py†L129-L144】【F:src/mmshap_medclip/tasks/isa.py†L13-L57】
5. **Preparación para SHAP**: `compute_text_token_lengths` y `make_image_token_ids` calculan el número de tokens de texto y generan identificadores de parches según `pixel_values` y `patch_size`; `concat_text_image_tokens` produce la secuencia conjunta requerida por SHAP.【F:src/mmshap_medclip/tasks/utils.py†L59-L168】
6. **Masker y predictor**: `build_masker` garantiza que los tokens especiales (BOS/EOS) se preserven al aplicar máscaras y `Predictor` reconstruye tensores de entrada, aplicando enmascaramiento de parches antes de llamar al modelo dentro de un contexto AMP opcional.【F:src/mmshap_medclip/shap_tools/masker.py†L5-L86】【F:src/mmshap_medclip/shap_tools/predictor.py†L8-L117】
7. **Explainer y métricas**: `_compute_isa_shap` inicializa `shap.Explainer`, ajusta dinámicamente `max_evals` según el número de tokens/paches y obtiene valores SHAP. Luego calcula **TScore/MM-score** (fracción textual) y **IScore** (fracción visual) para cada muestra.【F:src/mmshap_medclip/tasks/isa.py†L163-L248】【F:src/mmshap_medclip/metrics.py†L6-L120】
8. **Visualización**: `plot_isa` delega en `plot_text_image_heatmaps`, que reconstruye el mapa de calor de parches (interpolado a la resolución original) y colorea tokens según su contribución, generando colorbars coherentes para texto e imagen.【F:src/mmshap_medclip/tasks/isa.py†L125-L160】【F:src/mmshap_medclip/vis/heatmaps.py†L200-L562】

## Componentes clave en detalle
### Wrappers de modelos
- `CLIPWrapper` encapsula modelos CLIP de Hugging Face, exponiendo `processor` y `tokenizer` compatibles con `prepare_batch` y fijando el modo evaluación.【F:src/mmshap_medclip/models.py†L10-L88】
- `OpenCLIPWrapper` se adapta a modelos cargados con `open_clip`, asegurando que la salida se parezca al API de Hugging Face e infiriendo metadatos como `patch_size` e `image_size` para reutilizarlos en SHAP y visualizaciones.【F:src/mmshap_medclip/models.py†L91-L144】
- Las factorías registradas incluyen PubMedCLIP, OpenAI CLIP y WhyXrayCLIP; este último utiliza `open_clip.create_model_and_transforms`, adapta el tokenizador y fija propiedades de visión para operaciones posteriores.【F:src/mmshap_medclip/models.py†L146-L198】

### Dataset ROCO
`RocoDataset` carga el CSV del split deseado directamente desde el ZIP, indexa imágenes por basename priorizando rutas que contienen el split y el subdirectorio `images`, y en `__getitem__` abre la imagen solicitada y devuelve un diccionario con imagen, texto y metadatos (nombre original, ruta interna y split). Esto permite trabajar sin descomprimir el dataset completo.【F:src/mmshap_medclip/datasets/roco.py†L12-L70】

### Métricas multimodales
`compute_mm_score` y `compute_iscore` procesan el vector SHAP concatenado texto+imagen, separando contribuciones absolutas para texto e imagen. El resultado agrega los tokens (combinando sub-palabras) y produce diccionarios ordenados palabra→peso con signo, además de los puntajes escalares TScore/IScore.【F:src/mmshap_medclip/metrics.py†L6-L120】

### Pipeline ISA y explicabilidad
`run_isa_one` y `run_isa_batch` encapsulan toda la experiencia de usuario: preparar datos, ejecutar el modelo, obtener SHAP, calcular métricas y opcionalmente generar visualizaciones. `_compute_isa_shap` maneja automáticamente los detalles del explainer, incluyendo cálculo de `max_evals`, compatibilidad con `attention_mask` y soporte tanto para wrappers de Hugging Face como de OpenCLIP.【F:src/mmshap_medclip/tasks/isa.py†L13-L248】

### Visualizaciones
`plot_text_image_heatmaps` reconstruye la rejilla de parches a partir de la metadata de `make_image_token_ids`, aplica interpolación `nearest` para alinear los valores SHAP con la resolución original, controla la opacidad del overlay y centra los tokens con bounding boxes calculados dinámicamente, añadiendo barras de color separadas para texto e imagen.【F:src/mmshap_medclip/vis/heatmaps.py†L200-L562】

### Utilidades específicas para WhyXrayCLIP
El módulo `whyxrayclip` ofrece funciones orientadas a experimentos clínicos:
- `filter_roco_by_keywords` construye vistas inmutables del dataset filtrando captions por keywords, preservando índices originales para trazabilidad.【F:src/mmshap_medclip/tasks/whyxrayclip.py†L28-L104】
- `pick_chestxray_sample` y `sample_negative_captions` permiten seleccionar muestras relevantes y captions negativas para escenarios zero-shot, con opciones de reproducibilidad.【F:src/mmshap_medclip/tasks/whyxrayclip.py†L106-L179】
- `score_alignment` ejecuta WhyXrayCLIP calculando similitudes normalizadas e, opcionalmente, rankings contra captions negativas usando temperatura configurable.【F:src/mmshap_medclip/tasks/whyxrayclip.py†L181-L242】

## Experimentos listos para ejecutar
El directorio `experiments/` contiene notebooks en formato Jupytext que automatizan la instalación, montaje de Google Drive, carga de configuraciones y ejecución de `run_isa_one`. Sirven como referencia práctica para integrar el pipeline en flujos de Colab.【F:README.md†L47-L170】

## Extensibilidad
Para añadir un nuevo modelo o dataset:
1. Implementar el wrapper o dataset dentro de `src/mmshap_medclip/models.py` o `src/mmshap_medclip/datasets/` respectivamente.
2. Decorar la función fábrica con `@register_model("nombre")` o `@register_dataset("nombre")`.
3. Asegurar que el módulo se importe al inicializar el paquete (vía `__init__.py` o `datasets/__init__.py`).【F:src/mmshap_medclip/models.py†L146-L198】【F:src/mmshap_medclip/datasets/__init__.py†L1-L2】【F:src/mmshap_medclip/registry.py†L7-L63】
4. Referenciar el nuevo nombre en un YAML para que `build_model`/`build_dataset` lo construyan automáticamente.【F:src/mmshap_medclip/registry.py†L47-L63】

## Consideraciones de ejecución
- El pipeline asume que las imágenes ya están normalizadas según las estadísticas CLIP al pasar por el `processor`; al reconstruir visualizaciones se desnormalizan utilizando las constantes `_CLIP_MEAN` y `_CLIP_STD` para mostrar colores reales.【F:src/mmshap_medclip/vis/heatmaps.py†L14-L19】【F:src/mmshap_medclip/vis/heatmaps.py†L452-L466】
- El `Predictor` requiere que el tamaño de imagen sea divisible por el tamaño de parche y ofrece soporte automático para AMP en CUDA, acelerando las evaluaciones en GPU.【F:src/mmshap_medclip/shap_tools/predictor.py†L55-L116】
- `_compute_isa_shap` ajusta `max_evals` del Permutation Explainer a `2 * (tokens + parches) + 1` para garantizar estimaciones estables sin intervención manual.【F:src/mmshap_medclip/tasks/isa.py†L197-L239】

## Relaciones entre módulos
- `tasks/isa` y `tasks/whyxrayclip` dependen de los wrappers de `models` y utilidades de `tasks/utils`, y reutilizan `metrics`, `shap_tools` y `vis` para evaluar y visualizar resultados.
- `registry` actúa como punto central para crear instancias de datasets y modelos descritos en YAML, permitiendo que los experimentos se mantengan declarativos.
- `devices` y `io_utils` proporcionan funciones auxiliares ligeras para compatibilidad con Colab y ejecución en GPU.

Esta documentación resume el flujo interno del repositorio y cómo interactúan sus componentes para soportar experimentos de explicabilidad multimodal en modelos CLIP médicos.
