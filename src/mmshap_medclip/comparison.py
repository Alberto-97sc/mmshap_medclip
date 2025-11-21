# src/mmshap_medclip/comparison.py
"""
Módulo para comparar múltiples modelos CLIP en las mismas muestras.

Proporciona funciones para cargar múltiples modelos, ejecutar SHAP en todos ellos,
y visualizar los resultados de manera comparativa.
"""

from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
import torch
import torch.nn.functional as F

from mmshap_medclip.registry import build_model
from mmshap_medclip.tasks.isa import run_isa_one
from mmshap_medclip.vis.heatmaps import wrap_text


def load_all_models(device):
    """
    Carga los 4 modelos CLIP médicos principales.

    Args:
        device: Dispositivo de PyTorch (CPU/GPU)

    Returns:
        Diccionario con los modelos cargados {nombre: modelo}
    """
    models = {}
    model_configs = {
        "PubMedCLIP": {
            "name": "pubmedclip-vit-b32",
            "params": {}
        },
        "BioMedCLIP": {
            "name": "biomedclip",
            "params": {
                "model_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            }
        },
        "RCLIP": {
            "name": "rclip",
            "params": {
                "model_name": "kaveh/rclip"
            }
        },
        "WhyXRayCLIP": {
            "name": "whyxrayclip",
            "params": {
                "model_name": "hf-hub:yyupenn/whyxrayclip",
                "tokenizer_name": "ViT-L-14"
            }
        }
    }

    for model_name, config in model_configs.items():
        print(f"🔄 Cargando modelo {model_name}...")
        try:
            model = build_model(config, device=device)
            models[model_name] = model
            print(f"✅ {model_name} cargado exitosamente")
        except Exception as e:
            print(f"❌ Error cargando {model_name}: {e}")
            models[model_name] = None

    return models


def run_shap_on_all_models(
    models: Dict[str, Any],
    sample_idx: int,
    dataset,
    device,
    verbose=True
):
    """
    Ejecuta SHAP en la misma muestra con todos los modelos.

    Args:
        models: Diccionario con los modelos cargados
        sample_idx: Índice de la muestra en el dataset
        dataset: Dataset ROCO
        device: Dispositivo (CPU/GPU)
        verbose: Si True, imprime el progreso

    Returns:
        Tupla (results, image, caption) donde results es un diccionario
        con los resultados de cada modelo
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"🎯 Procesando muestra #{sample_idx}")
        print(f"{'='*60}\n")

    # Obtener muestra
    sample = dataset[sample_idx]
    image, caption = sample['image'], sample['text']

    results = {}

    for model_name, model in models.items():
        if model is None:
            if verbose:
                print(f"⏭️  Saltando {model_name} (no cargado)")
            continue

        if verbose:
            print(f"🔄 Ejecutando SHAP en {model_name}...")

        try:
            # Ejecutar ISA con SHAP
            res = run_isa_one(
                model=model,
                image=image,
                caption=caption,
                device=device,
                explain=True,
                plot=False  # No mostrar plot individual
            )

            results[model_name] = res

            if verbose:
                logit = res['logit']
                tscore = res.get('tscore', 0.0)
                iscore = res.get('iscore', 0.0)
                print(f"✅ {model_name}: logit={logit:.4f} | TScore={tscore:.2%} | IScore={iscore:.2%}")

        except Exception as e:
            if verbose:
                print(f"❌ Error en {model_name}: {e}")
            results[model_name] = None

    if verbose:
        print(f"\n{'='*60}\n")

    return results, image, caption


def plot_comparison_simple(
    results: Dict[str, Any],
    image,
    caption: str,
    sample_idx: int
):
    """
    Crea una visualización completa con imagen + texto para los 4 modelos.

    Args:
        results: Diccionario con resultados de cada modelo
        image: Imagen original (PIL)
        caption: Caption de la muestra
        sample_idx: Índice de la muestra

    Returns:
        Figura de matplotlib
    """
    from matplotlib.colors import Normalize

    # Constantes de normalización CLIP
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32)
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32)

    # Filtrar modelos con resultados válidos
    valid_results = {k: v for k, v in results.items() if v is not None}
    n_models = len(valid_results)

    if n_models == 0:
        print("❌ No hay resultados válidos para mostrar")
        return None

    # Organizar en grid de columnas
    if n_models <= 2:
        cols = n_models
    else:
        cols = 2

    model_names = list(valid_results.keys())

    # Crear figura con GridSpec para tener imagen + texto por modelo
    # Cada modelo tiene 2 filas: imagen (más grande) y texto (más pequeña)
    fig = plt.figure(figsize=(12 * cols, 7 * ((n_models + cols - 1) // cols)))

    # Envolver el caption largo en múltiples líneas SIN truncar
    # Ajustar automáticamente según la longitud del texto
    text_length = len(caption)
    if text_length > 200:
        wrapped_caption = wrap_text(caption, max_width=90, max_lines=6, prefer_long_lines=False)
    elif text_length > 120:
        wrapped_caption = wrap_text(caption, max_width=85, max_lines=4, prefer_long_lines=False)
    else:
        wrapped_caption = wrap_text(caption, max_width=80, max_lines=3, prefer_long_lines=False)

    fig.suptitle(
        f"🔬 Comparación de Modelos CLIP Médicos - Muestra #{sample_idx}\n\n\"{wrapped_caption}\"",
        fontsize=13, fontweight='bold', y=0.995
    )

    # Calcular valores de texto e imagen para normalización global
    all_text_values = []
    all_image_values = []

    for model_name in model_names:
        result = valid_results[model_name]
        mm_scores = result.get("mm_scores")
        shap_values = result.get("shap_values")
        inputs = result.get("inputs")
        text_len = result.get("text_len")

        if mm_scores and shap_values is not None and inputs is not None:
            # Valores de texto
            _, word_shap_dict = mm_scores[0]
            if word_shap_dict:
                word_vals = np.array([word_shap_dict[w] for w in word_shap_dict.keys()])
                all_text_values.append(word_vals)

            # Valores de imagen
            vals = shap_values.values if hasattr(shap_values, "values") else shap_values
            if vals.ndim == 1:
                vals = vals[None, :]
            elif vals.ndim == 3:
                vals = vals[:, 0, :]

            if text_len is not None:
                seq_len = text_len
            else:
                if "attention_mask" in inputs:
                    seq_len = int(inputs["attention_mask"][0].sum().item())
                else:
                    seq_len = vals.shape[1] // 2

            img_vals = vals[0, seq_len:]
            all_image_values.append(img_vals)

    # Normalización global para texto
    text_concat = np.concatenate(all_text_values) if all_text_values else np.zeros((1,))
    if np.any(text_concat < 0):
        absmax_text = float(np.percentile(np.abs(text_concat), 95))
        if absmax_text <= 0:
            absmax_text = float(np.max(np.abs(text_concat))) if text_concat.size else 1e-6
        absmax_text = max(absmax_text, 1e-6)
        norm_text = TwoSlopeNorm(vmin=-absmax_text, vcenter=0.0, vmax=absmax_text)
        cmap_text = plt.get_cmap("coolwarm")
    else:
        vmax_text = float(np.percentile(text_concat, 95)) if text_concat.size else 1e-6
        vmax_text = max(vmax_text, 1e-6)
        norm_text = Normalize(vmin=0.0, vmax=vmax_text)
        cmap_text = plt.get_cmap("Reds")

    # Crear GridSpec UNA VEZ para todos los modelos
    # Cada modelo ocupa 2 filas (imagen + texto)
    num_rows_total = ((n_models + cols - 1) // cols) * 2
    height_ratios = [4, 1] * ((n_models + cols - 1) // cols)

    gs = GridSpec(
        num_rows_total, cols,
        figure=fig,
        hspace=0.35,
        wspace=0.15,
        height_ratios=height_ratios,
        top=0.94,
        bottom=0.03
    )

    # Crear subplots para cada modelo
    for idx, model_name in enumerate(model_names):
        result = valid_results[model_name]

        # Posición en el grid
        col = idx % cols
        row = idx // cols

        # Cada modelo usa 2 filas consecutivas (imagen y texto)
        ax_img = fig.add_subplot(gs[row * 2, col])
        ax_txt = fig.add_subplot(gs[row * 2 + 1, col])

        # Obtener datos
        shap_values = result.get("shap_values")
        inputs = result.get("inputs")
        text_len = result.get("text_len")
        mm_scores = result.get("mm_scores")
        model_wrapper = result.get("model_wrapper")

        if shap_values is None or inputs is None or mm_scores is None:
            ax_img.text(
                0.5, 0.5, f"{model_name}\n(Error en procesamiento)",
                ha='center', va='center', fontsize=14
            )
            ax_img.axis('off')
            ax_txt.axis('off')
            continue

        # Extraer valores SHAP
        vals = shap_values.values if hasattr(shap_values, "values") else shap_values
        if vals.ndim == 1:
            vals = vals[None, :]
        elif vals.ndim == 3:
            vals = vals[:, 0, :]

        # Obtener longitud de texto
        if text_len is not None:
            seq_len = text_len
        else:
            if "attention_mask" in inputs:
                seq_len = int(inputs["attention_mask"][0].sum().item())
            else:
                seq_len = vals.shape[1] // 2

        # Separar valores de imagen
        img_vals = vals[0, seq_len:]

        # Calcular scores
        tscore = result.get('tscore', 0.0)
        iscore = result.get('iscore', 0.0)
        logit = result.get('logit', 0.0)

        # ===== IMAGEN CON HEATMAP =====
        px = inputs["pixel_values"][0].detach().cpu()
        H, W = int(px.shape[-2]), int(px.shape[-1])

        # Desnormalizar imagen
        mean = CLIP_MEAN.to(dtype=px.dtype, device=px.device).view(3, 1, 1)
        std = CLIP_STD.to(dtype=px.dtype, device=px.device).view(3, 1, 1)
        img_vis = torch.clamp(px * std + mean, 0, 1).permute(1, 2, 0).numpy()

        # Crear heatmap de imagen
        n_patches = len(img_vals)
        side = int(np.sqrt(n_patches))

        if side * side == n_patches:
            grid_h = grid_w = side
        else:
            grid_h = int(np.sqrt(n_patches))
            grid_w = n_patches // grid_h

        patch_grid = img_vals[:grid_h * grid_w].reshape(grid_h, grid_w)
        heat_tensor = torch.as_tensor(patch_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        heat_up = F.interpolate(heat_tensor, size=(H, W), mode='nearest').squeeze().numpy()

        # Normalización del heatmap
        vmax = np.percentile(np.abs(heat_up), 95)
        vmax = max(vmax, 1e-6)
        norm_img = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        # Mostrar imagen con overlay
        ax_img.imshow(img_vis, origin='upper', interpolation='nearest')
        ax_img.imshow(
            heat_up, cmap='coolwarm', norm=norm_img, alpha=0.4,
            origin='upper', interpolation='nearest'
        )

        # Título con métricas
        ax_img.set_title(
            f"{model_name}\nLogit: {logit:.4f} | TScore: {tscore:.2%} | IScore: {iscore:.2%}",
            fontsize=13, fontweight='bold', pad=10
        )
        ax_img.axis('off')

        # ===== TEXTO CON PALABRAS COLOREADAS =====
        ax_txt.axis('off')
        ax_txt.set_xlim(0, 1)
        ax_txt.set_ylim(0, 1)

        _, word_shap_dict = mm_scores[0]
        words = list(word_shap_dict.keys())
        word_vals = np.array([word_shap_dict[w] for w in words])

        if len(words) == 0:
            ax_txt.text(0.5, 0.5, "(sin palabras detectadas)",
                       ha='center', va='center', fontsize=10)
            continue

        # Formatear palabras
        def format_word(w):
            w = w.strip() if w else ""
            if len(w) > 15:
                w = w[:14] + "…"
            return w if w else "∅"

        words_display = [format_word(w) for w in words]

        # Calcular anchos con padding para el bbox
        char_width = 0.012  # ancho aproximado por caracter
        bbox_padding = 0.008  # padding del bbox
        widths = [(len(w) * char_width) + bbox_padding for w in words_display]
        gap = 0.02  # Espacio entre palabras

        # Dividir palabras en múltiples líneas si el texto es muy largo
        # Usar el 85% del ancho disponible como límite máximo por línea
        max_width_per_line = 0.85
        max_lines = 3  # Máximo de líneas para el texto coloreado

        # Agrupar palabras en líneas
        lines = []
        current_line_words = []
        current_line_vals = []
        current_line_widths = []
        current_line_width = 0

        for word, val, w in zip(words_display, word_vals, widths):
            word_width_with_gap = w + (gap if current_line_words else 0)

            # Si agregar esta palabra excedería el ancho máximo, empezar una nueva línea
            if current_line_words and (current_line_width + word_width_with_gap) > max_width_per_line:
                if len(lines) < max_lines - 1:  # Reservar espacio para al menos una línea
                    lines.append((current_line_words, current_line_vals, current_line_widths))
                    current_line_words = [word]
                    current_line_vals = [val]
                    current_line_widths = [w]
                    current_line_width = w
                else:
                    # Si ya tenemos el máximo de líneas, agregar a la última línea aunque exceda
                    current_line_words.append(word)
                    current_line_vals.append(val)
                    current_line_widths.append(w)
                    current_line_width += word_width_with_gap
            else:
                current_line_words.append(word)
                current_line_vals.append(val)
                current_line_widths.append(w)
                current_line_width += word_width_with_gap

        # Agregar la última línea
        if current_line_words:
            lines.append((current_line_words, current_line_vals, current_line_widths))

        # Calcular el espacio vertical necesario con mejor espaciado
        # Aumentar significativamente el espaciado entre líneas para evitar superposición
        if len(lines) > 5:
            line_height = 0.32  # Espaciado muy grande para muchas líneas
        elif len(lines) > 3:
            line_height = 0.28  # Espaciado grande para varias líneas
        elif len(lines) > 1:
            line_height = 0.25  # Espaciado medio para múltiples líneas
        else:
            line_height = 0.20  # Espaciado normal para una línea
        total_height = len(lines) * line_height
        start_y = 0.5 + total_height / 2 - line_height / 2

        # Dibujar cada línea de palabras con mejor espaciado uniforme
        for line_idx, (line_words, line_vals, line_widths) in enumerate(lines):
            line_total_w = sum(line_widths) + gap * max(0, len(line_words)-1)
            start_x = 0.5 - line_total_w / 2
            x = start_x
            # Calcular posición Y con espaciado uniforme entre líneas
            y = start_y - line_idx * line_height

            for word, val, w in zip(line_words, line_vals, line_widths):
                color = cmap_text(norm_text(val))
                ax_txt.text(
                    x, y, word,
                    ha="left", va="center", fontsize=11, color="black",
                    transform=ax_txt.transAxes,
                    bbox=dict(
                        facecolor=color,
                        alpha=0.8,
                        edgecolor="white",
                        linewidth=0.5,
                        boxstyle="square,pad=0.3"
                    )
                )
                x += w + gap

    return fig


def plot_individual_heatmaps(results: Dict[str, Any], image, caption: str):
    """
    Genera heatmaps individuales detallados para cada modelo.

    Args:
        results: Diccionario con resultados de cada modelo
        image: Imagen original (PIL)
        caption: Caption de la muestra
    """
    from mmshap_medclip.vis.heatmaps import plot_text_image_heatmaps

    valid_results = {k: v for k, v in results.items() if v is not None}

    for model_name, result in valid_results.items():
        print(f"\n{'='*60}")
        print(f"🔍 Heatmap detallado: {model_name}")
        print(f"{'='*60}\n")

        shap_values = result.get("shap_values")
        mm_scores = result.get("mm_scores")
        inputs = result.get("inputs")
        text_len = result.get("text_len")
        model_wrapper = result.get("model_wrapper")

        if any(x is None for x in [shap_values, mm_scores, inputs, model_wrapper]):
            print(f"⚠️  Datos incompletos para {model_name}")
            continue

        try:
            # Generar heatmap individual sin modificaciones
            fig = plot_text_image_heatmaps(
                shap_values=shap_values,
                inputs=inputs,
                tokenizer=model_wrapper.tokenizer,
                images=image,
                texts=[caption],
                mm_scores=mm_scores,
                model_wrapper=model_wrapper,
                return_fig=True,
                text_len=text_len,
            )

            plt.show()

            # Imprimir métricas en consola
            tscore = result.get('tscore', 0.0)
            iscore = result.get('iscore', 0.0)
            logit = result.get('logit', 0.0)
            print(f"📊 {model_name} - Logit: {logit:.4f} | TScore: {tscore:.2%} | IScore: {iscore:.2%}\n")

        except Exception as e:
            print(f"❌ Error generando heatmap para {model_name}: {e}\n")


def print_summary(results: Dict[str, Any]):
    """
    Imprime un resumen comparativo de todos los modelos.

    Args:
        results: Diccionario con resultados de cada modelo
    """
    print("\n" + "="*80)
    print("📊 RESUMEN COMPARATIVO".center(80))
    print("="*80 + "\n")

    # Tabla de resultados
    print(f"{'Modelo':<20} {'Logit':>10} {'TScore':>10} {'IScore':>10}")
    print("-" * 80)

    for model_name, result in results.items():
        if result is None:
            print(f"{model_name:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        else:
            logit = result.get('logit', 0.0)
            tscore = result.get('tscore', 0.0)
            iscore = result.get('iscore', 0.0)
            print(f"{model_name:<20} {logit:>10.4f} {tscore:>9.2%} {iscore:>9.2%}")

    print("\n" + "="*80 + "\n")

    # Análisis de balance multimodal
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        iscores = [v.get('iscore', 0.0) for v in valid_results.values()]
        tscores = [v.get('tscore', 0.0) for v in valid_results.values()]

        avg_iscore = np.mean(iscores)
        avg_tscore = np.mean(tscores)

        print("🎯 Balance Multimodal Promedio:")
        print(f"   • IScore promedio: {avg_iscore:.2%}")
        print(f"   • TScore promedio: {avg_tscore:.2%}")

        # Identificar modelo más balanceado (más cercano a 50/50)
        balance_diffs = [abs(0.5 - v.get('iscore', 0.0)) for v in valid_results.values()]
        most_balanced_idx = np.argmin(balance_diffs)
        most_balanced_model = list(valid_results.keys())[most_balanced_idx]

        print(f"\n🏆 Modelo más balanceado: {most_balanced_model}")
        print(f"   (IScore más cercano a 50%)")


def save_comparison(
    results: Dict[str, Any],
    image,
    caption: str,
    sample_idx: int,
    output_dir: str = "outputs"
):
    """
    Guarda la comparación en disco.

    Args:
        results: Resultados de los modelos
        image: Imagen original
        caption: Caption de la muestra
        sample_idx: Índice de la muestra
        output_dir: Directorio donde guardar
    """
    from pathlib import Path
    import json

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Guardar figura
    fig = plot_comparison_simple(results, image, caption, sample_idx)
    if fig is not None:
        fig_path = output_path / f"comparison_sample_{sample_idx}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"💾 Figura guardada en: {fig_path}")
        plt.close(fig)

    # Guardar resultados numéricos
    summary = {}
    for model_name, result in results.items():
        if result is not None:
            summary[model_name] = {
                "logit": float(result.get('logit', 0.0)),
                "tscore": float(result.get('tscore', 0.0)),
                "iscore": float(result.get('iscore', 0.0)),
            }

    json_path = output_path / f"comparison_sample_{sample_idx}.json"
    with open(json_path, 'w') as f:
        json.dump({
            "sample_idx": sample_idx,
            "caption": caption,
            "results": summary
        }, f, indent=2)

    print(f"💾 Resultados guardados en: {json_path}")


def analyze_multiple_samples(
    models: Dict[str, Any],
    dataset,
    device,
    sample_indices: list,
    verbose=False
):
    """
    Analiza múltiples muestras y retorna estadísticas agregadas.

    Args:
        models: Diccionario con los modelos cargados
        dataset: Dataset ROCO
        device: Dispositivo
        sample_indices: Lista de índices a analizar
        verbose: Si True, imprime progreso detallado

    Returns:
        DataFrame con resultados agregados
    """
    import pandas as pd

    all_results = []

    print(f"🔄 Analizando {len(sample_indices)} muestras...")

    for idx in sample_indices:
        if verbose:
            print(f"\n📍 Procesando muestra {idx}...")
        else:
            print(f".", end="", flush=True)

        results, _, caption = run_shap_on_all_models(
            models=models,
            sample_idx=idx,
            dataset=dataset,
            device=device,
            verbose=verbose
        )

        for model_name, result in results.items():
            if result is not None:
                all_results.append({
                    "sample_idx": idx,
                    "model": model_name,
                    "logit": result.get('logit', 0.0),
                    "tscore": result.get('tscore', 0.0),
                    "iscore": result.get('iscore', 0.0),
                    "caption": caption[:50] + "..."
                })

    if not verbose:
        print()  # Nueva línea después de los puntos

    df = pd.DataFrame(all_results)

    print("\n✅ Análisis completado")
    print("\n📊 Estadísticas por modelo:")
    print(df.groupby('model')[['logit', 'tscore', 'iscore']].mean().round(4))

    return df
