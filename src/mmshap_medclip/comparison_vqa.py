# src/mmshap_medclip/comparison_vqa.py
"""
Módulo para comparar múltiples modelos CLIP en VQA-Med 2019.

Proporciona funciones para cargar modelos (PubMedCLIP y BiomedCLIP), ejecutar SHAP en todos ellos,
y visualizar los resultados de manera comparativa.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
import torch
import torch.nn.functional as F

from mmshap_medclip.registry import build_model
from mmshap_medclip.tasks.vqa import run_vqa_one
from mmshap_medclip.vis.heatmaps import wrap_text, resize_patch_grid_for_display


def load_vqa_models(device):
    """
    Carga solo PubMedCLIP y BiomedCLIP para VQA.

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


def run_vqa_shap_on_models(
    models: Dict[str, Any],
    sample_idx: int,
    dataset,
    device,
    target_logit: str = "correct",
    verbose=True
):
    """
    Ejecuta VQA+SHAP en la misma muestra con todos los modelos.

    Args:
        models: Diccionario con los modelos cargados
        sample_idx: Índice de la muestra en el dataset
        dataset: Dataset VQA-Med 2019
        device: Dispositivo (CPU/GPU)
        target_logit: "correct" o "predicted" - qué logit explicar
        verbose: Si True, imprime el progreso

    Returns:
        Tupla (results, image, question, answer, candidates) donde results es un diccionario
        con los resultados de cada modelo
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"🎯 Procesando muestra VQA #{sample_idx}")
        print(f"{'='*60}\n")

    # Obtener muestra
    sample = dataset[sample_idx]
    image = sample['image']
    question = sample['question']
    answer = sample.get('answer')
    # Usar SOLO los candidatos del dataset, sin modificar ni filtrar
    candidates = sample.get('candidates', [])
    category = sample.get('category', 'unknown')
    
    # Validar que los candidatos no estén vacíos
    if not candidates:
        error_msg = (
            f"⚠️  ERROR: Muestra {sample_idx} tiene lista de candidatos vacía. "
            f"Categoría: {category}. "
            f"Esto no debería ocurrir si el dataset está correctamente construido."
        )
        if verbose:
            print(error_msg)
        raise ValueError(error_msg)

    if verbose:
        print(f"📝 Pregunta: {question}")
        print(f"📋 Categoría: {category}")
        print(f"✅ Respuesta correcta: {answer}")
        print(f"📊 Candidatos: {len(candidates)} opciones\n")

    results = {}

    for model_name, model in models.items():
        if model is None:
            if verbose:
                print(f"⏭️  Saltando {model_name} (no cargado)")
            continue

        if verbose:
            print(f"🔄 Ejecutando VQA+SHAP en {model_name}...")

        try:
            # Ejecutar VQA con SHAP
            res = run_vqa_one(
                model=model,
                image=image,
                question=question,
                candidates=candidates,
                device=device,
                answer=answer,
                explain=True,
                plot=False,
                target_logit=target_logit
            )

            results[model_name] = res

            if verbose:
                prediction = res.get('prediction', 'N/A')
                correct = res.get('correct', None)
                tscore = res.get('tscore', 0.0)
                iscore = res.get('iscore', 0.0)
                correct_str = "✅" if correct else "❌" if correct is False else "?"
                print(f"✅ {model_name}: Predicción={prediction} {correct_str} | TScore={tscore:.2%} | IScore={iscore:.2%}")

            # Limpiar memoria GPU después de cada modelo
            if device.type == "cuda":
                torch.cuda.empty_cache()
                # Forzar sincronización para liberar memoria inmediatamente
                torch.cuda.synchronize()

        except Exception as e:
            if verbose:
                print(f"❌ Error en {model_name}: {e}")
                import traceback
                traceback.print_exc()
            results[model_name] = None
            
            # Limpiar memoria GPU incluso si hay error
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if verbose:
        print(f"\n{'='*60}\n")

    return results, image, question, answer, candidates, category


def plot_vqa_comparison(
    results: Dict[str, Any],
    image,
    question: str,
    answer: Optional[str],
    candidates: List[str],
    sample_idx: int
):
    """
    Crea una visualización completa con imagen + pregunta para los modelos VQA.

    Args:
        results: Diccionario con resultados de cada modelo
        image: Imagen original (PIL)
        question: Pregunta de la muestra
        answer: Respuesta correcta (opcional)
        candidates: Lista de candidatos
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
    cols = n_models
    model_names = list(valid_results.keys())

    # Crear figura con GridSpec
    fig = plt.figure(figsize=(13 * cols, 7.5 * n_models))

    # Crear GridSpec
    num_rows_total = n_models * 2
    height_ratios = [4, 1] * n_models

    gs = GridSpec(
        num_rows_total, cols,
        figure=fig,
        hspace=0.35,
        wspace=0.15,
        height_ratios=height_ratios,
        top=0.94,
        bottom=0.08
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
        prediction = result.get('prediction', 'N/A')
        correct = result.get('correct', None)

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

        patch_grid, _, _ = resize_patch_grid_for_display(patch_grid, target_size=7)

        heat_tensor = torch.as_tensor(patch_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        heat_up = F.interpolate(heat_tensor, size=(H, W), mode='nearest').squeeze().numpy()

        # Normalización del heatmap
        vmax = np.percentile(np.abs(heat_up), 95)
        vmax = max(vmax, 1e-6)
        norm_img = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        alpha_consistent = 0.40

        # Mostrar imagen con overlay
        ax_img.imshow(img_vis, origin='upper', interpolation='nearest')
        ax_img.imshow(
            heat_up, cmap='coolwarm', norm=norm_img, alpha=alpha_consistent,
            origin='upper', interpolation='nearest'
        )

        # Título con métricas
        correct_str = "✅" if correct else "❌" if correct is False else "?"
        ax_img.set_title(
            f"{model_name}\nPredicción: {prediction} {correct_str}\nTScore: {tscore:.2%} | IScore: {iscore:.2%}",
            fontsize=13, fontweight='bold', pad=10
        )
        ax_img.axis('off')
        
        # Colorbar vertical por modelo para imagen
        img_pos = ax_img.get_position()
        cax_img = fig.add_axes([
            img_pos.x1 + 0.01,
            img_pos.y0,
            0.012,
            img_pos.height
        ])
        fig.colorbar(
            plt.cm.ScalarMappable(cmap='coolwarm', norm=norm_img),
            cax=cax_img,
            label="Valor SHAP (imagen)"
        )

        # ===== TEXTO CON PALABRAS COLOREADAS =====
        ax_txt.axis('off')
        ax_txt.set_xlim(0, 1)
        ax_txt.set_ylim(0, 1)

        _, word_shap_dict = mm_scores[0]
        words = list(word_shap_dict.keys())
        word_vals = np.array([word_shap_dict[w] for w in words], dtype=np.float32)
        
        if np.any(word_vals < 0):
            absmax_word = float(np.percentile(np.abs(word_vals), 95)) if word_vals.size else 0.0
            absmax_word = max(absmax_word, 1e-6)
            norm_text_local = TwoSlopeNorm(vmin=-absmax_word, vcenter=0.0, vmax=absmax_word)
            cmap_text_local = plt.get_cmap("coolwarm")
        else:
            vmax_word = float(np.percentile(word_vals, 95)) if word_vals.size else 0.0
            vmax_word = max(vmax_word, 1e-6)
            norm_text_local = Normalize(vmin=0.0, vmax=vmax_word)
            cmap_text_local = plt.get_cmap("Reds")

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

        # Calcular anchos
        char_width = 0.012
        bbox_padding = 0.008
        widths = [(len(w) * char_width) + bbox_padding for w in words_display]
        gap = 0.02

        # Dividir palabras en múltiples líneas
        max_width_per_line = 0.85
        max_lines = 3

        lines = []
        current_line_words = []
        current_line_vals = []
        current_line_widths = []
        current_line_width = 0

        for word, val, w in zip(words_display, word_vals, widths):
            word_width_with_gap = w + (gap if current_line_words else 0)

            if current_line_words and (current_line_width + word_width_with_gap) > max_width_per_line:
                if len(lines) < max_lines - 1:
                    lines.append((current_line_words, current_line_vals, current_line_widths))
                    current_line_words = [word]
                    current_line_vals = [val]
                    current_line_widths = [w]
                    current_line_width = w
                else:
                    current_line_words.append(word)
                    current_line_vals.append(val)
                    current_line_widths.append(w)
                    current_line_width += word_width_with_gap
            else:
                current_line_words.append(word)
                current_line_vals.append(val)
                current_line_widths.append(w)
                current_line_width += word_width_with_gap

        if current_line_words:
            lines.append((current_line_words, current_line_vals, current_line_widths))

        # Calcular espacio vertical
        if len(lines) > 5:
            line_height = 0.32
        elif len(lines) > 3:
            line_height = 0.28
        elif len(lines) > 1:
            line_height = 0.25
        else:
            line_height = 0.20
        total_height = len(lines) * line_height
        start_y = 0.5 + total_height / 2 - line_height / 2

        # Dibujar cada línea
        for line_idx, (line_words, line_vals, line_widths) in enumerate(lines):
            line_total_w = sum(line_widths) + gap * max(0, len(line_words)-1)
            start_x = 0.5 - line_total_w / 2
            x = start_x
            y = start_y - line_idx * line_height

            for word, val, w in zip(line_words, line_vals, line_widths):
                color = cmap_text_local(norm_text_local(val))
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
        
        # Colorbar horizontal por modelo para texto
        txt_pos = ax_txt.get_position()
        cax_txt = fig.add_axes([
            txt_pos.x0,
            txt_pos.y0 - 0.03,
            txt_pos.width,
            0.012
        ])
        fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap_text_local, norm=norm_text_local),
            cax=cax_txt,
            orientation="horizontal",
            label="Valor SHAP (texto)"
        )

    return fig


def print_vqa_summary(results: Dict[str, Any]):
    """
    Imprime un resumen comparativo de todos los modelos en VQA.

    Args:
        results: Diccionario con resultados de cada modelo
    """
    print("\n" + "="*80)
    print("📊 RESUMEN COMPARATIVO VQA".center(80))
    print("="*80 + "\n")

    # Tabla de resultados
    print(f"{'Modelo':<20} {'Predicción':<20} {'Correcto':<10} {'TScore':>10} {'IScore':>10}")
    print("-" * 80)

    for model_name, result in results.items():
        if result is None:
            print(f"{model_name:<20} {'N/A':<20} {'N/A':<10} {'N/A':>10} {'N/A':>10}")
        else:
            prediction = result.get('prediction', 'N/A')
            correct = result.get('correct', None)
            tscore = result.get('tscore', 0.0)
            iscore = result.get('iscore', 0.0)
            correct_str = "✅" if correct else "❌" if correct is False else "?"
            print(f"{model_name:<20} {prediction[:18]:<20} {correct_str:<10} {tscore:>9.2%} {iscore:>9.2%}")

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

        # Identificar modelo más balanceado
        balance_diffs = [abs(0.5 - v.get('iscore', 0.0)) for v in valid_results.values()]
        most_balanced_idx = np.argmin(balance_diffs)
        most_balanced_model = list(valid_results.keys())[most_balanced_idx]

        print(f"\n🏆 Modelo más balanceado: {most_balanced_model}")
        print(f"   (IScore más cercano a 50%)")

