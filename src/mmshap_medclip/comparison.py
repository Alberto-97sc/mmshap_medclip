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
    
    if verbose:
        print(f"📝 Caption: {caption[:100]}...")
        print()
    
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
    Crea una visualización simplificada con las 4 imágenes con overlays.
    
    Args:
        results: Diccionario con resultados de cada modelo
        image: Imagen original (PIL)
        caption: Caption de la muestra
        sample_idx: Índice de la muestra
    
    Returns:
        Figura de matplotlib
    """
    # Constantes de normalización CLIP
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32)
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32)
    
    # Filtrar modelos con resultados válidos
    valid_results = {k: v for k, v in results.items() if v is not None}
    n_models = len(valid_results)
    
    if n_models == 0:
        print("❌ No hay resultados válidos para mostrar")
        return None
    
    # Organizar en grid
    if n_models <= 2:
        rows, cols = 1, n_models
    else:
        rows = 2
        cols = 2
    
    # Crear figura
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 8 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_models > 1 else axes
    
    fig.suptitle(
        f"🔬 Comparación de Modelos CLIP Médicos - Muestra #{sample_idx}\n\"{caption[:100]}...\"", 
        fontsize=16, fontweight='bold', y=0.98
    )
    
    model_names = list(valid_results.keys())
    
    for idx, (model_name, ax) in enumerate(zip(model_names, axes)):
        result = valid_results[model_name]
        
        # Obtener datos
        shap_values = result.get("shap_values")
        inputs = result.get("inputs")
        text_len = result.get("text_len")
        mm_scores = result.get("mm_scores")
        
        if shap_values is None or inputs is None:
            ax.text(
                0.5, 0.5, f"{model_name}\n(Error en procesamiento)", 
                ha='center', va='center', fontsize=14
            )
            ax.axis('off')
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
        
        # Obtener imagen
        px = inputs["pixel_values"][0].detach().cpu()
        H, W = int(px.shape[-2]), int(px.shape[-1])
        
        # Desnormalizar imagen
        mean = CLIP_MEAN.to(dtype=px.dtype, device=px.device).view(3, 1, 1)
        std = CLIP_STD.to(dtype=px.dtype, device=px.device).view(3, 1, 1)
        img_vis = torch.clamp(px * std + mean, 0, 1).permute(1, 2, 0).numpy()
        
        # Crear heatmap
        n_patches = len(img_vals)
        side = int(np.sqrt(n_patches))
        
        if side * side == n_patches:
            grid_h = grid_w = side
        else:
            grid_h = int(np.sqrt(n_patches))
            grid_w = n_patches // grid_h
        
        # Reshape a grid
        patch_grid = img_vals[:grid_h * grid_w].reshape(grid_h, grid_w)
        
        # Interpolar a tamaño de imagen
        heat_tensor = torch.as_tensor(patch_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        heat_up = F.interpolate(heat_tensor, size=(H, W), mode='nearest').squeeze().numpy()
        
        # Normalización del heatmap
        vmax = np.percentile(np.abs(heat_up), 95)
        vmax = max(vmax, 1e-6)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        
        # Mostrar imagen
        ax.imshow(img_vis, origin='upper', interpolation='nearest')
        
        # Overlay del heatmap
        ax.imshow(
            heat_up, cmap='coolwarm', norm=norm, alpha=0.4, 
            origin='upper', interpolation='nearest'
        )
        
        # Título con métricas
        ax.set_title(
            f"{model_name}\nLogit: {logit:.4f} | TScore: {tscore:.2%} | IScore: {iscore:.2%}", 
            fontsize=13, fontweight='bold', pad=10
        )
        ax.axis('off')
    
    # Ocultar axes no usados
    for idx in range(len(model_names), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
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
            
            # Agregar título del modelo
            fig.suptitle(
                f"{model_name} - Análisis SHAP Detallado\n{caption[:80]}...", 
                fontsize=14, fontweight='bold', y=0.98
            )
            
            plt.show()
            
            # Imprimir métricas
            tscore = result.get('tscore', 0.0)
            iscore = result.get('iscore', 0.0)
            logit = result.get('logit', 0.0)
            print(f"Logit: {logit:.4f} | TScore: {tscore:.2%} | IScore: {iscore:.2%}\n")
            
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

