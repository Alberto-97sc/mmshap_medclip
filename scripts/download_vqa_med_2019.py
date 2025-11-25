#!/usr/bin/env python3
"""
Script para descargar el dataset VQA-Med 2019 desde Google Drive usando gdown.
Descarga solo el ZIP principal (VQA-Med-2019.zip) sin extraer archivos.
El dataset loader accederá directamente al Training.zip dentro del ZIP principal.
Uso: python scripts/download_vqa_med_2019.py
"""

import os
import sys
from pathlib import Path

def download_dataset():
    """Descarga el dataset VQA-Med 2019 desde Google Drive (solo el ZIP principal)."""

    print("📥 Descargando dataset VQA-Med 2019 desde Google Drive...")

    # Crear directorio data si no existe
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # ID del archivo en Google Drive
    # Link: https://drive.google.com/file/d/1Xu_Y2Z6lvZGgExxz0VdYY6wFgAFk3oZf/view?usp=sharing
    dataset_id = "1Xu_Y2Z6lvZGgExxz0VdYY6wFgAFk3oZf"
    main_zip = data_dir / "VQA-Med-2019.zip"  # ZIP principal

    print(f"🔗 ID del archivo: {dataset_id}")
    print(f"📁 ZIP principal: {main_zip}")
    print("ℹ️  El dataset loader accederá directamente al Training.zip dentro del ZIP principal")

    try:
        import gdown

        # Descargar ZIP principal usando gdown
        print("📥 Iniciando descarga del ZIP principal con gdown...")
        gdown.download(
            id=dataset_id,
            output=str(main_zip),
            quiet=False
        )

        # Verificar descarga del ZIP principal
        if not main_zip.exists() or main_zip.stat().st_size < 1000:
            print("❌ Error: El archivo descargado es muy pequeño o está vacío")
            return False

        size_mb = main_zip.stat().st_size / (1024 * 1024)
        print(f"✅ ZIP principal descargado exitosamente!")
        print(f"📊 Tamaño del archivo: {size_mb:.1f} MB")
        print("🎉 ¡Listo para usar en los experimentos!")
        print("ℹ️  El dataset loader accederá al Training.zip dentro de este ZIP")

        return True

    except ImportError:
        print("❌ Error: gdown no está instalado")
        print("💡 Instala gdown con: pip install gdown")
        return False
    except Exception as e:
        print(f"❌ Error durante la descarga: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal."""
    print("=" * 60)
    print("📥 Descargador de Dataset VQA-Med 2019")
    print("=" * 60)
    print()

    success = download_dataset()

    if not success:
        print("\n💡 Alternativas:")
        print("   1. Descargar manualmente desde:")
        print("      https://drive.google.com/file/d/1Xu_Y2Z6lvZGgExxz0VdYY6wFgAFk3oZf/view?usp=sharing")
        print("   2. Usar gdown directamente:")
        print("      gdown 1Xu_Y2Z6lvZGgExxz0VdYY6wFgAFk3oZf -O data/VQA-Med-2019.zip")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
