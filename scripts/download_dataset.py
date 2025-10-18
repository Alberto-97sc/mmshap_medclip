#!/usr/bin/env python3
"""
Script para descargar el dataset ROCO desde Google Drive usando gdown.
Uso: python scripts/download_dataset.py
"""

import os
import sys
from pathlib import Path

def download_dataset():
    """Descarga el dataset ROCO desde Google Drive."""
    
    print("📥 Descargando dataset ROCO desde Google Drive...")
    
    # Crear directorio data si no existe
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # ID del archivo en Google Drive
    dataset_id = "1eRUC8F8PtXffa9iArJnyB8AMqlPNoSwc"
    output_file = data_dir / "dataset_roco.zip"
    
    print(f"🔗 ID del archivo: {dataset_id}")
    print(f"📁 Destino: {output_file}")
    
    try:
        import gdown
        
        # Descargar usando gdown
        print("📥 Iniciando descarga con gdown...")
        gdown.download(
            id=dataset_id,
            output=str(output_file),
            quiet=False
        )
        
        # Verificar descarga
        if output_file.exists() and output_file.stat().st_size > 1000:  # Más de 1KB
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"✅ Dataset descargado exitosamente!")
            print(f"📊 Tamaño del archivo: {size_mb:.1f} MB")
            print("🎉 ¡Listo para usar en los experimentos!")
            return True
        else:
            print("❌ Error: El archivo descargado es muy pequeño o está vacío")
            return False
            
    except ImportError:
        print("❌ Error: gdown no está instalado")
        print("💡 Instala gdown con: pip install gdown")
        return False
    except Exception as e:
        print(f"❌ Error durante la descarga: {e}")
        return False

def main():
    """Función principal."""
    print("=" * 60)
    print("📥 Descargador de Dataset ROCO")
    print("=" * 60)
    print()
    
    success = download_dataset()
    
    if not success:
        print("\n💡 Alternativas:")
        print("   1. Descargar manualmente desde:")
        print("      https://drive.google.com/file/d/1eRUC8F8PtXffa9iArJnyB8AMqlPNoSwc/view")
        print("   2. Usar el script bash: bash scripts/download_dataset.sh")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
