# src/mmshap_medclip/registry.py
import importlib

_REG_MODELS = {}
_REG_DATASETS = {}

def register_model(name):
    def deco(fn): _REG_MODELS[name] = fn; return fn
    return deco

def register_dataset(name):
    def deco(fn): _REG_DATASETS[name] = fn; return fn
    return deco

def build_dataset(cfg):
    name = cfg["name"]
    if name not in _REG_DATASETS:
        # intenta importar mmshap_medclip.datasets.<name> (p.ej., roco)
        try:
            importlib.import_module(f"mmshap_medclip.datasets.{name}")
        except ModuleNotFoundError:
            # fallback: importar el paquete datasets (que puede importar todo)
            importlib.import_module("mmshap_medclip.datasets")
    fn = _REG_DATASETS.get(name)
    if fn is None:
        raise KeyError(f"Dataset '{name}' no registrado.")
    return fn(cfg.get("params", {}))
