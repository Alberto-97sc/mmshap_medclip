# src/mmshap_medclip/registry.py
import importlib

_REG_MODELS = {}
_REG_DATASETS = {}

def register_model(name: str):
    def deco(fn):
        _REG_MODELS[name] = fn
        return fn
    return deco

def register_dataset(name: str):
    def deco(fn):
        _REG_DATASETS[name] = fn
        return fn
    return deco

def _ensure_model(name: str):
    if name in _REG_MODELS:
        return
    # intenta cargar definiciones de modelos
    try:
        importlib.import_module("mmshap_medclip.models")
    except ModuleNotFoundError:
        pass
    # opcional: si luego separas en submódulos por nombre
    try:
        importlib.import_module(f"mmshap_medclip.models_impl.{name}")
    except ModuleNotFoundError:
        pass

def _ensure_dataset(name: str):
    if name in _REG_DATASETS:
        return
    # intenta importar el dataset concreto (p.ej. mmshap_medclip.datasets.roco)
    try:
        importlib.import_module(f"mmshap_medclip.datasets.{name}")
    except ModuleNotFoundError:
        pass
    # fallback: importar el paquete que a su vez importa todos
    try:
        importlib.import_module("mmshap_medclip.datasets")
    except ModuleNotFoundError:
        pass

def build_model(cfg: dict, device=None):
    name = cfg["name"]
    _ensure_model(name)
    fn = _REG_MODELS.get(name)
    if fn is None:
        raise KeyError(f"Modelo '{name}' no registrado.")
    params = dict(cfg.get("params", {}))
    params["_device"] = device
    return fn(params)

def build_dataset(cfg: dict):
    name = cfg["name"]
    _ensure_dataset(name)
    fn = _REG_DATASETS.get(name)
    if fn is None:
        raise KeyError(f"Dataset '{name}' no registrado.")
    return fn(cfg.get("params", {}))
