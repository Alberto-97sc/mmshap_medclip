_REG_MODELS = {}
_REG_DATASETS = {}

def register_model(name):
    def deco(fn):
        _REG_MODELS[name] = fn
        return fn
    return deco

def register_dataset(name):
    def deco(fn):
        _REG_DATASETS[name] = fn
        return fn
    return deco

def build_model(cfg, device=None):
    fn = _REG_MODELS[cfg["name"]]
    params = dict(cfg.get("params", {}))
    params["_device"] = device
    return fn(params)

def build_dataset(cfg):
    fn = _REG_DATASETS[cfg["name"]]
    return fn(cfg.get("params", {}))


