import os
import re
import zipfile
from io import BytesIO

import pandas as pd
from PIL import Image

from mmshap_medclip.datasets.base import DatasetBase
from mmshap_medclip.registry import register_dataset

def _apply_caption_filter(dataset, caption_regex: str, casefold: bool = True):
    if not caption_regex:
        return

    if not hasattr(dataset, "df") or not hasattr(dataset, "caption_key"):
        raise AttributeError(
            "El dataset de ROCO no expone 'df' o 'caption_key'; no se puede aplicar el filtro de captions."
        )

    flags = re.IGNORECASE if casefold else 0
    pattern = re.compile(caption_regex, flags)
    captions = dataset.df[dataset.caption_key].astype(str)
    mask = captions.map(lambda text: bool(pattern.search(text)))
    dataset.df = dataset.df[mask].reset_index(drop=True)


@register_dataset("roco")
def _build_roco(params):
    params = dict(params)
    columns = dict(params.pop("columns")) if "columns" in params else {}

    images_subdir = params.pop("images_subdir", None)
    if images_subdir is None:
        images_subdir = columns.pop("images_subdir", None)

    params["columns"] = columns
    if images_subdir is not None:
        params["images_subdir"] = images_subdir

    caption_regex = params.pop("caption_regex", None)
    casefold = params.pop("casefold", True)

    try:
        dataset = RocoDataset(caption_regex=caption_regex, casefold=casefold, **params)
    except TypeError:
        dataset = RocoDataset(**params)
        _apply_caption_filter(dataset, caption_regex, casefold)
    else:
        if not getattr(dataset, "_caption_pattern", None) and caption_regex:
            _apply_caption_filter(dataset, caption_regex, casefold)

    return dataset

class RocoDataset(DatasetBase):
    def __init__(
        self,
        zip_path: str,
        split: str,
        columns: dict,
        images_subdir: str = None,
        n_rows="all",
        caption_regex: str = None,
        casefold: bool = True,
    ):
        self.zip_path = zip_path
        self.split = split
        self.image_key = columns["image_key"]
        self.caption_key = columns["caption_key"]
        self.images_subdir = images_subdir
        self._caption_pattern = re.compile(caption_regex, re.IGNORECASE if casefold else 0) if caption_regex else None

        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_name = next(n for n in zf.namelist() if split in n.lower() and n.lower().endswith(".csv"))
            self.df = pd.read_csv(zf.open(csv_name)) if n_rows=="all" else pd.read_csv(zf.open(csv_name), nrows=int(n_rows))

            if self._caption_pattern is not None:
                mask = self.df[self.caption_key].astype(str).map(lambda x: bool(self._caption_pattern.search(x)))
                self.df = self.df[mask].reset_index(drop=True)

            # índice basename -> ruta completa (preferimos las que tienen /images/ y el split)
            self._name_to_path = {}
            for n in zf.namelist():
                if n.endswith("/") or not n.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                base = os.path.basename(n)
                score = int("/images/" in n.lower()) + int(self.split in n.lower())
                prev = self._name_to_path.get(base)
                if prev is None or score > prev[0]:
                    self._name_to_path[base] = (score, n)
            # quedarnos solo con la ruta
            self._name_to_path = {k: v[1] for k, v in self._name_to_path.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fname = row[self.image_key]
        caption = row[self.caption_key]

        with zipfile.ZipFile(self.zip_path, "r") as zf:
            path = None

            # 1) si se dio images_subdir, prueba esa ruta literal
            if self.images_subdir:
                candidate = f"{self.images_subdir.rstrip('/')}/{fname}"
                if candidate in zf.namelist():
                    path = candidate

            # 2) si no, usa el índice por basename
            if path is None:
                path = self._name_to_path.get(os.path.basename(fname))

            if path is None:
                # 3) último intento: búsqueda por sufijo
                candidates = [
                    n
                    for n in zf.namelist()
                    if n.lower().endswith("/" + fname.lower())
                    or os.path.basename(n).lower() == fname.lower()
                ]
                if candidates:
                    path = candidates[0]

            if path is None:
                raise KeyError(f"No encontré '{fname}' dentro del ZIP. Revisa images_subdir en YAML o la estructura interna del ZIP.")

            with zf.open(path) as f:
                image = Image.open(BytesIO(f.read())).convert("RGB")

        return {"image": image, "text": caption, "meta": {"filename": fname, "zip_path": path, "split": self.split}}
