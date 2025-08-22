import zipfile
import pandas as pd
from PIL import Image
from io import BytesIO

from .base import DatasetBase
from mmshap_medclip.registry import register_dataset

@register_dataset("roco")
def _build_roco(params):
    return RocoDataset(**params)

class RocoDataset(DatasetBase):
    def __init__(self, zip_path: str, split: str, columns: dict, images_subdir: str = None, n_rows="all"):
        self.zip_path = zip_path
        self.split = split
        self.image_key = columns["image_key"]
        self.caption_key = columns["caption_key"]
        self.images_subdir = images_subdir
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_name = next(n for n in zf.namelist() if split in n.lower() and n.lower().endswith(".csv"))
            self.df = pd.read_csv(zf.open(csv_name)) if n_rows=="all" else pd.read_csv(zf.open(csv_name), nrows=int(n_rows))
            self._zf_names = set(zf.namelist())

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fname = row[self.image_key]
        caption = row[self.caption_key]
        img_path = self.images_subdir + "/" + fname if self.images_subdir else fname
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            with zf.open(img_path) as f:
                image = Image.open(BytesIO(f.read())).convert("RGB")
        return {"image": image, "text": caption, "meta": {"filename": fname, "split": self.split}}
