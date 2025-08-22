from torch.utils.data import Dataset

class DatasetBase(Dataset):
    @classmethod
    def from_config(cls, cfg: dict):
        return cls(**cfg)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError
