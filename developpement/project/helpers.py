from typing import Any
from torchvision import datasets
from torch.utils.data import Dataset


class ImageAndPathsDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        img, _ = super(ImageAndPathsDataset, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, path

    def __len__(self) -> int:
        return super().__len__()


class Embedder:
    model: Any
    preprocessor: Any
    dataset: Dataset
    dim: int

    def prepare(self): ...

    def __call__(self, inputs):
        self.model(inputs)

    @classmethod
    def load_pretrained(cls, path): ...

    def save(self, path): ...

    def create_annoy_db(self, path, *args): ...


def search(annoy_index, query_vector, k=5):
    indices = annoy_index.get_nns_by_vector(query_vector, k)
    return indices
