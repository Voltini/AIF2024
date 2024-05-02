import torch
import torchvision.models as models
import pandas as pd
import torchvision.transforms as transforms
from annoy import AnnoyIndex
from torch.utils.data import DataLoader
from helpers import Embedder, ImageAndPathsDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageEmbedder(Embedder):
    def __init__(self, prod=True) -> None:
        self.dim = 576
        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        self.model = torch.nn.Sequential(
            mobilenet.features, mobilenet.avgpool, torch.nn.Flatten()
        ).to(device)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean, std)
        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
        )
        self.preprocessor = transform
        self.dataset = None
        if not prod:
            dataset = ImageAndPathsDataset("../MLP-20M", self.preprocessor)
            self.dataset = DataLoader(dataset, batch_size=128, shuffle=True)

    def __call__(self, inputs):
        ready = self.preprocessor(inputs)
        if ready.dim() == 3:
            ready = ready.unsqueeze(0)
        return self.model(ready)

    @classmethod
    def load_pretrained(cls, path, map_location="cpu", prod=True):
        embedder = cls(prod=prod)
        with open(path, "rb") as f:
            embedder.model = torch.load(f, map_location=map_location)
        return embedder

    def save(self, path):
        with self, open(path, "wb") as f:
            torch.save(self.model, f)

    def create_annoy_db(self, path):
        self.model.eval()
        features_list = []
        paths_list = []
        for imgs, paths in self.dataset:
            with torch.no_grad():
                embeddings = self.model(imgs.to(device))
                features_list.extend(embeddings.cpu().numpy())
                paths_list.extend(paths)

        df = pd.DataFrame({"features": features_list, "path": paths_list})
        df.to_pickle("dataframes/feature-path.pickle")

        dim = self.dim
        annoy_index = AnnoyIndex(dim, "angular")
        for i, embedding in enumerate(features_list):
            annoy_index.add_item(i, embedding)

        annoy_index.build(10)
        annoy_index.save(path)


if __name__ == "__main__":
    embedder = ImageEmbedder.load_pretrained(
        "models/model.pth", map_location="cuda", prod=False
    )
    embedder.create_annoy_db("annoy_indices/img_index.ann")
