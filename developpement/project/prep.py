import torch
import torchvision.models as models
import pandas as pd

from annoy import AnnoyIndex
from torch.utils.data import DataLoader
from helpers import ImageAndPathsDataset, transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model():
    mobilenet = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    model = torch.nn.Sequential(
        mobilenet.features, mobilenet.avgpool, torch.nn.Flatten()
    ).to(device)
    torch.save(model, "model.pth")


def create_annoy_db():
    dataset = ImageAndPathsDataset("../MLP-20M", transform())
    dataloader = DataLoader(
        dataset, batch_size=128, num_workers=2, shuffle=False
    )
    features_list = []
    paths_list = []
    with open("model.pth", "rb") as model_file:
        model = torch.load(model_file)
    for x, paths in dataloader:
        with torch.no_grad():
            embeddings = model(x.to(device))
            features_list.extend(embeddings.cpu().numpy())
            paths_list.extend(paths)

    df = pd.DataFrame({"features": features_list, "path": paths_list})
    df.to_pickle("feature-path.pickle")

    dim = len(features_list[0])
    annoy_index = AnnoyIndex(dim, "angular")
    for i, embedding in enumerate(features_list):
        annoy_index.add_item(i, embedding)

    annoy_index.build(10)
    annoy_index.save("annoy_index.ann")


if __name__ == "__main__":
    create_model()
    create_annoy_db()
