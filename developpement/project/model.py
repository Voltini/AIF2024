import torchvision.models as models
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from annoy import AnnoyIndex

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageAndPathsDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        img, _ = super(ImageAndPathsDataset, self).__getitem__(index)
        path = self.imgs[index][0]
        return img, path

    def __len__(self) -> int:
        return super().__len__()


def create_model():
    mobilenet = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    model = torch.nn.Sequential(
        mobilenet.features, mobilenet.avgpool, torch.nn.Flatten()
    ).to(device)
    torch.save(model, "model.pth")


def transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )
    return transform


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


def search(df, annoy_index, query_vector, k=5):
    indices = annoy_index.get_nns_by_vector(query_vector, k)
    paths = df["path"][indices]
    return paths.tolist()


if __name__ == "__main__":
    create_model()
    create_annoy_db()
