import torch
import torchvision.models as models
import pandas as pd

from annoy import AnnoyIndex
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from helpers import ImageAndPathsDataset, transform, NlpDataset

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
    dataloader = DataLoader(dataset, batch_size=128, num_workers=2, shuffle=False)
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


def create_annoy_db_text():
    train_df = pd.read_csv("data/train.csv", names=["label", "title", "text"]).sample(
        40000
    )
    train_text, train_labels = train_df["text"], train_df["labels"] - 1
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    dataset = NlpDataset(train_text, train_labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=2, shuffle=False)
    features_list = []
    names_list = []
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=4
    ).to(device)
    for x, name in dataloader:
        with torch.no_grad():
            embeddings = model(x.to(device))
            features_list.extend(embeddings.cpu().numpy())
            names_list.extend(name)

    df = pd.DataFrame({"features": features_list, "names": names_list})
    df.to_pickle("feature-names.pickle")

    dim = len(features_list[0])
    annoy_index = AnnoyIndex(dim, "angular")
    for i, embedding in enumerate(features_list):
        annoy_index.add_item(i, embedding)

    annoy_index.build(10)
    annoy_index.save("annoy_index_text.ann")


if __name__ == "__main__":
    # create_model()
    # create_annoy_db()
    create_annoy_db_text()
