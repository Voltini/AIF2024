from annoy import AnnoyIndex
import pandas as pd
import torch.utils
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
import torch
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NlpDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.df = df
        self.encodings = tokenizer(
            self.df["overview"].tolist(), truncation=True, padding=True
        )

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx]) for key, val in self.encodings.items()
        }
        item["titles"] = self.df["original_title"][idx]
        item["overviews"] = self.df["overview"][idx]
        return item

    def __len__(self):
        return self.df.shape[0]


class BertClf(nn.Module):
    def __init__(self, distilbert):
        super(BertClf, self).__init__()
        self.distilbert = distilbert
        for name, param in distilbert.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def forward(self, sent_id, mask):
        out = self.distilbert(sent_id, attention_mask=mask)
        logits = out.logits
        attn = out.attentions
        hidden_states = out.hidden_states
        return logits, hidden_states, attn


class BertEmbedder:
    def __init__(self, df) -> None:
        self.preprocessor = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )
        distilbert = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=4,
            output_attentions=True,
            output_hidden_states=True,
        ).to(device)
        self.model = BertClf(distilbert)
        self.dim = 768
        self.dataset = (
            DataLoader(
                NlpDataset(df, self.preprocessor),
                batch_size=16,
                shuffle=True,
            )
            if df is not None
            else None
        )

    def __call__(self, inputs):
        inputs = self.preprocessor(inputs, truncation=True, padding=True)
        return self.model(
            torch.tensor(inputs["input_ids"]),
            torch.tensor(inputs["attention_mask"]),
        )[1][-1][:, 0, :]

    @classmethod
    def load_pretrained(cls, path, df=None, map_location="cpu"):
        embedder = BertEmbedder(df)
        with open(path, "rb") as f:
            embedder.model = torch.load(f, map_location=map_location)
        return embedder

    def save(self, path):
        with self, open(path, "wb") as f:
            torch.save(self.model, f)

    def create_annoy_db(self, path):
        self.model.eval()
        embeddings = []
        overviews = []
        titles = []
        with torch.no_grad():
            for batch in tqdm(self.dataset):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                titles.extend(batch["titles"])
                overviews.extend(batch["overviews"])
                _, emb, _ = self.model(input_ids, mask=attention_mask)
                last_layer_cls = emb[-1][:, 0, :]
                embeddings.extend(
                    last_layer_cls.squeeze(0).squeeze(0).cpu().numpy()
                )
            embeddings = [e for e in embeddings]
        df = pd.DataFrame({"features": embeddings, "path": titles})
        df.to_pickle("feature-path.pickle")

        dim = len(embeddings[0])
        annoy_index = AnnoyIndex(dim, "angular")
        for i, embedding in enumerate(embeddings):
            annoy_index.add_item(i, embedding)

        annoy_index.build(10)
        annoy_index.save(path)


if __name__ == "__main__":
    data = pd.read_csv("movies_metadata.csv")
    data = data[
        data["overview"].notna() & data["original_title"].notna()
    ].reset_index()
    embedder = BertEmbedder.load_pretrained(
        "bert.pth", df=data, map_location=device
    )
    embedder.create_annoy_db("BERT_index.ann")
