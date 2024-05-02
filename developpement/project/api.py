from flask import Flask, jsonify, request
from PIL import Image
import torch
import io
from annoy import AnnoyIndex
from helpers import search
from prep_BERT import BertEmbedder, BertClf
from prep_BoW import BagOfWordsEmbedder, StemTokenizer
from prep_image import ImageEmbedder

device = torch.device("cpu")

app = Flask(__name__)
image_embedder = ImageEmbedder.load_pretrained("models/model.pth")
image_embedder.model.eval()
image_index = AnnoyIndex(image_embedder.dim, "angular")
image_index.load("annoy_indices/img_index.ann")

bert_embedder = BertEmbedder.load_pretrained("models/bert.pth")
bert_embedder.model.eval()
bert_index = AnnoyIndex(bert_embedder.dim, "angular")
bert_index.load("annoy_indices/BERT_index.ann")

bow_embedder = BagOfWordsEmbedder.load_pretrained("models/tfidf.pickle")
bow_index = AnnoyIndex(bow_embedder.dim, "angular")
bow_index.load("annoy_indices/BoW_index.ann")


@app.route("/predict_image", methods=["POST"])
def predict_image():
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary))
    with torch.no_grad():
        embeddings = image_embedder(img_pil).squeeze()
    pred = search(image_index, embeddings)
    return jsonify(pred)


@app.route("/predict_text", methods=["POST"])
def predict_text():
    req = request.json
    text = req["text"]
    model = req["model"]
    if model == "bert":
        embeddings = bert_embedder([text]).squeeze()
        pred = search(bert_index, embeddings)
    elif model == "tfidf":
        embeddings = bow_embedder([text])
        pred = search(bow_index, embeddings[0])
    else:
        raise ValueError("Unsupported choice")
    return jsonify(pred)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
