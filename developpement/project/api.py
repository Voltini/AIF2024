from flask import Flask, jsonify, request
from PIL import Image
import torch
import io
import helpers as helpers
from annoy import AnnoyIndex
from helpers import search

device = torch.device("cpu")

app = Flask(__name__)
metadata = {
    "image": ["model_image.pth", "rec_imdb.ann", 576, helpers.transform()],
    "text": ["model_text.pth", "rec_imdb.ann", 0, helpers.tokenizer()],
}


def loader(metadata, key):
    index = AnnoyIndex(metadata[key][2], "angular")
    index.load(metadata[key][1])
    transform = metadata[key][3]
    if key == "image":
        model = torch.load(metadata[key][0], map_location=torch.device("cpu")).eval()
    elif key == "text":
        model = torch.load(metadata[key][0], map_location=torch.device("cpu")).eval()
    else:
        raise ValueError("Invalid key")
    return transform, model, index


@app.route("/predict_image", methods=["POST"])
def predict_image():
    transform, model, index = loader(metadata, "image")
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary))
    tensor = transform(img_pil).to(device)
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor).squeeze()
    pred = search(index, outputs)
    return jsonify(pred)


@app.route("/predict_text", methods=["POST"])
def predict_text():
    transform, model, index = loader(metadata, "text")
    txt_binary = request.data
    tensor = transform(
        txt_binary,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor).squeeze()
    pred = search(index, outputs)
    return jsonify(pred)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
