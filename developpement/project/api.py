from flask import Flask, jsonify, request
import pandas as pd
from PIL import Image
import torch
import io
import model as model
from annoy import AnnoyIndex
from model import search

device = torch.device("cpu")

app = Flask(__name__)
MODEL_PATH = "model.pth"
INDEX_PATH = "annoy_index.ann"
DF_PATH = "./feature-path.pickle"

index = AnnoyIndex(576, "angular")
index.load(INDEX_PATH)
transform = model.transform()
model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
df = pd.read_pickle(DF_PATH)


@app.route("/predict", methods=["POST"])
def predict():
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary))
    tensor = transform(img_pil).to(device)
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor).squeeze()
    pred = search(df, index, outputs)
    return jsonify(pred)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
