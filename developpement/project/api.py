from flask import Flask, jsonify, request
from PIL import Image
import torch
import io
import helpers as helpers
from annoy import AnnoyIndex
from helpers import search

device = torch.device("cpu")

app = Flask(__name__)
MODEL_PATH = "model.pth"
INDEX_PATH = "rec_imdb.ann"
DF_PATH = "./feature-path.pickle"

index = AnnoyIndex(576, "angular")
index.load(INDEX_PATH)
transform = helpers.transform()
model = torch.load(MODEL_PATH, map_location=torch.device("cpu")).eval()


@app.route("/predict", methods=["POST"])
def predict():
    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary))
    tensor = transform(img_pil).to(device)
    tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor).squeeze()
    pred = search(index, outputs)
    return jsonify(pred)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
