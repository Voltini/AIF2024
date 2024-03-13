import io
import gradio as gr
import requests
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

DF_PATH = "./feature-path.pickle"
df = pd.read_pickle(DF_PATH)


def process_image(image):
    image = Image.fromarray(image.astype("uint8"))
    # image.resize(224, 224)
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")

    response = requests.post(
        "http://localhost:5000/predict", data=img_binary.getvalue()
    )
    if response.status_code == 200:
        paths = response.json()

        # Plot the images
        fig, axs = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5))
        for i, path in enumerate(paths):
            img = Image.open(path)
            axs[i].imshow(img)
            axs[i].axis("off")
        return fig
    else:
        return "Error in API request"


iface = gr.Interface(fn=process_image, inputs="image", outputs="plot")
iface.launch(
    server_name="0.0.0.0"
)  # the server will be accessible externally under this address
