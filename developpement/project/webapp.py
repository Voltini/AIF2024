import io
import gradio as gr
import requests
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

df = pd.read_pickle("feature-path.pickle")


def process_image(image):
    image = Image.fromarray(image.astype("uint8"))
    # image.resize(224, 224)
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")

    response = requests.post(
        "http://api:5001/predict_image", data=img_binary.getvalue()
    )
    if response.status_code == 200:
        indices = response.json()
        paths = df["path"][indices].tolist()
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
    server_name="0.0.0.0", server_port=80
)  # the server will be accessible externally under this address
