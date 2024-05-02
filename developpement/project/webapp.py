import io
import gradio as gr
import requests
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

df_image = pd.read_pickle("dataframes/feature-path.pickle")
df_bert = pd.read_pickle("dataframes/feature-title-BERT.pickle")
df_tfidf = pd.read_pickle("dataframes/feature-title-BoW.pickle")


def process_image(image):
    image = Image.fromarray(image.astype("uint8"))
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")

    response = requests.post(
        "http://api:5001/predict_image", data=img_binary.getvalue()
    )
    if response.status_code == 200:
        indices = response.json()
        paths = df_image["path"][indices].tolist()
        # Plot the images
        fig, axs = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5))
        for i, path in enumerate(paths):
            img = Image.open(path)
            axs[i].imshow(img)
            axs[i].axis("off")
        return fig
    else:
        return "Error in API request"


def process_text(text, model):
    json = {
        "text": text,
        "model": model,
    }
    response = requests.post("http://api:5001/predict_text", json=json)
    if response.status_code == 200:
        indices = response.json()
        match model:
            case "bert":
                titles = df_bert["title"][indices].tolist()
                overviews = df_bert["overview"][indices].tolist()
            case "tfidf":
                titles = df_tfidf["title"][indices].tolist()
                overviews = df_bert["overview"][indices].tolist()
            case _:
                raise ValueError("Unsupported model.")
        result = pd.DataFrame(
            data={
                "Rank": list(range(1, len(indices) + 1)),
                "Tilte": titles,
                "Overview": overviews,
            }
        )
        return result


with gr.Blocks() as iface:
    with gr.Tab("Image"):
        input_image = gr.Image()
        output_image = gr.Plot()
        image_submit_button = gr.Button("Submit")
    with gr.Tab("Text"):
        input_text = gr.Textbox(
            placeholder="Write the synopsis of the movie here.",
            label="Synopsis",
        )
        model_selection = gr.Dropdown(["bert", "tfidf"], value="bert")
        output_df = gr.Dataframe(
            headers=["Rank", "Title", "Overview"], interactive=False
        )
        submit_button = gr.Button("Submit")

    image_submit_button.click(
        process_image, inputs=input_image, outputs=output_image
    )
    submit_button.click(
        process_text, inputs=[input_text, model_selection], outputs=output_df
    )


iface.launch(server_name="0.0.0.0", server_port=80)
