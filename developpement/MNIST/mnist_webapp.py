import gradio as gr
from PIL import Image
import requests
import io


def recognize_digit(image):
    # Convert to PIL Image necessary if using the API method

    image = Image.fromarray(image.astype("uint8"))
    image = image.convert("1")
    image = image.resize((28, 28))
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    response = requests.post(
        "http://127.0.0.1:5000/predict", data=img_binary.getvalue()
    )
    return response.json()["prediction"]


if __name__ == "__main__":
    gr.Interface(
        fn=recognize_digit,
        inputs="sketchpad",
        outputs="label",
        live=True,
        description="Draw a number on the sketchpad to see the model's prediction.",
    ).launch(debug=True, share=True)
