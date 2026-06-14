from flask import Flask, render_template, request
import numpy as np
import joblib
from PIL import Image

app = Flask(__name__)

model = joblib.load("savedmodel.pth")

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        image = Image.open(request.files["file"])

        image = image.convert("L")
        image = image.resize((64, 64))

        image = np.array(image).flatten().reshape(1, -1)
        image = image / 255.0

        prediction = model.predict(image)[0]

    return render_template(
        "index.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
