//https://drive.google.com/drive/folders/1osrck82G_gDLl-aqsI-9Jm-DzmcBAMHl?usp=drive_link(files)


from flask import Flask, render_template, request
import numpy as np
import requests
from joblib import load

app = Flask(__name__)

# Load model and vectorizer
rf = load("compressmodel.joblib")
vectorizer = load("vectorizer_compressed.joblib")

@app.route("/", methods=["GET", "POST"])
def predict_news():
    result = ""

    if request.method == "POST":
        news_text = request.form['news'].lower()

        # ML prediction
        text_vect = vectorizer.transform([news_text])
        prdct = rf.predict(text_vect)[0]

        if prdct == 0:
            result = "Fake news (ML prediction)"
        else:
            result = "Real news (ML prediction)"

        # Optional API verification
        url = "https://newsdata.io/api/1/latest"
        params = {
            "apikey": "pub_c8b49d885fa74b248b80aefa03b11a70",
            "q": news_text,
            "language": "en",
        }

        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                api_data = response.json()
                if api_data['status'] == 'success' and api_data['totalResults'] > 0:
                    result += " + Verified by API"
        except:
            pass

        return render_template("index.html", prediction=result, user_input=news_text)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
