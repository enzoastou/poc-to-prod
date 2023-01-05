from flask import Flask, request, render_template

from run import TextPredictionModel

app = Flask(__name__)


@app.route('/')
def root():
    htmlString="""
    <form name = "Tag Prediction" action="." method="POST">
    <label>Insert forum title/text to predict: </label>
    <input name="text" value="Example: C# print syntax"></input>
    <button type="submit">Predict</button>
    </form>
"""
    return htmlString


@app.route("/", methods=['POST'])
def retrieve_prediction():
    model = TextPredictionModel.from_artefacts(
        "D:\\EPF\\5A\\POCtoPROD\\poc-to-prod-capstone\\train\\data\\artefacts\\2023-01-04-22-31-58")
    text = request.form['text']
    predictions = model.predict([text], top_k=1)

    return str(predictions)


@app.route("/predict", methods=["GET"])
def request_prediction():
    model = TextPredictionModel.from_artefacts("train/data/artefacts/train/2023-01-04-17-25-24")
    text = request.args.get('text')
    predictions = model.predict([text], top_k=1)

    return str(predictions)


if __name__ == '__main__':
    app.run()