# app.py
from flask import Flask, request, render_template
from model import predict_best_model

app = Flask(__name__, template_folder="/home/prabhakar/PycharmProject/House_Price_Prediction/template")


@app.route('/', methods=["GET", "POST"])
def home():
    predictedPrice = None
    model_used = None

    if request.method == "POST":
        NumberOfRooms = float(request.form['NumberOfRooms'])
        NumberOfBedrooms = float(request.form['NumberOfBedrooms'])
        HouseAge = float(request.form['HouseAge'])
        AvgAreaIncome = float(request.form['AvgAreaIncome'])
        AreaPopulation = float(request.form['AreaPopulation'])

        predictedPrice, model_used = predict_best_model(NumberOfRooms, NumberOfBedrooms, HouseAge, AvgAreaIncome,
                                                        AreaPopulation)

    return render_template("index.html", predictedPrice=predictedPrice, model_used=model_used)


if __name__ == "__main__":
    app.run(debug=True)
