# Import Flask class
from flask import Flask, request, render_template, make_response

from linearRegression import linearRegression

# Create Flask application instance
app = Flask(__name__,template_folder="/home/prabhakar/PycharmProject/House_Price_Prediction/template")


# Define a route (URL path) and a view function
@app.route('/model', methods=["POST"])
def LinearRegression():
    NumberOfRooms = request.form['NumberOfRooms']
    NumberOfBedrooms = request.form['NumberOfBedrooms']
    HouseAge = request.form['HouseAge']
    AvgAreaIncome = request.form['AvgAreaIncome']
    AreaPopulation = request.form['AreaPopulation']
    predictedPrice = linearRegression(NumberOfRooms=int(NumberOfRooms), NumberOfBedrooms=int(NumberOfBedrooms), HouseAge=int(HouseAge),
                                      AvgAreaIncome=int(AvgAreaIncome), AreaPopulation=int(AreaPopulation))
    return make_response(predictedPrice)

@app.route('/',methods=["GET"])
def home():
    return render_template("index.html")


# Another route~e
@app.route('/about')
def about():
    return render_template("about.html")


# Run the app (start the Flask development server)
if __name__ == "__main__":
    app.run(debug=True)
