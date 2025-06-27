from flask import Flask,request,render_template
import pickle

application=Flask(__name__)
app=application

model = pickle.load(open('Boston_Decision_Tree.pkl', 'rb'))

features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM','AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

features_full = {
    'CRIM': 'Per capita crime rate by town',
    'ZN': 'Proportion of residential land zoned for large lots',
    'INDUS': 'Proportion of non-retail business acres per town',
    'CHAS': 'Charles River dummy variable (1 if bounds river, else 0)',
    'NOX': 'Nitric oxides concentration (ppm)',
    'RM': 'Average number of rooms per dwelling',
    'AGE': 'Proportion of owner-occupied units built before 1940',
    'DIS': 'Weighted distance to employment centers',
    'RAD': 'Index of accessibility to radial highways',
    'TAX': 'Full-value property tax rate per $10,000',
    'PTRATIO': 'Pupil-teacher ratio by town',
    'B': '1000(Bk - 0.63)^2 (proportion of Black residents)',
    'LSTAT': 'Percentage of lower status population'
}

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            input_values = [float(request.form[f]) for f in features]
            prediction = round(model.predict([input_values])[0], 2)
        except Exception as e:
            prediction = f"Error: {str(e)}"
    return render_template('predict.html', features=features, features_full=features_full, prediction=prediction)

if __name__=='__main__':
    app.run(debug=True)