import dash
import mlflow
import os
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pickle
import pandas as pd
import numpy as np

app = dash.Dash(__name__)



#Set mlflow tracking uri
mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th/")
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"
model_name = "st124984-a3-nipa"
model_version = 1

# loading the models
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

app.layout = html.Div(
    id='form-container',
    children=[
        html.H1('Welcome to Car Price Prediction 2025 - Ver.3', style={'padding-top':'20px', 'color':'green', 'font-style':'bold', 'text-decoration':'underline', 'text-align':'center', 'padding-bottom':'10px'}),
        html.P('Instructions for using this Web Application.', style={'font-weight':'bold'}),
        html.P("Step 1: Navigate to the Prediction Section from the Navigation bar at the top. Click on the tab named PredictV1: Old Model or PredictV2: New Model or PredictV3: New Version Model with Classification to open the prediction form."),
        html.P("Step 2: The form will ask for the following details about the car. Enter the information in the respective fields:"),
        html.Ul([
            html.Li("Year: Enter the manufacturing year of the car."),
            html.Li("Kilometers Driven: Enter the total distance the car has been driven (in kilometers)."),
            html.Li("Mileage: Enter the mileage of the car in kmpl (kilometers per liter)."),
            html.Li("Engine: Enter the engine capacity in cc (cubic centimeters)."),
        ]),
        html.P("Step 3: Once all fields are filled, click the Submit button at the bottom of the form."),
        html.P("View the Prediction Classification:"),
        html.Ul([
            html.Li("After submission, the application will process the data."),
            html.Li("The predicted car price will be displayed on the screen."),
            html.Li("Review the prediction to get an estimate of your car's price based on the provided details."),
        ]),
        html.Label('Enter the values in the relative field', style={'font-weight': 'bold', 'font-size': '16px'}),
        html.Br(),
        html.Br(),
        html.Label('Engine:', style={'font-weight': 'bold', 'font-size': '14px'}),
        html.Br(),
        dcc.Input(id='engine', type='number', placeholder='e.g., 1197', style={
            'width': '100%',
            'padding': '8px',
            'margin-top': '5px',
            'margin-bottom': '20px',
            'border': '1px solid #ccc',
            'border-radius': '5px',
            'box-sizing': 'border-box',
            'font-size': '14px'
        }),
        html.Br(),
        html.Label('Mileage:', style={'font-weight': 'bold', 'font-size': '14px'}),
        html.Br(),
        dcc.Input(id='mileage', type='number', placeholder='e.g., 15.5', style={
            'width': '100%',
            'padding': '8px',
            'margin-top': '5px',
            'margin-bottom': '20px',
            'border': '1px solid #ccc',
            'border-radius': '5px',
            'box-sizing': 'border-box',
            'font-size': '14px'
        }),
         html.Br(),
        html.Label('Km_driven:', style={'font-weight': 'bold', 'font-size': '14px'}),
        html.Br(),
        dcc.Input(id='km_driven', type='number', placeholder='e.g., 45000', style={
            'width': '100%',
            'padding': '8px',
            'margin-top': '5px',
            'margin-bottom': '20px',
            'border': '1px solid #ccc',
            'border-radius': '5px',
            'box-sizing': 'border-box',
            'font-size': '14px'
        }),
        html.Br(),
        html.Label('Year:', style={'font-weight': 'bold', 'font-size': '14px'}),
        html.Br(),
        dcc.Input(id='year', type='number', placeholder='e.g., 2015', style={
            'width': '100%',
            'padding': '8px',
            'margin-top': '5px',
            'margin-bottom': '20px',
            'border': '1px solid #ccc',
            'border-radius': '5px',
            'box-sizing': 'border-box',
            'font-size': '14px'
        }),
        html.Br(),
        html.Button('Submit', id='submit', n_clicks=0, style={
            'background-color': '#007bff',
            'color': 'white',
            'padding': '10px 15px',
            'border': 'none',
            'border-radius': '5px',
            'cursor': 'pointer',
            'font-size': '16px',
            'display': 'block',
            'margin': '20px 0'
        }),
        html.Div(id='output-predict', style={
            'font-size': '16px',
            'margin-top': '20px',
            'color': '#333'
        })
    ],
    style={
        'margin': 'auto',
        'width': '50%',
        'padding': '20px',
        'border': '2px solid #f0f0f0',
        'border-radius': '10px',
        'background-color': '#f9f9f9',
        'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.6)'
    }
)


    
def prediction(year: float, km_driven: float, mileage: float ,engine: float):
    try:
        print("Type of np:", type(np))
        data = np.array([[year, km_driven, mileage, engine]])
        prediction = model.predict(data)
        return prediction
    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")



def getDefaultValue():
    try:
        df = pd.read_csv(os.path.join(os.getcwd(), "Cars.csv"))
        df['owner'] = df['owner'].map({
            "First Owner": 1,
            "Second Owner": 2,
            "Third Owner": 3,
            "Fourth & Above Owner": 4,
            "Test Drive Car": 5
        })
        df = df[(df['fuel'] != 'CNG') & (df['fuel'] != 'LPG')]
        df['mileage'] = df['mileage'].str.split().str[0].astype(float)
        df['engine'] = df['engine'].str.split().str[0].str.replace('CC', '').astype(float)
        df['max_power'] = df['max_power'].str.replace('bhp', '').str.extract(r'(\d+\.?\d*)').astype(float)
        df['name'] = df['name'].str.split().str[0]
        df = df.drop(columns=['torque'])
        df = df[df['owner'] != 5]

        median_engine = df['engine'].median()
        median_year = df['year'].median()
        mean_mileage = df['mileage'].mean()
        median_km_driven = df['km_driven'].median()
        return median_year, median_engine, mean_mileage, median_km_driven
    except Exception as e:
        raise ValueError(f"Error in processing data: {str(e)}")
    
def get_X(user_year, user_km_driven, user_mileage, user_engine):
    default_year, default_engine, default_mileage , default_km_driven = getDefaultValue()
            
    user_year = user_year if user_year else default_year
    user_engine = user_engine if user_engine else default_engine
    user_mileage = user_mileage if user_mileage else default_mileage
    user_km_driven = user_km_driven if user_km_driven else default_km_driven

    return user_year, user_km_driven, user_mileage, user_engine

@app.callback(
    Output('output-predict', 'children'),
    [Input('submit', 'n_clicks')],
    [State('year', 'value'),
     State('km_driven', 'value'),
     State('mileage', 'value'),
     State('engine', 'value')]
)


def update_output(n_clicks, user_year, user_km_driven, user_mileage, user_engine):
    prediction_label = {0: "Cheap", 1: "Affordable", 2: "Expensive", 3: "Very Expensive"}
    try:
        if n_clicks > 0:
            user_year, user_km_driven, user_mileage, user_engine = get_X(user_year, user_km_driven, user_mileage, user_engine)
            pred_val = prediction(float(user_year), float(user_km_driven), float(user_mileage), float(user_engine))
            if not isinstance(pred_val, (list, np.ndarray)) or len(pred_val) == 0:
                raise ValueError("Invalid prediction output")
            pred_class = int(pred_val[0])
            return f"Predicted Price: {prediction_label.get(pred_class, 'Unknown')}"
    except Exception as e:
        return f"Error in prediction: {str(e)}"  # Removed "Mank" typo
    return 'Click "Submit" to view the predicted price.'



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8050)

