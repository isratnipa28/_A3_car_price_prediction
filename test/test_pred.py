

import os
import cloudpickle
import numpy as np

base_path = os.path.dirname(os.path.abspath(__file__))

model = cloudpickle.load(open(os.path.join(base_path, "../model/carprice_prediction_a3.model"), 'rb'))

def test_model_accepts_input():
    """Confirm that the model accepts input and executes without issues."""
    try:
        data = np.array([[2022, 3500, 50, 2500]])
        np.exp(model.predict(data))
        passed = True
    except Exception as e:
        passed = False
    assert passed, "The model encountered an error due to incompatible input format."


def test_model_output_shape():
    """Verify that the model's output shape is (1,)"""
    data = np.array([[2022, 3500, 50, 2500]])
    prediction = np.exp(model.predict(data))
    assert prediction.shape == (1,),f"Expected shape (1,), but got {prediction.shape}"