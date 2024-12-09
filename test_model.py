import torch
from model import LightMNIST, train_model, count_parameters

def test_parameter_count():
    model = LightMNIST()
    param_count = count_parameters(model)
    assert param_count < 25000, f"Model has {param_count} parameters, should be less than 25000"

def test_model_accuracy():
    model, accuracy = train_model()
    print(f"Model accuracy is {accuracy}%")
    assert accuracy >= 95.0, f"Model accuracy is {accuracy}%, should be at least 95%" 