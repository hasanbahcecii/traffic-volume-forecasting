from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np 
import torch
import torch.nn as nn
import joblib

# start a fast api app
app = FastAPI(tiltle= "GRU Traffic Prediction API")

class InputData(BaseModel): # 24 hours input data
    sequence: list  # list 24 x 7


# GRU model implementation

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUNet, self).__init__()

        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # output layer: fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, X): # X_shape: (batch_size, seq_len, input_size)
        out, _ = self.gru(X) # out = (batch_size, seq_len, hidden_size)
        last_out = out[:, -1, :] # out = (all, last one, all)
        out = self.fc(last_out)
        return out

# hyperparameters
INPUT_SIZE = 7 
HIDDEN_SIZE = 64  # hidden layers in gru
NUM_LAYERS = 2
OUTPUT_SIZE = 1 # traffic_volume

# gru model
model = GRUNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
model.load_state_dict(torch.load("gru_model.pth"))
model.eval() # evaluation mode

scaler_y = joblib.load("scaler_y.save") # trafic volume scaler

# API endpoint: /predict
# takes a POST request that returns the prediction
app.post("/predict")
def predict(data: InputData):
    """
    with 24 hours input sequence input it makes a traffic volume prediction
    """
    # to numpy array
    input_seq = np.array(data.sequence).astype(np.float32)

    # check the input size
    if input_seq.shape != (24, 7):
        return ("Error: dimension is not correct, expected size (24, 7)")
    
    # pytorch tensor transformation
    input_tensor = torch.tensor(input_seq).unsqueeze(0) # shape: (1, 24, 7)

    # predict
    with torch.no_grad(): # do not calculate grads
        prediction = model(input_tensor)
        prediction = prediction.numpy() # convert to numpy array

    # inverse transform
    prediction_org = scaler_y.inverse_transform(prediction)

    predicted_value = float(prediction_org[0][0])

    return (f"Predicted traffic volume: {predicted_value}")