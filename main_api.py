from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np 
import torch
import random
import torch.nn as nn
import joblib

# start a fast api app
app = FastAPI(title="GRU Traffic Prediction API")

class InputData(BaseModel): # 24 hours input data
    sequence: list  # list 24 x 7

# GRU model implementation
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X): # X_shape: (batch_size, seq_len, input_size)
        out, _ = self.gru(X)
        last_out = out[:, -1, :]
        out = self.fc(last_out)
        return out

# hyperparameters
INPUT_SIZE = 7 
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1

# load model
model = GRUNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
model.load_state_dict(torch.load("gru_model.pth", map_location="cpu"))
model.eval()


#load scaler 
scaler_X = joblib.load("scaler_X.save") 
scaler_y = joblib.load("scaler_y.save")


# API endpoint
@app.post("/predict")
def predict(data: InputData):

    """
        It makes a traffic volume prediction by using the 24 hours input sequence.
    """
    input_seq = np.array(data.sequence).astype(np.float32)

    if input_seq.shape != (24, 7):
        return {"error": "Dimension is not correct, expected size (24, 7)"}
     # normalize input with scaler_X 
    input_seq_scaled = scaler_X.transform(input_seq)
    input_tensor = torch.tensor(input_seq_scaled).unsqueeze(0) # (1, 24, 7)

    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = prediction.detach().cpu().numpy()

    prediction_org = scaler_y.inverse_transform(prediction)
    predicted_value = float(prediction_org[0][0])

    return {"predicted_traffic_volume": predicted_value}

# test generator
sample_sequence = []
for i in range(24):
    temp = random.uniform(280, 300) # kelvin 
    rain = 0
    snow = 0
    clouds = random.randint(0, 100)
    hour = 1
    dayofweek = random.randint(0, 6)
    month = 10
    sample_sequence.append([temp, rain, snow, clouds, hour, dayofweek, month])

print(sample_sequence)    