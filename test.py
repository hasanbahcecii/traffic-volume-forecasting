import numpy as np 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error # evaluation metrics

# GRU model class
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
SEQ_LEN = 24 # 24 hours
INPUT_SIZE = 7 # feature
HIDDEN_SIZE = 64  # hidden layers in gru
NUM_LAYERS = 2
OUTPUT_SIZE = 1 # traffic_volume


# load test data
X_test = np.load("X_test.npy") # input: (num_examples, 24, 7)
y_test = np.load("y_test.npy") # target: (num_examples, 1)  1 = traffic_volume

# numpy to torch
X_test_tensor = torch.tensor(X_test, dtype= torch.float32)

# create model
model = GRUNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

# load model
model.load_state_dict(torch.load("gru_model.pth"))
model.eval() # evaluation mode

# model prediction
with torch.no_grad(): # do not calculate gradient at the prediction phase
    predictions = model(X_test_tensor) # prediction

predictions = predictions.numpy() # tensor to numpy array

# inverse transform
scaler_y = joblib.load("scaler_y.save")

# transform normalized predictions to their original version
predictions_org = scaler_y.inverse_transform(predictions)

# same thing for y_test values
y_test_org = scaler_y.inverse_transform(y_test)

# evaluation metrics
rmse = root_mean_squared_error(y_test_org, predictions_org)
mae = mean_absolute_error(y_test_org, predictions_org)

print(f"rmse: {rmse}")
print(f"mae: {mae}")

# visualization
plt.figure(figsize=(15,5))
plt.plot(y_test_org[:200], label="Real Values", color= "blue")
plt.plot(predictions_org[:200], label="Prediction Values", color= "orange")
plt.title("Traffic Volume Prediction - GRU")
plt.xlabel("Time")
plt.ylabel("Traffic Volume")
plt.legend()
plt.show()