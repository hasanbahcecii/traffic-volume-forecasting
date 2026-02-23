import numpy as np 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        print("FORWARD INPUT SHAPE:", X.shape)

        out, _ = self.gru(X)
        print("GRU OUTPUT SHAPE:", out.shape)

        last_out = out[:, -1, :]
        print("LAST OUT SHAPE:", last_out.shape)

        out = self.fc(last_out)
        print("FC OUTPUT SHAPE:", out.shape)

        return out


SEQ_LEN = 24
INPUT_SIZE = 7
HIDDEN_SIZE = 64
NUM_LAYERS = 2
OUTPUT_SIZE = 1

# load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

print("X_test numpy shape:", X_test.shape)
print("y_test numpy shape:", y_test.shape)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
print("X_test tensor shape:", X_test_tensor.shape)

model = GRUNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

model.load_state_dict(torch.load("gru_model.pth"))
model.eval()

with torch.no_grad():
    predictions = model(X_test_tensor)

print("PREDICTIONS SHAPE:", predictions.shape)

predictions = predictions.numpy()

scaler_y = joblib.load("scaler_y.save")

print("Predictions before inverse:", predictions.shape)
print("y_test before inverse:", y_test.shape)

predictions_org = scaler_y.inverse_transform(predictions)
y_test_org = scaler_y.inverse_transform(y_test)

rmse = root_mean_squared_error(y_test_org, predictions_org)
mae = mean_absolute_error(y_test_org, predictions_org)

print(f"rmse: {rmse}")
print(f"mae: {mae}")

plt.figure(figsize=(15,5))
plt.plot(y_test_org[:200], label="Real Values", color="blue")
plt.plot(predictions_org[:200], label="Prediction Values", color="orange")
plt.title("Traffic Volume Prediction - GRU")
plt.xlabel("Time")
plt.ylabel("Traffic Volume")
plt.legend()
plt.show()