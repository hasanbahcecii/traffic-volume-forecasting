import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# hyperparameters
BATCH_SIZE = 64
INPUT_SIZE = 7 
HIDDEN_SIZE = 64  # hidden layers in gru
NUM_LAYERS = 2
OUTPUT_SIZE = 1 # traffic_volume
LEARNING_RATE = 0.001
NUM_EPOCHS = 5

# load data
X_train = np.load("X_train.npy") # shape: (num of examples, 24, 7)
y_train = np.load("y_train.npy") # shape: (num of examples, 1)  1 = target value(traffic_volume)

# convert numpy to tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

# data loader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size= BATCH_SIZE, shuffle= True)

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


# gru model
model = GRUNet(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    
# loss function and optimization
criterion = nn.MSELoss() # mean square entopy loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# train loop
loss_list = [] # keep loss for each epoch 

for epoch in range(NUM_EPOCHS):
    model.train() # train mode
    epoch_loss = 0

    for X_batch, y_batch in train_loader:

        # generate estimates
        outputs = model(X_batch)

        # calculate loss
        loss = criterion(outputs, y_batch)

        # backpropagation
        optimizer.zero_grad()
        loss.backward() # calculate gradients
        optimizer.step() # update parameters

        # total loss
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)    
    loss_list.append(avg_loss)
    print(f"Epoch: [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss: .4f}")

    
# loss graph
plt.figure()
plt.plot(loss_list, marker = "o")
plt.title("Train Loss Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# save the model
torch.save(model.state_dict(), "gru_model.pth")
print("Model saved successfully.")