# rewriting sphynx.py in pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

## -------- model architecture -------- ##
class BranchNetwork(nn.Module):
    ''' Branch network for one position '''
    def __init__(self, input_size, output_size):
        super(BranchNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

class SphinxModel(nn.Module):
    ''' Sphinx architecture model '''
    def __init__(self, input_size, num_branches, output_size):
        super(SphinxModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)

        # Creating multiple branches
        self.branches = nn.ModuleList([BranchNetwork(128, output_size) for _ in range(num_branches)])

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        outputs = [branch(x) for branch in self.branches]
        return torch.stack(outputs, dim=1)
    
def load_data(data_path):
    x = np.load(f'data/x_{data_path}.npy')
    y = np.load(f'data/y_{data_path}.npy')

    # Convert numpy arrays to PyTorch tensors
    x = torch.tensor(x, dtype=torch.float32)
    y_class_indices = [np.argmax(y[:, i, :], axis=1) for i in range(num_branches)]

    # Convert to PyTorch tensors
    y = [torch.tensor(y_indices, dtype=torch.long) for y_indices in y_class_indices]

    # split into train and validation and test
    len_data = len(x)
    train_split = int(0.8 * len_data)
    val_split = int(0.9 * len_data)

    x_train, x_val, x_test = x[:train_split], x[train_split:val_split], x[val_split:]
    y_train, y_val, y_test = y[:train_split], y[train_split:val_split], y[val_split:]

    return x_train, y_train, x_val, y_val, x_test, y_test

def create_model(input_size, num_branches, output_size):
    '''Initiates the model'''
    model = SphinxModel(input_size=input_size, num_branches=num_branches, output_size=output_size).to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion
    
def train(batch_size, epochs, model, optimizer, criterion, x_train, y_train, num_branches):
    ''' Trains the model '''
    model.train()
    try:
        for epoch in range(epochs):
            for i in range(0, len(x_train), batch_size):
                end_idx = min(i + batch_size, len(x_train))

                # Move data to the device
                x_batch = x_train[i:end_idx].to(device)
                y_batch = [y_train[j][i:end_idx].to(device) for j in range(num_branches)]

                optimizer.zero_grad()

                # Forward pass
                outputs = model(x_batch)

                # Calculate and backpropagate loss
                loss = sum([criterion(outputs[:, j], y_batch[j]) for j in range(num_branches)])
                loss.backward()

                optimizer.step()
            print(f'Epoch {epoch+1}, Batch {i+1}: loss {loss.item():.3f}')
    
    except KeyboardInterrupt:
        print('Interrupted, saving model...')

    ## --------- Save the model --------- ##
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), os.path.join('models', f'model_{model_name}.pt'))

if __name__ == '__main__':

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

     ## --------- Create and compile the model --------- ##
    N2 = 3
    max_depth = 20
    num_gates = 5 + N2 # Rx Ry Rz P CNOT on any qubit and then allowing CNOT target on any of the N2
    model_name = 'v0'
    output_size = num_gates
    num_branches = N2*max_depth

    # Load data
    x_train, y_train, x_val, y_val, x_test, y_test = load_data(f'{N2}_{max_depth}_100000')

    # Create model
    model, optimizer, criterion = create_model(2**N2, num_branches, output_size)

    # Train model
    batch_size = 64
    epochs = 100
    train(batch_size, epochs, model, optimizer, criterion, x_train, y_train, num_branches)
    
