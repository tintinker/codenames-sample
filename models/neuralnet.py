import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import sys
import random
from pathlib import Path
import os

random.seed(42)
MAP_FILENAME = "map.json"

hidden_dims = [800, 256, 512]
dropout_rates =[0.5, 0.2, 0.3] 
weight_decays = [0,  1e-5, 1e-4]
batch_norms = [False]  # Example batch normalization
learning_rates = [0.001, 0.0005, 0.0001]

# Define a more complex neural network model
class MulticlassClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, weight_decay, batch_norm):
        super(MulticlassClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.batch_norm = nn.BatchNorm1d(hidden_dim) if batch_norm else None
        
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.fc3(x)
        return x

def run_training(model: MulticlassClassifier, optimizer, criterion, train_loader: DataLoader, val_loader: DataLoader, weight_decay: float, patience: int = 5, num_epochs: int = 200, val_frequency: int = 5):
    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in tqdm(range(num_epochs), leave=False):
        model.train()
        train_loss = 0.0
        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            # Apply weight decay
            l2_reg = torch.tensor(0.0)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += weight_decay * l2_reg
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_loader)

        if epoch % val_frequency == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_inputs, batch_labels in val_loader:
                    outputs = model(batch_inputs)
                    val_loss += criterion(outputs, batch_labels).item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= patience:
                tqdm.write(f"Early stopping: No improvement for {patience} epochs.")
                break
        
            tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

def run_evaluate(model: MulticlassClassifier, hyperparams: dict, X_test_tensor: torch.FloatTensor, y_test, folder: str = "output"):
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        predicted = model(X_test_tensor)
        predicted_classes = (predicted > 0).int()
        test_accuracy = accuracy_score(y_test, predicted_classes.numpy())
    
    tqdm.write(f"Test accuracy: {test_accuracy:.2f}")

    info = hyperparams.copy()
    info["accuracy"] = test_accuracy
    info["name"] = "_".join([str(v) for k,v in info.items()]) + ".pt"

    torch.save(model.state_dict(), f"{folder}/{info['name']}")

    with open(f"{folder}/{MAP_FILENAME}") as f:
        m = json.load(f)
    m.append(info)

    with open(f"{folder}/{MAP_FILENAME}", "w+") as f:
        json.dump(m, f)

def init_output_directory(output_folder):

        # Create a Path object for the file
    file_path = Path(output_folder) / MAP_FILENAME

    # Create the target directory if it doesn't exist
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if the file exists
    if not file_path.exists():
        # File doesn't exist, create it with an empty list
        with open(file_path, "w+") as file:
            json.dump([], file)
            print(f"Created {MAP_FILENAME} with an empty list.")
    else:
        # File already exists
        print(f"{MAP_FILENAME} already exists.")

def main(df, output_folder = "output", glove_dim = 300):
    word_list = df.index.tolist()
    categories = df.columns

    print(len(categories), len(word_list), df.shape)
    target = df.to_numpy()

    # Convert words to GloVe vectors
    glove = GloVe(name='6B', dim=glove_dim)
    word_vectors = [glove[word] for word in word_list]

    # Convert word vectors and target to numpy arrays
    X = np.array(word_vectors)
    y = target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    
    input_dim = glove_dim
    output_dim = len(categories)

    init_output_directory(output_folder)

    for dropout_rate in tqdm(dropout_rates, position=0):
        for weight_decay in tqdm(weight_decays,position=1, leave=False):
            for batch_norm in tqdm(batch_norms, position=2, leave=False):
                for learning_rate in tqdm(learning_rates, position=3, leave=False):
                    for hidden_dim in tqdm(hidden_dims, position=4, leave=False):
                        model = MulticlassClassifier(input_dim, hidden_dim, output_dim, dropout_rate, weight_decay, batch_norm)
                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                        criterion = nn.BCEWithLogitsLoss()

                        hyperparams = {
                            "dropout_rate": dropout_rate,
                            "weight_decay": weight_decay,
                            "batch_norm": batch_norm,
                            "learning_rate": learning_rate,
                            "hidden_dim": hidden_dim
                        }

                        run_training(model, optimizer, criterion, train_loader, val_loader, weight_decay)
                        run_evaluate(model, hyperparams, X_test_tensor, y_test, output_folder)


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], index_col=0)
    main(df, sys.argv[2])