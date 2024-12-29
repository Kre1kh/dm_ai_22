import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

infrared_thermography_temperature = fetch_ucirepo(id=925)
X = infrared_thermography_temperature.data.features
y = infrared_thermography_temperature.data.targets

y = y['aveOralF']  # Или 'aveOralM'

for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])

# Обработка пропусков
X = X.fillna(X.mean())
y = y.fillna(y.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


class LayerwiseAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(LayerwiseAutoencoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            nn.Linear(input_dim, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32)
        ])
        self.decoder_layers = nn.ModuleList([
            nn.Linear(32, 64),
            nn.Linear(64, 128),
            nn.Linear(128, input_dim)
        ])
        self.activation = nn.ReLU()

    def forward(self, x):
        encoded = x
        for layer in self.encoder_layers:
            encoded = self.activation(layer(encoded))
        decoded = encoded
        for layer in self.decoder_layers:
            decoded = self.activation(layer(decoded))
        return decoded

    def encode(self, x, upto_layer):
        for i in range(upto_layer + 1):
            x = self.activation(self.encoder_layers[i](x))
        return x

    def decode(self, x, from_layer):
        for i in range(len(self.decoder_layers) - from_layer - 1, len(self.decoder_layers)):
            x = self.activation(self.decoder_layers[i](x))
        return x


def train_layer(autoencoder, train_loader, optimizer, criterion, layer_idx, epochs=10):
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, _ in train_loader:
            optimizer.zero_grad()
            # Кодировка/декодировка только до текущего слоя
            encoded = autoencoder.encode(X_batch, layer_idx)
            decoded = autoencoder.decode(encoded, layer_idx)
            loss = criterion(decoded, X_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Layer {layer_idx + 1} Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")


autoencoder = LayerwiseAutoencoder(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

for layer_idx in range(len(autoencoder.encoder_layers)):
    print(f"Training Layer {layer_idx + 1}")
    train_layer(autoencoder, train_loader, optimizer, criterion, layer_idx, epochs=10)


class CNNRegressor(nn.Module):
    def __init__(self, encoder):
        super(CNNRegressor, self).__init__()
        self.encoder = encoder
        self.regressor = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Регрессия с одним выходом
        )

    def forward(self, x):
        encoded = self.encoder.encode(x, len(self.encoder.encoder_layers) - 1)
        return self.regressor(encoded)


cnn_model = CNNRegressor(autoencoder)
criterion = nn.MSELoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.0001)

epochs = 1000
train_losses = []

for epoch in range(epochs):
    cnn_model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = cnn_model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))
    if (epoch + 1) % 100 == 0:
        print(f"Regression Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

plt.plot(range(1, epochs + 1), train_losses, label="Regression Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Regression Training Loss vs Epochs")
plt.show()

cnn_model.eval()
y_pred = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        preds = cnn_model(X_batch)
        y_pred.extend(preds.numpy())

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Истинные значения", color="blue")
plt.plot(y_pred, label="Прогнозируемые значения", color="orange")
plt.xlabel("Образцы")
plt.ylabel("Значение")
plt.title("Прогнозируемые vs Истинные значения")
plt.legend()
plt.show()

mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE: {mape: .4f}")
