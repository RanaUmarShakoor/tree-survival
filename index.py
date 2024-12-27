import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

import torch 
import torch.nn as NN
from torch.utils.data import Dataset, DataLoader

import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------

data = pd.read_csv('./Tree_Data.csv')

print(data.head())

# ----------------------------------------------------------------------

# Checking for null values
print(data.isnull().sum())

# ----------------------------------------------------------------------

# Impute Event with the mode
data['Event'].fillna(data['Event'].mode()[0], inplace=True)

# ----------------------------------------------------------------------

# Impute EMF with the mean
data['EMF'].fillna(data['EMF'].mean(), inplace=True)

# ----------------------------------------------------------------------

# Drop columns with high missing data
data.drop(columns=['Harvest', 'Alive'], inplace=True)

# ----------------------------------------------------------------------

# Select relevant features for clustering
features = ['AMF', 'EMF', 'Phenolics', 'Lignin', 'NSC', 'Light_ISF']
X = data[features]

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering
inertia = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Elbow method visualization
plt.plot(range(2, 10), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Hierarchical Clustering
linkage_matrix = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Dendrogram for Hierarchical Clustering')
plt.show()

from lifelines import KaplanMeierFitter

# Assign cluster labels
optimal_k = 3  # Example value from elbow/silhouette analysis
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Kaplan-Meier survival analysis
kmf = KaplanMeierFitter()

plt.figure(figsize=(10, 6))
for cluster in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster]
    kmf.fit(durations=cluster_data['Time'], event_observed=cluster_data['Event'], label=f'Cluster {cluster}')
    kmf.plot_survival_function()

plt.title('Kaplan-Meier Survival Curves by Cluster')
plt.xlabel('Time (days)')
plt.ylabel('Survival Probability')
plt.legend()
plt.show()


features = ['AMF', 'EMF', 'Phenolics', 'Lignin', 'NSC', 'Light_ISF']
X = data[features]
Y = data['Event'] 

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
X_test, X_dev, Y_test, Y_dev = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

class TreeSurvivalDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.to_numpy(), dtype=torch.float32)
        self.labels = torch.tensor(labels.to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    

# Create Datasets
train_dataset = TreeSurvivalDataset(X_train, Y_train)
test_dataset = TreeSurvivalDataset(X_test, Y_test)
dev_dataset = TreeSurvivalDataset(X_dev, Y_dev)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)

# Create datasets
train_dataset = TreeSurvivalDataset(X_train, Y_train)
test_dataset = TreeSurvivalDataset(X_test, Y_test)
dev_dataset = TreeSurvivalDataset(X_dev, Y_dev)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define the neural network
class TreeSurvivalNN(NN.Module):
    def __init__(self, input_size):
        super(TreeSurvivalNN, self).__init__()
        self.model = NN.Sequential(
        NN.Linear(input_size, 64),
        NN.ReLU(),
        NN.Dropout(0.3),          
        NN.Linear(64, 1),
        NN.Sigmoid()
)

    def forward(self, x):
        return self.model(x)
    

# Instantiate the model
input_size = X_train.shape[1]
model = TreeSurvivalNN(input_size)

import torch.optim as optim

# Loss function and optimizer
criterion = NN.BCELoss()  # Binary Cross-Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

num_epochs = 50
best_dev_loss = float('inf')  # For early stopping

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0.0

    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()  # Reset gradients
        outputs = model(batch_features).squeeze()  # Forward pass
        loss = criterion(outputs, batch_labels)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        train_loss += loss.item()

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    dev_loss = 0.0
    with torch.no_grad():
        for batch_features, batch_labels in dev_loader:
            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_labels)
            dev_loss += loss.item()

    # Print progress
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Dev Loss: {dev_loss/len(dev_loader):.4f}")

    # Save the best model
    if dev_loss < best_dev_loss:
        best_dev_loss = dev_loss
        torch.save(model.state_dict(), "best_tree_survival_model.pth")

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Load the best model
model.load_state_dict(torch.load("best_tree_survival_model.pth"))
model.eval()

# Evaluate on test data
y_pred_list = []
y_true_list = []

with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        outputs = model(batch_features).squeeze()
        y_pred = (outputs >= 0.5).float()  # Convert probabilities to binary predictions
        y_pred_list.extend(y_pred.tolist())
        y_true_list.extend(batch_labels.tolist())

# Calculate metrics
accuracy = accuracy_score(y_true_list, y_pred_list)
precision = precision_score(y_true_list, y_pred_list)
recall = recall_score(y_true_list, y_pred_list)
conf_matrix = confusion_matrix(y_true_list, y_pred_list)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Example new data point
# features = ['AMF', 'EMF', 'Phenolics', 'Lignin', 'NSC', 'Light_ISF']
new_data = [[39.02, 26.48, -1.20, 8.78, 9.56, 0.047]]

new_data_scaled = scaler.transform(new_data)
new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

# Make prediction
model.eval()
with torch.no_grad():
    prediction = model(new_data_tensor).item()  # Get the probability (sigmoid output)
    predicted_class = 1 if prediction >= 0.5 else 0  # Apply threshold
    print(f"Predicted Probability: {prediction:.4f}")
    print(f"Predicted Class: {predicted_class}")