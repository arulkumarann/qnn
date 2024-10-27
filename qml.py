import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 30  # Increased from 10 to 30
batch_size = 16  # Reduced batch size for better convergence
learning_rate_cnn = 0.001
learning_rate_qnn = 0.001  # Smaller learning rate for QNN stability
n_qubits = 4
q_depth = 4  # Increased quantum circuit depth
n_samples = 1000  # Number of training samples to use

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((4, 4)),  # Reduce image size for QNN
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                         train=True,
                                         transform=transform,
                                         download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                        train=False,
                                        transform=transform,
                                        download=True)

# Create subset of data
train_subset = Subset(train_dataset, range(n_samples))
test_subset = Subset(test_dataset, range(n_samples//5))

# Data loaders
train_loader = DataLoader(dataset=train_subset,
                         batch_size=batch_size,
                         shuffle=True)

test_loader = DataLoader(dataset=test_subset,
                        batch_size=batch_size,
                        shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Added batch normalization
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)  # Added dropout
        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class QNN(nn.Module):
    def __init__(self):
        super(QNN, self).__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        
        # Enhanced pre-processing layers
        self.pre_net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, n_qubits),
            nn.Tanh()  # Bound inputs to [-1, 1]
        )
        
        # Create quantum devices for each qubit
        self.q_devices = [qml.device("default.qubit", wires=1) for _ in range(self.n_qubits)]
        
        def create_quantum_circuit(idx):
            @qml.qnode(self.q_devices[idx])
            def quantum_circuit(inputs, weights):
                # Enhanced quantum circuit
                qml.RY(inputs[0] * np.pi, wires=0)  # Amplitude encoding
                
                for i in range(self.q_depth):
                    qml.RY(weights[i], wires=0)
                    qml.RZ(weights[i], wires=0)
                    if i < self.q_depth - 1:
                        qml.RX(np.pi/2, wires=0)  # Add non-commuting operations
                
                return qml.expval(qml.PauliZ(0))
            return quantum_circuit
        
        self.quantum_circuits = [create_quantum_circuit(i) for i in range(self.n_qubits)]
        
        # Initialize trainable weights with smaller values
        self.weights = nn.Parameter(0.1 * torch.randn(self.n_qubits, self.q_depth))
        
        # Enhanced post-processing layers
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 10)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        x = self.pre_net(x)
        
        q_out = torch.zeros(batch_size, self.n_qubits, device=device)
        for i in range(batch_size):
            for j in range(self.n_qubits):
                q_out[i, j] = self.quantum_circuits[j](x[i:i+1, j], self.weights[j])
        
        x = self.post_net(q_out)
        return x
# Modified training function with learning rate scheduling
def train_model(model, train_loader, criterion, optimizer, device, model_type):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for images, labels in tqdm(train_loader, desc=f'Training {model_type}'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    end_time = time.time()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    epoch_time = end_time - start_time
    
    return epoch_loss, epoch_acc, epoch_time
# Training function

# Testing function
def test_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    
    return test_loss, test_acc

# Training and evaluation

# Modified run_experiment function with learning rate scheduling
def run_experiment():
    results = {
        'cnn': {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'time': []},
        'qnn': {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'time': []}
    }
    
    cnn = CNN().to(device)
    qnn = QNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Optimizers with different learning rates
    cnn_optimizer = optim.Adam(cnn.parameters(), lr=learning_rate_cnn)
    qnn_optimizer = optim.Adam(qnn.parameters(), lr=learning_rate_qnn)
    
    # Learning rate schedulers
    cnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(cnn_optimizer, mode='min', factor=0.5, patience=3)
    qnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(qnn_optimizer, mode='min', factor=0.5, patience=3)
    
    best_cnn_acc = 0
    best_qnn_acc = 0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        
        cnn_train_loss, cnn_train_acc, cnn_time = train_model(
            cnn, train_loader, criterion, cnn_optimizer, device, "CNN")
        cnn_test_loss, cnn_test_acc = test_model(
            cnn, test_loader, criterion, device)
        
        qnn_train_loss, qnn_train_acc, qnn_time = train_model(
            qnn, train_loader, criterion, qnn_optimizer, device, "QNN")
        qnn_test_loss, qnn_test_acc = test_model(
            qnn, test_loader, criterion, device)
        
        # Update learning rates
        cnn_scheduler.step(cnn_train_loss)
        qnn_scheduler.step(qnn_train_loss)
        
        # Store results
        for model_type, train_loss, train_acc, test_loss, test_acc, time in [
            ('cnn', cnn_train_loss, cnn_train_acc, cnn_test_loss, cnn_test_acc, cnn_time),
            ('qnn', qnn_train_loss, qnn_train_acc, qnn_test_loss, qnn_test_acc, qnn_time)
        ]:
            results[model_type]['train_loss'].append(train_loss)
            results[model_type]['train_acc'].append(train_acc)
            results[model_type]['test_loss'].append(test_loss)
            results[model_type]['test_acc'].append(test_acc)
            results[model_type]['time'].append(time)
        
        # Track best accuracies
        best_cnn_acc = max(best_cnn_acc, cnn_test_acc)
        best_qnn_acc = max(best_qnn_acc, qnn_test_acc)
        
        print(f'CNN - Train Loss: {cnn_train_loss:.4f}, Train Acc: {cnn_train_acc:.2f}%, '
              f'Test Acc: {cnn_test_acc:.2f}%, Time: {cnn_time:.2f}s')
        print(f'QNN - Train Loss: {qnn_train_loss:.4f}, Train Acc: {qnn_train_acc:.2f}%, '
              f'Test Acc: {qnn_test_acc:.2f}%, Time: {qnn_time:.2f}s')
        print(f'Best CNN Acc: {best_cnn_acc:.2f}%, Best QNN Acc: {best_qnn_acc:.2f}%')
    save_model(cnn, 'cnn_model')
    save_model(qnn, 'qnn_model')
        
    return results
# Plot results
import matplotlib.pyplot as plt

# Plot results and save each metric separately
def plot_results(results):
    epochs = range(1, len(results['cnn']['train_loss']) + 1)

    def save_metric_plot(epochs, cnn_values, qnn_values, title, ylabel, filename):
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, cnn_values, 'b-', label='CNN')
        plt.plot(epochs, qnn_values, 'r-', label='QNN')

        # Annotate each point with its value
        for i, (x, y_cnn, y_qnn) in enumerate(zip(epochs, cnn_values, qnn_values)):
            plt.text(x, y_cnn, f'{y_cnn:.2f}', ha='center', va='bottom', fontsize=8, color='blue')
            plt.text(x, y_qnn, f'{y_qnn:.2f}', ha='center', va='bottom', fontsize=8, color='red')

        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot as an image
        plt.savefig(filename)
        plt.close()

    # Save individual plots
    save_metric_plot(
        epochs, results['cnn']['train_loss'], results['qnn']['train_loss'], 
        'Training Loss', 'Loss', 'training_loss.png'
    )
    save_metric_plot(
        epochs, results['cnn']['train_acc'], results['qnn']['train_acc'], 
        'Training Accuracy', 'Accuracy (%)', 'training_accuracy.png'
    )
    save_metric_plot(
        epochs, results['cnn']['test_acc'], results['qnn']['test_acc'], 
        'Test Accuracy', 'Accuracy (%)', 'test_accuracy.png'
    )
    save_metric_plot(
        epochs, results['cnn']['time'], results['qnn']['time'], 
        'Training Time per Epoch', 'Time (seconds)', 'training_time.png'
    )
    
    epochs = range(1, num_epochs + 1)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training Loss
    ax1.plot(epochs, results['cnn']['train_loss'], 'b-', label='CNN')
    ax1.plot(epochs, results['qnn']['train_loss'], 'r-', label='QNN')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Training Accuracy
    ax2.plot(epochs, results['cnn']['train_acc'], 'b-', label='CNN')
    ax2.plot(epochs, results['qnn']['train_acc'], 'r-', label='QNN')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    # Test Accuracy
    ax3.plot(epochs, results['cnn']['test_acc'], 'b-', label='CNN')
    ax3.plot(epochs, results['qnn']['test_acc'], 'r-', label='QNN')
    ax3.set_title('Test Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    
    # Training Time
    ax4.plot(epochs, results['cnn']['time'], 'b-', label='CNN')
    ax4.plot(epochs, results['qnn']['time'], 'r-', label='QNN')
    ax4.set_title('Training Time per Epoch')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Time (seconds)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('qnn_vs_cnn_comparison.png')
    plt.close()

def save_model(model, model_name):
    torch.save(model.state_dict(), f'{model_name}.pth')
    print(model_name + " is saved as pth file...")
if __name__ == "__main__":
    print("Starting experiment...")
    results = run_experiment()
    plot_results(results)
    print("Experiment completed. Results and model are saved.")