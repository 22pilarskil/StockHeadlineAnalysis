import matplotlib.pyplot as plt
import numpy as np

def parse_training_log(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract metadata from first line
    metadata = lines[0].strip()
    
    # Parse the data
    data = []
    for line in lines[2:]:  # Skip first two lines (metadata and header)
        if line.strip():  # Skip empty lines
            parts = line.split(',')
            data.append([float(x.strip()) for x in parts])
    
    # Convert to numpy array
    data = np.array(data)
    
    return metadata, data

def plot_training_curves(file_path):
    metadata, data = parse_training_log(file_path)
    
    epochs = data[:, 0]
    loss = data[:, 1]
    accuracy = data[:, 2]
    f1 = data[:, 3]
    
    plt.figure(figsize=(18, 5))  # Wider figure for side-by-side plots
    
    # Plot Loss
    plt.subplot(1, 3, 1)  # 1 row, 3 columns, position 1
    plt.plot(epochs, loss, 'b-', label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss\n{metadata}')
    plt.grid(True)
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 3, 2)  # 1 row, 3 columns, position 2
    plt.plot(epochs, accuracy, 'r-', label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.grid(True)
    plt.legend()
    
    # Plot F1 Score
    plt.subplot(1, 3, 3)  # 1 row, 3 columns, position 3
    plt.plot(epochs, f1, 'g-', label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training F1 Score')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.show()

# Example usage:
plot_training_curves('report_run2.txt')