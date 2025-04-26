import torch

def convert_to_class_indices(labels, neutral_window):
    class_labels = torch.empty(labels.size(0), device=labels.device, dtype=torch.long)
    class_labels[labels < -neutral_window] = 0  # Negative class
    class_labels[(labels >= -neutral_window) & (labels <= neutral_window)] = 1  # Neutral class
    class_labels[labels > neutral_window] = 2  # Positive class

    return class_labels


def save_model(epoch, model, optimizer, neutral_window, model_save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'neutral_window': neutral_window
    }

    torch.save(checkpoint, model_save_path)

def load_model(model_save_path, model, optimizer):
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    current_epoch = checkpoint['epoch']
    neutral_window = checkpoint['neutral_window']
    return model, optimizer, current_epoch, neutral_window


def create_report_file(report_path, batch_size, neutral_window):
    with open(report_path, 'w') as file:
        file.write(f"Batch Size: {batch_size}, Neutral Window: {neutral_window}\n")
        file.write("Epoch, Loss, Accuracy, F1 Score\n")

def append_loss_data(report_path, epoch, loss, accuracy, f1_score):
    with open(report_path, 'a') as file:
        file.write(f"{epoch}, {loss:.4f}, {accuracy:.4f}, {f1_score:.4f}\n")