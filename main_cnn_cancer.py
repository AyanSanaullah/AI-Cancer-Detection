import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from medmnist import INFO, Evaluator, BreastMNIST
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import pandas as pd
import imageio
from torchvision.utils import make_grid

class BreastCNNModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(BreastCNNModel, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ModelAnalyzer:
    def __init__(self, model, test_loader, device, save_dir):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        self.class_names = ['Benign', 'Malignant']
        
        # Get predictions
        self.y_true, self.y_pred, self.y_score = self._get_predictions()
    
    def _get_predictions(self):
        self.model.eval()
        y_true = []
        y_pred = []
        y_score = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                y_true.extend(targets.squeeze().numpy())
                y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                y_score.extend(probs[:, 1].cpu().numpy())
        
        return np.array(y_true).squeeze(), np.array(y_pred), np.array(y_score)
    
    def plot_confusion_matrix(self):
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        annotations = np.array([[f'{count}\n({percent:.1f}%)' 
                               for count, percent in zip(row_counts, row_percentages)]
                               for row_counts, row_percentages in zip(cm, cm_percent)])
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'graphs2', 'confusion_matrix.png'))
        plt.close()
        
        return cm
    
    def plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'graphs2', 'roc_curve.png'))
        plt.close()
        
        return roc_auc
    
    def plot_precision_recall_curve(self):
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_score)
        avg_precision = average_precision_score(self.y_true, self.y_score)
        
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'graphs2', 'precision_recall_curve.png'))
        plt.close()
        
        return avg_precision
    
    def plot_score_distribution(self):
        plt.figure(figsize=(10, 6))
        for i, label in enumerate(self.class_names):
            mask = (self.y_true == i)
            scores = self.y_score[mask]
            if len(scores) > 0:
                plt.hist(scores, bins=50, alpha=0.5, label=label, density=True)
        
        plt.xlabel('Model Score')
        plt.ylabel('Density')
        plt.title('Score Distribution by Class')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'graphs2', 'score_distribution.png'))
        plt.close()
    
    def generate_full_report(self):
        cm = self.plot_confusion_matrix()
        roc_auc = self.plot_roc_curve()
        avg_precision = self.plot_precision_recall_curve()
        self.plot_score_distribution()
        
        report = classification_report(self.y_true, self.y_pred,
                                    target_names=self.class_names,
                                    output_dict=True)
        detailed_report = pd.DataFrame(report).transpose()
        
        tn, fp, fn, tp = cm.ravel()
        summary = {
            'Accuracy': (tp + tn) / (tp + tn + fp + fn),
            'AUC-ROC': roc_auc,
            'Average Precision': avg_precision,
            'Sensitivity': tp / (tp + fn),
            'Specificity': tn / (tn + fp),
            'PPV': tp / (tp + fp),
            'NPV': tn / (tn + fn)
        }
        
        # Save summary to file
        with open(os.path.join(self.save_dir, 'graphs2', 'model_performance_summary.txt'), 'w') as f:
            f.write("=== Model Performance Summary ===\n")
            for metric, value in summary.items():
                f.write(f"{metric}: {value:.3f}\n")
            
            f.write("\n=== Confusion Matrix ===\n")
            f.write(f"True Negative: {tn}\n")
            f.write(f"False Positive: {fp}\n")
            f.write(f"False Negative: {fn}\n")
            f.write(f"True Positive: {tp}\n")
            
            f.write("\n=== Detailed Classification Report ===\n")
            f.write(detailed_report.to_string())
        
        return summary, detailed_report

def create_combined_frame(model, val_loader, device, metrics, epoch, val_true_labels, val_preds, val_pred_probs, samples, true_labels, save_dir):
    """Create a single frame combining all visualizations."""
    plt.figure(figsize=(20, 20))
    
    # 1. Confusion Matrix (Top Left)
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(val_true_labels, val_preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Benign', 'Malignant'],
               yticklabels=['Benign', 'Malignant'])
    plt.title(f'Confusion Matrix (Epoch {epoch + 1})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 2. Learning Curves (Top Right)
    plt.subplot(2, 2, 2)
    epochs = range(1, epoch + 2)
    plt.plot(epochs, metrics['train_accs'][:epoch+1], label='Train Acc')
    plt.plot(epochs, metrics['val_accs'][:epoch+1], label='Val Acc')
    plt.plot(epochs, [auc * 100 for auc in metrics['train_aucs'][:epoch+1]], label='Train AUC')
    plt.plot(epochs, [auc * 100 for auc in metrics['val_aucs'][:epoch+1]], label='Val AUC')
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Score Distribution (Bottom Left)
    plt.subplot(2, 2, 3)
    benign_scores = np.array(val_pred_probs)[np.array(val_true_labels) == 0]
    malignant_scores = np.array(val_pred_probs)[np.array(val_true_labels) == 1]
    
    plt.hist(benign_scores, bins=20, alpha=0.5, label='Benign', color='green',
            density=True, range=(0, 1))
    plt.hist(malignant_scores, bins=20, alpha=0.5, label='Malignant', color='red',
            density=True, range=(0, 1))
    
    plt.title(f'Score Distribution\nVal Acc: {metrics["val_accs"][epoch]:.1f}%, AUC: {metrics["val_aucs"][epoch]:.3f}')
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Sample Images with Predictions (Bottom Right)
    plt.subplot(2, 2, 4)
    if samples and len(samples) >= 4:
        # Create a 2x2 grid of sample images
        grid_size = 2
        for idx in range(4):
            plt.subplot(2, 2, 4 + idx + 1 - 4)  # Adjust subplot position
            img = samples[idx].squeeze().numpy()
            img = (img + 1) / 2  # Denormalize
            plt.imshow(img, cmap='gray')
            true_label = 'Malignant' if true_labels[idx] == 1 else 'Benign'
            plt.title(f'True: {true_label}')
            plt.axis('off')
    
    plt.tight_layout()
    frame_path = os.path.join(save_dir, 'graphs2', 'combined_frames', f'frame_{epoch:03d}.png')
    plt.savefig(frame_path, bbox_inches='tight', dpi=150)
    plt.close()
    return frame_path

def train(model, train_loader, val_loader, device, save_dir, num_epochs=200):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_val_auc = 0
    metrics = {
        'train_losses': [], 'train_accs': [], 'train_aucs': [],
        'val_losses': [], 'val_accs': [], 'val_aucs': []
    }
    
    # Create directory for combined frames
    combined_frames_dir = os.path.join(save_dir, 'graphs2', 'combined_frames')
    os.makedirs(combined_frames_dir, exist_ok=True)
    
    # Get some sample images for visualization
    samples = []
    sample_labels = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            if len(samples) < 4:
                samples.extend([inputs[i] for i in range(min(4, len(inputs)))])
                sample_labels.extend([targets[i].item() for i in range(min(4, len(targets)))])
            else:
                break
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        train_pred_probs = []
        train_true_labels = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            probs = torch.softmax(outputs, dim=1)
            train_pred_probs.extend(probs[:, 1].detach().cpu().numpy())
            train_true_labels.extend(targets.cpu().numpy())
        
        # Calculate training metrics
        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_auc = roc_auc_score(train_true_labels, train_pred_probs)
        
        metrics['train_losses'].append(avg_loss)
        metrics['train_accs'].append(train_acc)
        metrics['train_aucs'].append(train_auc)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_pred_probs = []
        val_true_labels = []
        val_preds = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
                probs = torch.softmax(outputs, dim=1)
                val_pred_probs.extend(probs[:, 1].cpu().numpy())
                val_true_labels.extend(targets.cpu().numpy())
                val_preds.extend(predicted.cpu().numpy())
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_auc = roc_auc_score(val_true_labels, val_pred_probs)
        
        metrics['val_losses'].append(avg_val_loss)
        metrics['val_accs'].append(val_acc)
        metrics['val_aucs'].append(val_auc)
        
        # Create combined frame
        frame_path = create_combined_frame(
            model, val_loader, device, metrics, epoch,
            val_true_labels, val_preds, val_pred_probs,
            samples, sample_labels, save_dir
        )
        
        # Learning rate scheduling
        scheduler.step(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(save_dir, 'graphs2', 'best_breast_model.pth'))
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training - Loss: {avg_loss:.4f}, Acc: {train_acc:.2f}%, AUC: {train_auc:.4f}')
        print(f'Validation - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%, AUC: {val_auc:.4f}')
        print('-' * 70)
    
    # Create final animation
    frames = []
    frame_files = sorted([f for f in os.listdir(combined_frames_dir) if f.startswith('frame_')])
    
    for frame_file in frame_files:
        frames.append(imageio.imread(os.path.join(combined_frames_dir, frame_file)))
    
    # Save animation with longer duration for better visibility
    imageio.mimsave(os.path.join(save_dir, 'graphs2', 'training_animation.gif'), 
                    frames, duration=0.3)  # 0.3 seconds per frame
    
    # Cleanup frame files
    for frame_file in os.listdir(combined_frames_dir):
        os.remove(os.path.join(combined_frames_dir, frame_file))
    os.rmdir(combined_frames_dir)
    
    return metrics

def save_sample_predictions(model, test_loader, device, save_dir, num_samples=8):
    """Save a grid of sample predictions with actual images."""
    model.eval()
    samples = []
    predictions = []
    true_labels = []
    confidences = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            if len(samples) >= num_samples:
                break
                
            batch_size = inputs.size(0)
            for i in range(batch_size):
                if len(samples) >= num_samples:
                    break
                samples.append(inputs[i])
                true_labels.append(targets[i].item())
                
                # Get prediction and confidence
                input_tensor = inputs[i:i+1].to(device)
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1).item()
                conf = probs[0][pred].item()
                
                predictions.append(pred)
                confidences.append(conf)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    for idx, (img, pred, true, conf) in enumerate(zip(samples, predictions, true_labels, confidences)):
        # Denormalize the image
        img = img.squeeze().numpy()
        img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        # Plot the image
        axes[idx].imshow(img, cmap='gray')
        
        # Set title with prediction info
        pred_label = 'Malignant' if pred == 1 else 'Benign'
        true_label = 'Malignant' if true == 1 else 'Benign'
        color = 'green' if pred == true else 'red'
        
        title = f'Pred: {pred_label}\nTrue: {true_label}\nConf: {conf:.2f}'
        axes[idx].set_title(title, color=color)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'graphs2', 'sample_predictions.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Set save directory to current workspace
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "model_outputs")
    os.makedirs(os.path.join(save_dir, 'graphs2'), exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Get dataset info
    data_flag = 'breastmnist'
    info = INFO[data_flag]
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    
    # Data transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    # Load datasets
    train_dataset = BreastMNIST(split='train', transform=transform, download=True)
    val_dataset = BreastMNIST(split='val', transform=transform, download=True)
    test_dataset = BreastMNIST(split='test', transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BreastCNNModel(in_channels=n_channels, num_classes=n_classes).to(device)
    
    # Train model
    metrics = train(model, train_loader, val_loader, device, save_dir)
    
    # Load best model and analyze
    model.load_state_dict(torch.load(os.path.join(save_dir, 'graphs2', 'best_breast_model.pth')))
    
    # Save sample predictions
    save_sample_predictions(model, test_loader, device, save_dir)
    
    analyzer = ModelAnalyzer(model, test_loader, device, save_dir)
    summary, detailed_report = analyzer.generate_full_report()
    
    # Plot comprehensive learning curves
    plt.figure(figsize=(15, 10))
    
    epochs = range(1, len(metrics['train_losses']) + 1)
    
    # Plot Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics['train_losses'], label='Train')
    plt.plot(epochs, metrics['val_losses'], label='Validation')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics['train_accs'], label='Train')
    plt.plot(epochs, metrics['val_accs'], label='Validation')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot AUC
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics['train_aucs'], label='Train')
    plt.plot(epochs, metrics['val_aucs'], label='Validation')
    plt.title('AUC Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add a text box with final metrics
    plt.subplot(2, 2, 4)
    plt.axis('off')
    final_text = f"Current Metrics (Epoch {len(epochs)}):\n\n"
    final_text += f"Training:\n"
    final_text += f"Loss: {metrics['train_losses'][-1]:.4f}\n"
    final_text += f"Accuracy: {metrics['train_accs'][-1]:.2f}%\n"
    final_text += f"AUC: {metrics['train_aucs'][-1]:.4f}\n\n"
    final_text += f"Validation:\n"
    final_text += f"Loss: {metrics['val_losses'][-1]:.4f}\n"
    final_text += f"Accuracy: {metrics['val_accs'][-1]:.2f}%\n"
    final_text += f"AUC: {metrics['val_aucs'][-1]:.4f}\n\n"
    final_text += f"Best Validation AUC: {max(metrics['val_aucs']):.4f}"
    plt.text(0.1, 0.5, final_text, fontsize=10, family='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'graphs2', 'learning_curves.png'))
    plt.close()
    
    print("\nTraining completed and analysis saved!")
    print(f"\nAll files have been saved to: {os.path.join(save_dir, 'graphs2')}")
    print("\nGenerated files:")
    print(f"- {os.path.join(save_dir, 'graphs2', 'best_breast_model.pth')} (Best model weights)")
    print(f"- {os.path.join(save_dir, 'graphs2', 'model_performance_summary.txt')} (Detailed metrics)")
    print(f"- {os.path.join(save_dir, 'graphs2', 'confusion_matrix.png')} (Confusion matrix visualization)")
    print(f"- {os.path.join(save_dir, 'graphs2', 'roc_curve.png')} (ROC curve with AUC score)")
    print(f"- {os.path.join(save_dir, 'graphs2', 'precision_recall_curve.png')} (Precision-Recall curve)")
    print(f"- {os.path.join(save_dir, 'graphs2', 'score_distribution.png')} (Distribution of model scores)")
    print(f"- {os.path.join(save_dir, 'graphs2', 'learning_curves.png')} (Training loss and validation AUC)")
    print(f"- {os.path.join(save_dir, 'graphs2', 'sample_predictions.png')} (Sample predictions with actual images)")

if __name__ == '__main__':
    main()