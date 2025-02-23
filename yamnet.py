import os
import torch
import numpy as np
import pandas as pd
import gc
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import librosa
import json
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import EarlyStoppingCallback
import wandb
import colorama
from colorama import Fore, Back, Style

# Initialize colorama for cross-platform colored output
colorama.init()

class YAMNetBase(nn.Module):
    def __init__(self, num_classes=3):
        super(YAMNetBase, self).__init__()
        
        # YAMNet-like architecture
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Adaptive pooling to handle variable length inputs
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # Input shape: (batch_size, 1, time_steps)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Global pooling
        x = self.adaptive_pool(x)
        x = x.flatten(1)
        
        # Classification
        x = self.classifier(x)
        return x

class AudioAugmenter:
    """Audio augmentation techniques with length preservation"""
    @staticmethod
    def pad_or_truncate(audio, max_length=32000):
        if len(audio) > max_length:
            return audio[:max_length]
        elif len(audio) < max_length:
            return np.pad(audio, (0, max_length - len(audio)), 'constant')
        return audio

    @staticmethod
    def add_noise(audio, noise_factor=0.005):
        try:
            noise = np.random.randn(len(audio))
            augmented = audio + noise_factor * noise
            augmented = np.clip(augmented, -1.0, 1.0)
            return augmented
        except Exception as e:
            print(f"Warning: Error in add_noise: {str(e)}")
            return audio

    @staticmethod
    def time_shift(audio, shift_max=0.1):
        try:
            shift = int(len(audio) * shift_max)
            return np.roll(audio, shift) if shift > 0 else audio
        except Exception as e:
            print(f"Warning: Error in time_shift: {str(e)}")
            return audio

    @staticmethod
    def change_speed(audio, speed_factor=0.2):
        try:
            audio = np.clip(audio, -1.0, 1.0)
            speed_change = np.random.uniform(low=0.9, high=1.1)
            
            augmented = librosa.effects.time_stretch(audio, rate=speed_change)
            augmented = np.clip(augmented, -1.0, 1.0)
            
            # Ensure fixed length after speed change
            augmented = AudioAugmenter.pad_or_truncate(augmented)
            
            return augmented
        except Exception as e:
            print(f"Warning: Error in change_speed: {str(e)}")
            return audio

    @staticmethod
    def augment(audio):
        audio = AudioAugmenter.pad_or_truncate(audio)
        audio = np.clip(audio, -1.0, 1.0)
        
        augmentation_list = ['noise', 'shift', 'speed']
        num_augments = np.random.randint(1, 3)
        selected_augments = np.random.choice(augmentation_list, num_augments, replace=False)
        
        augmented = audio.copy()
        for aug_type in selected_augments:
            try:
                if aug_type == 'noise':
                    augmented = AudioAugmenter.add_noise(augmented)
                elif aug_type == 'shift':
                    augmented = AudioAugmenter.time_shift(augmented)
                elif aug_type == 'speed':
                    augmented = AudioAugmenter.change_speed(augmented)
                
                augmented = np.clip(augmented, -1.0, 1.0)
                
            except Exception as e:
                print(f"Warning: Error during {aug_type} augmentation: {str(e)}")
                continue
        
        return augmented.astype(np.float32)

class AudioDataset(Dataset):
    def __init__(self, audio_data, labels, max_length=32000):
        self.audio_data = audio_data
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        audio = self.audio_data[idx].astype(np.float32)
        audio = self.pad_or_truncate(audio)
        
        # Add channel dimension for Conv1d
        audio = torch.FloatTensor(audio).unsqueeze(0)
        
        return {
            'input_values': audio,
            'label': torch.tensor(self.labels[idx])
        }
    
    def pad_or_truncate(self, audio):
        if len(audio) > self.max_length:
            return audio[:self.max_length]
        elif len(audio) < self.max_length:
            return np.pad(audio, (0, self.max_length - len(audio)), 'constant')
        return audio

class ConsoleVisualizer:
    """Handles console-based visualization of plots"""
    @staticmethod
    def plot_confusion_matrix(cm, labels):
        print("\nConfusion Matrix:")
        print("-" * 40)
        
        # Header
        print(f"{'':>10}", end='')
        for label in labels:
            print(f"{label:>10}", end='')
        print("\n")
        
        # Matrix
        for i, label in enumerate(labels):
            print(f"{label:>10}", end='')
            for j in range(len(labels)):
                if cm[i][j] == 0:
                    color = Fore.WHITE
                elif cm[i][j] == np.max(cm[i]):
                    color = Fore.GREEN
                else:
                    color = Fore.YELLOW
                print(f"{color}{cm[i][j]:>10}{Style.RESET_ALL}", end='')
            print()
        print("-" * 40)

    @staticmethod
    def plot_training_history(history):
        print("\nTraining History:")
        print("-" * 40)
        
        for epoch, metrics in enumerate(history):
            print(f"Epoch {epoch+1:>2}: "
                  f"Loss: {metrics['train_loss']:.4f} "
                  f"Val Loss: {metrics.get('val_loss', 'N/A')} "
                  f"Acc: {metrics.get('val_accuracy', 'N/A')}")

class YAMNetClassifier:
    def __init__(self, num_labels=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YAMNetBase(num_labels).to(self.device)
        self.label_map = {'crying': 0, 'screaming': 1, 'normal': 2}
        self.augmenter = AudioAugmenter()
        self.visualizer = ConsoleVisualizer()
        self.max_length = 32000  # 2 seconds at 16kHz

    def load_audio_file(self, file_path, target_sr=16000):
        try:
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
            audio = audio / (np.max(np.abs(audio)) + 1e-6)
            audio = AudioAugmenter.pad_or_truncate(audio, self.max_length)
            return audio.astype(np.float32)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def prepare_dataset(self, data_dir, metadata_file, augment=False):
        print(f"Loading metadata from {metadata_file}")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        df = pd.read_csv(metadata_file)
        print(f"Loaded {len(df)} entries from metadata")
        
        audio_data = []
        labels = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading audio files"):
            file_path = os.path.join(data_dir, row['file_name'])
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
                
            audio = self.load_audio_file(file_path)
            if audio is not None:
                audio_data.append(audio)
                labels.append(self.label_map[row['label']])
                
                if augment:
                    augmented_audio = self.augmenter.augment(audio)
                    audio_data.append(augmented_audio)
                    labels.append(self.label_map[row['label']])
        
        dataset = AudioDataset(audio_data, labels, self.max_length)
        print(f"Created dataset with {len(dataset)} examples")
        return dataset

    def save_model(self, path):
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        
        # Save label map
        with open(os.path.join(path, "label_map.json"), "w") as f:
            json.dump(self.label_map, f)
        
        print(f"Model saved at {path}")

    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            
            inputs = batch['input_values'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(train_loader), correct / total

    def evaluate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                inputs = batch['input_values'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return (
            total_loss / len(val_loader),
            correct / total,
            all_preds,
            all_labels
        )

    def train_kfold(self, dataset, output_dir, n_splits=5, use_wandb=False):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        indices = np.arange(len(dataset))
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
            print(f"\nTraining Fold {fold + 1}/{n_splits}")
            
            train_fold = torch.utils.data.Subset(dataset, train_idx)
            val_fold = torch.utils.data.Subset(dataset, val_idx)
            
            train_loader = DataLoader(train_fold, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_fold, batch_size=8)
            
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-5, weight_decay=0.01)
            criterion = nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=2, verbose=True
            )
            
            best_val_loss = float('inf')
            patience = 3
            patience_counter = 0
            history = []
            
            for epoch in range(25):
                train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
                val_loss, val_acc, _, _ = self.evaluate(val_loader, criterion)
                
                metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                }
                history.append(metrics)
                
                if use_wandb:
                    wandb.log(metrics)
                
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(output_dir, f"fold_{fold + 1}_best.pt")
                    torch.save(self.model.state_dict(), best_model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
            
            # Load best model for evaluation
            self.model.load_state_dict(torch.load(best_model_path))
            val_loss, val_acc, preds, labels = self.evaluate(val_loader, criterion)
            
            # Calculate and store fold metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average='weighted'
            )
            
            fold_metrics.append({
                'val_loss': val_loss,
                'accuracy': val_acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            })
            
            # Visualize fold results
            self.visualizer.plot_training_history(history)
            
            # Clear CUDA memory
            del train_loader, val_loader
            torch.cuda.empty_cache()
            gc.collect()
        
        # Print average metrics across folds
        print("\nAverage Metrics Across Folds:")
        avg_metrics = {
            key: np.mean([fold[key] for fold in fold_metrics])
            for key in fold_metrics[0].keys()
        }
        print(json.dumps(avg_metrics, indent=2))
        
        return avg_metrics

    def generate_performance_report(self, test_dataset, output_dir):
        """Generate and visualize performance metrics"""
        test_loader = DataLoader(test_dataset, batch_size=8)
        criterion = nn.CrossEntropyLoss()
        
        # Evaluate model
        val_loss, accuracy, preds, labels = self.evaluate(test_loader, criterion)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        
        metrics = {
            'test_loss': val_loss,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
        # Generate and display confusion matrix
        cm = confusion_matrix(labels, preds)
        label_names = list(self.label_map.keys())
        self.visualizer.plot_confusion_matrix(cm, label_names)
        
        # Print metrics
        print("\nPerformance Metrics:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics

def main():
    # Set up output directory for YAMNet
    output_dir = "yamnet_model_output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("Initializing YAMNet classifier...")
        classifier = YAMNetClassifier()
        
        print("\nPreparing datasets with augmentation...")
        dataset = classifier.prepare_dataset(
            "Split_Data/train",
            "Split_Data/train_metadata.csv",
            augment=True
        )
        
        # Train with k-fold cross validation
        print("\nStarting k-fold cross validation training...")
        metrics = classifier.train_kfold(dataset, output_dir, n_splits=5)
        
        # Save best model
        best_model_path = os.path.join(output_dir, "best_model")
        classifier.save_model(best_model_path)
        print(f"\nBest model saved at: {best_model_path}")
        
        # Generate performance report on test set
        print("\nPreparing test dataset...")
        test_dataset = classifier.prepare_dataset(
            "Split_Data/test",
            "Split_Data/test_metadata.csv",
            augment=False
        )
        
        print("\nGenerating performance report...")
        test_metrics = classifier.generate_performance_report(test_dataset, output_dir)
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()