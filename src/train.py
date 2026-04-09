import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from losses import ClassBalancedFocalLoss
from network import DuaDeepImproved
from encoders import AntibodyEncoder, AntigenEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from data_processing import setup_balanced_dataset
import os
from evaluation import evaluate_and_plot

class BioInteractionDataset(Dataset):
    def __init__(self, ab_seqs, ag_seqs, labels):
        self.ab_seqs = ab_seqs
        self.ag_seqs = ag_seqs
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ab_seqs[idx], self.ag_seqs[idx], self.labels[idx]

def train_duadeep(data_path="AbRank_dataset.csv", epochs=10, batch_size=32, lr=1e-4, weight_decay=1e-2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing Training Pipeline on {device}")
    
    # Clear CUDA cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 1. Parse and preprocess ensuring MMseqs2 cluster representation.
    # Targeted undersampling controls the massive HIV/COVID data skew.
    df = setup_balanced_dataset(data_path, undersample_threshold=5000)
    
    # Stratified Splits keeping identical Clusters out of validation tests.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['cluster']))
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    # Setup Dataset Loaders
    train_dataset = BioInteractionDataset(
        list(train_df['antibody']), list(train_df['antigen']), list(train_df['label'])
    )
    val_dataset = BioInteractionDataset(
        list(val_df['antibody']), list(val_df['antigen']), list(val_df['label'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Extract specific counts to calculate Class-Balanced Focal weights natively
    value_counts = train_df['label'].value_counts().sort_index().values
    samples_per_cls = value_counts.tolist()
    print("Class Dist. for CB-Focal computation:", samples_per_cls)
    
    # 3. Initialize Architectures and CB-Focal function scaling against any residual class bias
    # Use frozen encoders to save memory
    model = DuaDeepImproved(ab_freeze=True, ag_freeze=True)
    model.to(device)
    
    classes_list = torch.tensor(samples_per_cls).float().to(device)
    criterion = ClassBalancedFocalLoss(samples_per_cls=classes_list, num_classes=2, beta=0.9999)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 4. Benchmarking/Training Loop mapping AUPRC & ROC-AUC over predictions
    os.makedirs('checkpoints', exist_ok=True)
    best_val_auc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] [Train]")
        for ab_batch, ag_batch, labels_batch in train_bar:
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            
            # Extract features across mapped representations
            logits = model(ab_batch, ag_batch)
            loss = criterion(labels_batch, logits)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
            
        print(f"Epoch [{epoch+1}/{epochs}] Final Loss: {total_loss/len(train_loader):.4f}")
        
        # Eval Step for generalizability metrics testing against untouched variants
        model.eval()
        val_loss, all_labels, all_preds = 0, [], []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] [Val]")
            for ab_batch, ag_batch, labels_batch in val_bar:
                labels_batch = labels_batch.to(device)
                logits = model(ab_batch, ag_batch)
                
                loss = criterion(labels_batch, logits)
                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())
                
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_labels.extend(labels_batch.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())
                
        # Benchmark validation splits natively mapping AUPRC for imbalanced logic success 
        val_roc = roc_auc_score(all_labels, all_preds)
        val_prauc = average_precision_score(all_labels, all_preds)
        val_preds = (np.array(all_preds) > 0.5).astype(int)
        val_f1 = f1_score(all_labels, val_preds)
        
        print(f"Val ROC-AUC: {val_roc:.4f} | Val AUPRC: {val_prauc:.4f} | Val F1: {val_f1:.4f}\n")
        
        # Checkpointing based on ROC-AUC performance preventing overfitting decay.
        if val_roc > best_val_auc:
            best_val_auc = val_roc
            chkpt_path = os.path.join('checkpoints', 'best_model.pth')
            torch.save(model.state_dict(), chkpt_path)
            print(f"--> Saved better model with ROC-AUC {val_roc:.4f} @ epoch {epoch+1}")
            
    # Step 5. Extrapolated Evaluation Plotting Mapping
    print("Training Completed. Formatting evaluation reports on Best Generalization Module...")
    
    # Reload Best weights to guarantee report maps correctly.
    best_model = DuaDeepImproved(ab_freeze=True, ag_freeze=True)
    best_model.load_state_dict(torch.load(os.path.join('checkpoints', 'best_model.pth')))
    best_model.to(device)
    best_model.eval()
    
    final_labels, final_preds = [], []
    with torch.no_grad():
        for ab_batch, ag_batch, labels_batch in val_loader:
            labels_batch = labels_batch.to(device)
            logits = best_model(ab_batch, ag_batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            final_labels.extend(labels_batch.cpu().numpy())
            final_preds.extend(probs.cpu().numpy())
            
    evaluate_and_plot(final_labels, final_preds, out_dir="results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DuaDeep Improved Training Pipeline")
    parser.add_argument("--data_path", type=str, default="AbRank_dataset.csv", help="Path to CSV dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for optimizer")
    
    args = parser.parse_args()
    
    train_duadeep(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
