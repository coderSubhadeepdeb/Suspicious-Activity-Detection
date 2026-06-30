import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support
import csv

from models.autoencoder import ConvLSTMAE
from models.dataset import VideoFrameDataset
from training.config import *


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        print("ERROR: No checkpoint found. Train the model first.")
        return

    model = ConvLSTMAE(ENCODER_CHANNELS, CONVLSTM_HIDDEN, SEQUENCE_LENGTH).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    # Load thresholds
    threshold_path = os.path.join(CHECKPOINT_DIR, 'thresholds.npy')
    if os.path.exists(threshold_path):
        thresholds = np.load(threshold_path)
        threshold_low, threshold_high = thresholds[0], thresholds[1]
    else:
        threshold_low  = THRESHOLD_LOW
        threshold_high = THRESHOLD_HIGH
    print(f"Using thresholds: LOW={threshold_low:.6f}, HIGH={threshold_high:.6f}")

    # Load test dataset
    test_dataset = VideoFrameDataset(TEST_DIR, SEQUENCE_LENGTH, IMAGE_SIZE, stride=1)
    print(f"Test sequences: {len(test_dataset)}")

    if len(test_dataset) == 0:
        print("ERROR: No test data found. Check data/test/ folder.")
        return

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            error = torch.mean((output - batch) ** 2, dim=[2, 3, 4])
            scores = error.cpu().numpy().flatten().tolist()
            all_scores.extend(scores)
            # Label: 1 if anomaly, 0 if normal (based on threshold)
            labels = [1 if s >= threshold_low else 0 for s in scores]
            all_labels.extend(labels)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # AUC-ROC
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_scores)
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        print(f"\nAUC-ROC Score: {auc:.4f}")

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve.png'))
        print("ROC curve saved to outputs/roc_curve.png")
    else:
        print("Only one class in predictions — skipping AUC-ROC.")

    # Precision, Recall, F1
    pred_labels = (all_scores >= threshold_low).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, pred_labels, average='binary', zero_division=0
    )
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # Export scores to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'anomaly_scores.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence_idx', 'anomaly_score', 'predicted_label'])
        for i, (score, label) in enumerate(zip(all_scores, pred_labels)):
            writer.writerow([i, f"{score:.6f}", label])
    print(f"Anomaly scores exported to {csv_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    evaluate()