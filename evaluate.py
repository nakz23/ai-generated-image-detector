import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from utils.dataset import get_dataloaders
from models.resnet_model import get_model
from torchvision import datasets

device = torch.device("cpu")
data_dir = "data/small_dataset"

# ----------------------------
# Load Data
# ----------------------------
train_loader, val_loader, test_loader = get_dataloaders(data_dir)

# ----------------------------
# Load Model
# ----------------------------
model = get_model()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# ----------------------------
# Get True Class Mapping
# ----------------------------
ds = datasets.ImageFolder(root=data_dir + "/test")
idx_to_class = {v: k for k, v in ds.class_to_idx.items()}

display_map = {
    "real": "Real",
    "fake": "AI-Generated"
}

target_names = [
    display_map.get(idx_to_class[i], idx_to_class[i].capitalize())
    for i in range(len(idx_to_class))
]

print("Class mapping:", ds.class_to_idx)

# ----------------------------
# Inference on Test Set
# ----------------------------
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probabilities[:, 1].cpu().numpy())  # Probability of class index 1

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# ----------------------------
# Accuracy
# ----------------------------
accuracy = (all_preds == all_labels).mean() * 100
print(f"\nTest Accuracy: {accuracy:.2f}%\n")

# ----------------------------
# Classification Report
# ----------------------------
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=target_names))

# ----------------------------
# Confusion Matrix (Visual)
# ----------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ----------------------------
# ROC Curve
# ----------------------------
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

print(f"\nROC AUC Score: {roc_auc:.4f}")
