import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from dataset_utils import download_dataset, get_dataloaders


# Force line buffering for stdout
sys.stdout.reconfigure(line_buffering=True)

def main():

    # 1. Download/Get Dataset
    path = download_dataset()

    # 2. Data Transforms & 3. Load Data & Dataloaders
    batch_size = 128
    dataloaders, dataset_sizes, class_names = get_dataloaders(path, batch_size=batch_size)
    print(f"Classes: {class_names}")

    # 4. Setup Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")

    # 5. Load Vision Transformer (ViT)
    # Why is this unique?
    # Unlike CNNs (ResNet) that look at local pixels with sliding windows, 
    # ViT splits the image into 16x16 patches and processes them like a sequence of words (NLP style).
    # It uses "Self-Attention" to understand global relationships in the image instantly.
    print("Loading Pretrained Vision Transformer (ViT-B/16)...")
    try:
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
    except:
        print("Downloading weights failed or older torchvision. Using non-pretrained for structure check.")
        model = models.vit_b_16(pretrained=False)

    # Modify the classification head
    # In torchvision's ViT, the head is stored in 'heads'
    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(num_ftrs, 2)
    ) # Binary: REAL vs FAKE

    model = model.to(device)

    # Use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # 6. Loss and Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # ViT often benefits from AdamW optimizer instead of SGD
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    # 7. Training Loop
    num_epochs = 5 # Increased to 5 for better convergence
    print(f"Starting training for {num_epochs} epochs...")
    
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'train' and i == 1:
                     print(f'  Batch {i+1}/{len(dataloaders[phase])} Loss: {loss.item():.4f}')

                if phase == 'train' and (i+1) % 25 == 0:
                     print(f'  Batch {i+1}/{len(dataloaders[phase])} Loss: {loss.item():.4f}')
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                scheduler.step()
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                test_loss_history.append(epoch_loss)
                test_acc_history.append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # 8. Save
    torch.save(model.state_dict(), 'cifake_vit.pth')
    print("Model saved to cifake_vit.pth")

    # 9. Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(test_loss_history, label='Test Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Acc')
    plt.plot(test_acc_history, label='Test Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig('training_graphs.png')
    print("Training graphs saved to training_graphs.png")

    # 10. Evaluation & Stats
    print("\nEvaluating model on Test Set...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1]) # Probability for class 1 (Fake/Faulty)

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Classification Report
    target_names = ['Healthy', 'Faulty'] # Assuming 0=Real(Healthy), 1=Fake(Faulty) based on user prompt names
    # Note: User prompt used "Healthy" and "Faulty". CIFake is Real/Fake. 
    # Usually Real=0, Fake=1. I will map Real->Healthy, Fake->Faulty to match user request format.
    
    print("\nEvaluation Results:")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix saved to confusion_matrix.png")

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    print("ROC Curve saved to roc_curve.png")

    # Model Summary (Formatted as requested)
    print("\nViT Model Summary")
    print("====")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Optimizer: AdamW")
    print(f"Learning Rate: 0.0001")
    print(f"Weight Decay: 0.01")
    print(f"Loss Function: CrossEntropyLoss")
    print("===")
    
    # Support details from confusion matrix/labels
    # The user example showed 'support' with numbers. 
    # Classification report already shows support. 
    # I will print the raw counts as well if needed to match "support 150014..." style roughly.
    unique, counts = np.unique(all_labels, return_counts=True)
    print("Support (Test Set):")
    for u, c in zip(unique, counts):
        print(f"  {target_names[u]}: {c}")



if __name__ == '__main__':
    main()
