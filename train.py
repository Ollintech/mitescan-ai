import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.cnn import MiteScanCNN
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import shutil

# Garantir pasta de saída
os.makedirs("outputs", exist_ok=True)

# Transformação
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Dataset
train_data = datasets.ImageFolder("dataset/train", transform=transform)
val_data = datasets.ImageFolder("dataset/val", transform=transform)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

# Modelo
model = MiteScanCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Listas para gráfico
train_acc_list = []
val_acc_list = []

best_val_acc = 0

# Treino
for epoch in range(50):
    model.train()
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total
    train_acc_list.append(train_acc)

    # Validação
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1} | Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f}")

    # Salvar melhor modelo com tratamento anti-trava do Windows
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        temp_path = "outputs/temp_model.pth"
        final_path = "outputs/best_model.pth"
        try:
            torch.save(model.state_dict(), temp_path)
            if os.path.exists(final_path):
                os.remove(final_path)
            shutil.move(temp_path, final_path)
        except Exception as e:
            print(f"Aviso ao salvar modelo: {e}")

# Gráfico de Acurácia
plt.plot(train_acc_list, label="Train Accuracy")
plt.plot(val_acc_list, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.savefig("outputs/training_graph.png")
plt.show()

# Avaliação final e métricas detalhadas (Precisão, Recall, F1-score)
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

class_names = train_data.classes

# Gerar e imprimir o Relatório de Classificação no terminal
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print("\n================ RELATÓRIO DE MÉTRICAS ================")
print(report)
print("=======================================================")

# Salvar o relatório em texto na pasta outputs para o TCC
with open("outputs/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

# Matriz de confusão
cm = confusion_matrix(all_labels, all_preds, normalize='true')

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.show()