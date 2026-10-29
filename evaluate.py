import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.cnn import MiteScanCNN
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Garantir que a pasta 'results' existe
os.makedirs("results", exist_ok=True)

# Transformações padrão de validação
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Carregar dados de validação
val_data = datasets.ImageFolder("dataset/val", transform=transform)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

# Instanciar a sua CNN autoral
model = MiteScanCNN()

# ---> CORREÇÃO RÁPIDA: Passo falso (dummy input) para forçar a criação das camadas dinâmicas <---
dummy_input = torch.randn(1, 3, 224, 224)
_ = model(dummy_input)

# Carregar o arquivo .pth do seu melhor treino anterior
model_path = "outputs/best_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"Modelo carregado com sucesso de {model_path}!")
else:
    print(f"Erro: O arquivo {model_path} não foi encontrado. Verifique se ele está na pasta outputs.")
    exit()

model.eval()
all_preds = []
all_labels = []

# Executar inferência no conjunto de validação
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

class_names = val_data.classes

# 1. Gerar o Relatório Completo (Precisão, Recall, F1-score por classe e F1 Macro)
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print("\n================ RELATÓRIO DE MÉTRICAS ESSENCIAIS ================")
print(report)
print("===================================================================")

# Salvar o relatório detalhado em .txt na pasta results
with open("results/classification_report.txt", "w", encoding="utf-8") as f:
    f.write("Métricas Essenciais do Modelo (TCC)\n")
    f.write("="*50 + "\n")
    f.write(report)
print("-> Relatório guardado com sucesso em: results/classification_report.txt")

# 2. Gerar e salvar a Matriz de Confusão Normalizada
cm = confusion_matrix(all_labels, all_preds, normalize='true')

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted (Previsto)")
plt.ylabel("True (Real)")
plt.title("Matriz de Confusão Normalizada")

plt.savefig("results/confusion_matrix.png", bbox_inches='tight')
plt.close()
print("-> Matriz de confusão guardada com sucesso em: results/confusion_matrix.png")