from sklearn.metrics import classification_report, confusion_matrix

from training import NERDataset, NERModel, get_dfs, get_labels, load_model
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 16

def get_predicted_and_actual_labels(dataset: NERDataset, model: NERModel):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    device  = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()
    predicted_labels = torch.tensor([])
    actual_labels = torch.tensor([])

    for tokenized_sentence, labels in tqdm(dataloader):
        label_batch: torch.Tensor = labels.to(device)
        attention_mask = tokenized_sentence['attention_mask'].squeeze(1).to(device)
        input_ids = tokenized_sentence['input_ids'].squeeze(1).to(device)

       
        _, logits = model(input_ids, attention_mask, label_batch)
        for i in range(logits.shape[0]):
            clean_logits = logits[i][label_batch[i] != -100]
            clean_labels = labels[i][labels[i] != -100]
            preds = clean_logits.argmax(dim=1)
            
            predicted_labels =  torch.cat((predicted_labels, preds.to("cpu")))
            actual_labels = torch.cat((actual_labels, clean_labels.to("cpu")))
            
           
        
    return predicted_labels, actual_labels


dfs = get_dfs()
labels=get_labels(dfs)
model = NERModel(len(labels))
model = load_model(model, "epoch_1_val_acc_0.94_train_acc_0.92.pt")
print(len(labels))
model.bert.requires_grad_(False)
labels_to_ids = {v: k for k, v in enumerate(sorted(labels)) }
dataset = NERDataset(dfs["devel"], lables_to_ids=labels_to_ids)
predicted_labels, actual_labels = get_predicted_and_actual_labels(dataset, model)




def get_classification_report(actual_labels, predicted_labels, labels):
    return classification_report(actual_labels.tolist(), predicted_labels.tolist(), target_names=sorted(labels))

    
def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots()
    matrix = ax.matshow(cm, interpolation="nearest", cmap="coolwarm")

    for (x, y), value in np.ndenumerate(cm.T):
        ax.text(x, y, str(value),  ha='center', va='center')
    fig.colorbar(matrix)
    ax.set_xticks([i for i in range(len(labels))])
    ax.xaxis.set_ticklabels(sorted(labels))
    ax.set_yticks([i for i in range(len(labels))])
    ax.yaxis.set_ticklabels(sorted(labels))
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    fig.set_size_inches(20, 10)
    plt.show()

print(get_classification_report(actual_labels, predicted_labels, labels))
cm = confusion_matrix(actual_labels.tolist(), predicted_labels.tolist())
plot_confusion_matrix(cm, labels)