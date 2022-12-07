from sklearn.metrics import classification_report, confusion_matrix

from training import NERDataset, NERModel, LABEL_COUNT, get_dfs
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
BATCH_SIZE = 128

def get_predicted_and_actual_labels(dataset: NERDataset, model: NERModel):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    model.eval()
    predicted_labels = []
    actual_labels = []

    for tokenized_sentence, labels in tqdm(dataloader):
        attention_mask = tokenized_sentence['attention_mask'].squeeze(1)
        input_ids = tokenized_sentence['input_ids'].squeeze(1)

        actual_labels.append(labels)
        print(actual_labels)
        _, logits = model(input_ids, attention_mask, labels)
        print(logits)
        for i in range(logits.shape[0]):
            clean_logits = logits[i][labels[i] != -100]
            preds = clean_logits.argmax(dim=1)
            print(preds)
            predicted_labels.append(preds)
        
    return predicted_labels, actual_labels


model=NERModel(LABEL_COUNT)
dfs = get_dfs()
dataset = NERDataset(dfs["test"])
get_predicted_and_actual_labels(dataset, model)