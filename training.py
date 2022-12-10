import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd 
import os
import re
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def get_csv_files_in_dir(path_to_dir):
    return [f for f in os.listdir(path_to_dir) if csv_file_pattern.match(f)]


def get_train_devel_test_dirs():
    train_devel_test_file_dirs = {}
    for d in train_test_devel_data_dirs:
        file_dirs = [os.path.join(d, genre_dir, "no-morph") for genre_dir in os.listdir(d) if os.path.isdir(os.path.join(d, genre_dir)) and "no-morph" in os.listdir(os.path.join(d, genre_dir))]
        train_devel_test_file_dirs[os.path.basename(d)] = file_dirs
    return train_devel_test_file_dirs


def load_all_csv_files_in_dir(path_to_dir, train_test_devel, genre, save_intermediate_dataframes_to_csv = False):
    data_file_paths = [os.path.join(path_to_dir, cf) for cf in get_csv_files_in_dir(path_to_dir)]
    combined_df = pd.DataFrame()
    for csv_file in data_file_paths:
        df = pd.read_csv(csv_file)
        if "sentence_index" in combined_df:
            df["sentence_index"] = df["sentence_index"] + (combined_df["sentence_index"].max() + 1)
        combined_df = pd.concat([combined_df, df])
    return combined_df


def get_sentences(df: pd.DataFrame):
    copy_df = df.copy()
    copy_df = copy_df.sort_values(["sentence_index", "position_number_in_sentence"])
    sentences = []
    for i in range(copy_df["sentence_index"].max()):
        form_tag_pairs = copy_df[copy_df["sentence_index"]==i][["position_number_in_sentence", "FORM", "CONLL:NER"]]
        if (len(form_tag_pairs) > 0):
            sentences.append({"FORM": form_tag_pairs["FORM"].tolist(),"TAG": form_tag_pairs["CONLL:NER"].tolist()})
        
    return sentences



def align_labels_of_tokenized_sentence(sentence, labels, labels_to_ids, should_tokenize_sub_words = False):
    SPECIAL_TOKEN_ID = -100
    label_ids = []
    previous_word_id = None
    for word_id in sentence:
        if word_id is None:
            label_ids.append(SPECIAL_TOKEN_ID)
        elif word_id != previous_word_id:
            label_ids.append(labels_to_ids[labels[word_id]])
        else:
            label_ids.append(labels_to_ids[labels[word_id]] if should_tokenize_sub_words else SPECIAL_TOKEN_ID)
        previous_word_id = word_id
    return label_ids

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, data_df, labels_to_ids):
        sentences = get_sentences(data_df)
        self.tokenized_sentences = [tokenizer(sentence["FORM"], padding='max_length', max_length=512, truncation=True, return_tensors="pt", is_split_into_words=True) for sentence in sentences]
        self.labels = [align_labels_of_tokenized_sentence(tokenized_sentences.word_ids(), sentence["TAG"], labels_to_ids) for tokenized_sentences, sentence in zip(self.tokenized_sentences, sentences)]

        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
            return self.tokenized_sentences[index],  torch.LongTensor(self.labels[index])


class NERDatasetFromSentenceDF(torch.utils.data.Dataset):
    def __init__(self, sentences_df: pd.DataFrame, labels_to_ids, original_df_size, augmention_factor):
        
        self.original_df_size = original_df_size
        self.augmention_factor = augmention_factor
        self.augmented_size = int((len(sentences_df) - original_df_size) / augmention_factor)
        self.augmented_df = sentences_df
        self.sentences = sentences_df[0:original_df_size+self.augmented_size].copy()
        self.tokenized_sentences = [tokenizer(eval(sentence["FORM"]), padding='max_length', max_length=512, truncation=True, return_tensors="pt", is_split_into_words=True) for _, sentence in self.sentences.iterrows()]
        self.labels = [align_labels_of_tokenized_sentence(tokenized_sentences.word_ids(), eval(sentence["TAG"]), labels_to_ids) for tokenized_sentences, (_, sentence) in zip(self.tokenized_sentences, self.sentences.iterrows())]
        self.labels_to_ids = labels_to_ids

        
    def __len__(self):
        return len(self.augmented_df)
    
    def __getitem__(self, index):
            if (index < self.original_df_size):
                return self.tokenized_sentences[index], torch.LongTensor(self.labels[index])
            else:
                new_index = self.original_df_size + ((index - self.original_df_size) % self.augmented_size)
                
                return self.tokenized_sentences[new_index], torch.LongTensor(self.labels[new_index])
               



tokenizer = AutoTokenizer.from_pretrained("SZTAKI-HLT/hubert-base-cc")

train_test_devel_data_path = os.path.join("data", "train-devel-test")
train_test_devel_data_dirs = [os.path.join(train_test_devel_data_path, data_dir) for data_dir in os.listdir(train_test_devel_data_path) if os.path.isdir(os.path.join(train_test_devel_data_path, data_dir))]
csv_file_pattern = re.compile(".*_full.csv") 


def get_dfs():
    dfs = {}
    train_devel_test_dirs = get_train_devel_test_dirs()

    for data_set in train_devel_test_dirs:
        dfs[data_set] = pd.DataFrame()
        for genre_dir in train_devel_test_dirs[data_set]:
            print(f"Loading: {genre_dir}")
            genre = genre_dir.split(os.path.sep)[-2]
            df = load_all_csv_files_in_dir(genre_dir, data_set, genre, True)
            if "sentence_index" in dfs[data_set]:
                df["sentence_index"] = df["sentence_index"] + (dfs[data_set]["sentence_index"].max() + 1)
            dfs[data_set] = pd.concat([dfs[data_set], df], ignore_index=True)
    return dfs

def get_labels(dfs):
    combined_df = pd.DataFrame()
    for df_key in dfs:
        combined_df = pd.concat([combined_df, dfs[df_key]])
    labels = combined_df["CONLL:NER"].unique()
    return labels


class NERModel(torch.nn.Module):
    
    def __init__(self, num_labels, loss_fn=nn.CrossEntropyLoss(), token_to_filter=-100):
        super(NERModel, self).__init__()
        self.num_labels = num_labels
        self.loss_fn = loss_fn
        self.token_to_filter = token_to_filter

        self.bert = AutoModel.from_pretrained("SZTAKI-HLT/hubert-base-cc")
        self.dropout1 = nn.Dropout(0.1)
        self.linear1 = nn.Linear(in_features=768, out_features=512)
        self.relu1 =  nn.ReLU()
        self.linear2 = nn.Linear(in_features=512, out_features=num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels) :
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout1(x[0])
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        
        loss = self.calculate_loss(x, labels, attention_mask)
        return loss, x

    def calculate_loss(self, predicted_labels: torch.Tensor, actual_labels: torch.Tensor, attention_mask: torch.Tensor)-> torch.Tensor:
        #Code from: https://github.com/abhishekkrthakur/bert-entity-extraction/tree/master
        mask = attention_mask.view(-1) == 1
        active_logits = predicted_labels.view(-1, self.num_labels)
        actual_labels_with_ignore_index = torch.where(mask,
        actual_labels.view(-1),
        torch.tensor(self.loss_fn.ignore_index).type_as(actual_labels))
        loss = self.loss_fn(active_logits, actual_labels_with_ignore_index)
        return loss




#Training code made with help from: https://towardsdatascience.com/custom-named-entity-recognition-with-bert-cf1fd4510804
def train_loop(model, train_df, devel_df, test_df, writer, labels_to_ids):
    original_train_size = len(train_df)
    train_augmentation_factor = 10
    train_df = get_augmented_train_df(train_augmentation_factor)    
    train_dataset, devel_dataset, test_dataset = NERDatasetFromSentenceDF(train_df, labels_to_ids, original_train_size, train_augmentation_factor), NERDataset(devel_df, labels_to_ids), NERDataset(test_df, labels_to_ids)

    batch_size = 64
    epoch_num = 10
    learning_rate = 0.0001
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    devel_dataloader = DataLoader(devel_dataset, batch_size=batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    is_cuda_available = torch.cuda.is_available()
    device  = "cuda" if is_cuda_available else "cpu"
    if is_cuda_available:
        model.to(device)
    model.bert.requires_grad_(False)
   
    for epoch in range(epoch_num):
        model.train()
        for tokenized_sentence, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            train_labels: torch.Tensor = labels.to(device)
            attention_mask = tokenized_sentence['attention_mask'].squeeze(1).to(device)
            input_ids = tokenized_sentence['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask, train_labels)

            
            
            loss.backward()
            optimizer.step()
        
        model.eval()
        train_accuracy, total_matches_train = get_accuracy_on_dataset(model, train_dataloader, device, "train")
        devel_accuracy, total_matches_devel = get_accuracy_on_dataset(model, devel_dataloader, device, "devel")
        print(f"Epoch: {epoch+1} Train Accuracy: {train_accuracy * 100:.2f}% Devel Accuracy: {devel_accuracy*100:.2f}%")
        print(f"Train matches: Got {total_matches_train} matches out of {len(train_df)}")
        print(f"Devel matches: Got {total_matches_devel} matches out of {len(devel_df)}")

        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/devel", devel_accuracy, epoch)
        save_model(model, f"epoch_{epoch + 1}_val_acc_{devel_accuracy:.2f}_train_acc_{train_accuracy:.2f}.pt")

def get_accuracy_on_dataset(model: NERModel, dataset_loader: DataLoader, device, dataset_name):
  model.eval()
  total_matches = 0
  total_tokens = 0
  for tokenized_sentence, label in tqdm(dataset_loader, desc=f"Evaluating model on {dataset_name}"):
    labels: torch.Tensor = label.to(device)
    attention_mask = tokenized_sentence['attention_mask'].squeeze(1).to(device)
    input_ids = tokenized_sentence['input_ids'].squeeze(1).to(device)
    loss, logits = model(input_ids, attention_mask, labels)
    matches, tokens = get_matching_label_count(logits, labels)
    total_matches += matches
    total_tokens += tokens
  
  return (total_matches)/total_tokens, total_matches

# Idea from: https://towardsdatascience.com/custom-named-entity-recognition-with-bert-cf1fd4510804
def get_matching_label_count(logits, labels):
  count_of_matches = 0
  total_tokens = 0
  for i in range(logits.shape[0]):
    clean_logits = logits[i][labels[i] != -100]
    clean_labels = labels[i][labels[i] != -100]
    preds = clean_logits.argmax(dim=1)
    count_of_matches += (preds == clean_labels).sum()
    total_tokens += len(clean_labels)

  return count_of_matches, total_tokens

def save_model(model, file_name):
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, file_name))

def load_model(model: NERModel, file_name: str) -> NERModel:
    model.load_state_dict(torch.load(
        os.path.join(MODEL_SAVE_DIR, file_name)))
    return model


MODEL_SAVE_DIR = "models"



def get_augmented_train_df(add_no_o_tag_sentences_num_times = 1): 
  train_df =  pd.read_csv(os.path.join(train_test_devel_data_path, "train", "original_sentences.csv"))
  train_sentences_removed_o_tag_df =  pd.read_csv(os.path.join(train_test_devel_data_path, "train", "removed_o_tag_sentences.csv"))
  for i in range(add_no_o_tag_sentences_num_times):
      train_df = pd.concat([train_df, train_sentences_removed_o_tag_df], ignore_index=True)
  return train_df

def main(writer, labels_to_ids): 
  dfs = get_dfs()
  dfs["train"] = pd.read_csv(os.path.join(train_test_devel_data_path, "train", "original_sentences.csv"))
  model = NERModel(LABEL_COUNT)
  train_loop(model=model, train_df=dfs["train"], devel_df=dfs["devel"], test_df=dfs["test"], writer=writer, labels_to_ids=labels_to_ids)

if __name__ == '__main__':
    dfs = get_dfs()
    labels=get_labels(dfs)
    ids_to_labels = {k: v for k, v in enumerate(sorted(labels)) }
    labels_to_ids = {v: k for k, v in enumerate(sorted(labels)) }
    LABEL_COUNT = len(labels)

    if(not os.path.exists(MODEL_SAVE_DIR)):
        os.mkdir(MODEL_SAVE_DIR)

    with SummaryWriter() as writer:
        main(writer,labels_to_ids)