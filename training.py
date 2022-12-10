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

# This function returns all the directories containing data for the model
def get_train_devel_test_dirs():
    train_devel_test_file_dirs = {}
    for d in train_test_devel_data_dirs:
        file_dirs = [os.path.join(d, genre_dir, "no-morph") for genre_dir in os.listdir(d) if os.path.isdir(os.path.join(d, genre_dir)) and "no-morph" in os.listdir(os.path.join(d, genre_dir))]
        train_devel_test_file_dirs[os.path.basename(d)] = file_dirs
    return train_devel_test_file_dirs

# This function loads the csv files in the given directory
def load_all_csv_files_in_dir(path_to_dir, train_test_devel, genre, save_intermediate_dataframes_to_csv = False):
    data_file_paths = [os.path.join(path_to_dir, cf) for cf in get_csv_files_in_dir(path_to_dir)]
    combined_df = pd.DataFrame()
    for csv_file in data_file_paths:
        df = pd.read_csv(csv_file)
        if "sentence_index" in combined_df:
            df["sentence_index"] = df["sentence_index"] + (combined_df["sentence_index"].max() + 1)
        combined_df = pd.concat([combined_df, df])
    return combined_df

# This function returns a list containing the sentences from the given data frame
def get_sentences(df: pd.DataFrame):
    copy_df = df.copy()
    copy_df = copy_df.sort_values(["sentence_index", "position_number_in_sentence"]) # sort values by sentence index and position in sentence to make iterating easier
    sentences = []
    for i in range(copy_df["sentence_index"].max()):
        form_tag_pairs = copy_df[copy_df["sentence_index"]==i][["position_number_in_sentence", "FORM", "CONLL:NER"]]
        if (len(form_tag_pairs) > 0): # there are some sentence ids that do not have any words, this is how I filter them
            sentences.append({"FORM": form_tag_pairs["FORM"].tolist(),"TAG": form_tag_pairs["CONLL:NER"].tolist()}) # get only the form and the NER tag of the record
        
    return sentences


# Code from: https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a
# aligns the labels of a tokenized sentence
def align_labels_of_tokenized_sentence(sentence, labels, labels_to_ids, should_tokenize_sub_words = False):
    SPECIAL_TOKEN_ID = -100 # tokens with this id are skipped by the binary cross entropy loss
    label_ids = []
    previous_word_id = None
    for word_id in sentence:
        if word_id is None: # None is the word_id of special tokens such as [SEP] etc.
            label_ids.append(SPECIAL_TOKEN_ID) 
        elif word_id != previous_word_id: # new word
            label_ids.append(labels_to_ids[labels[word_id]]) 
        else: # Enters this branch if a word was broken into multiple parts
            label_ids.append(labels_to_ids[labels[word_id]] if should_tokenize_sub_words else SPECIAL_TOKEN_ID)
        previous_word_id = word_id
    return label_ids

# This is the PyTorch Dataset for loading the data for the model
class NERDataset(torch.utils.data.Dataset): 
    # needs the data frame containing all the words and a labels to id mapping
    def __init__(self, data_df, labels_to_ids):
        # gets all the sentences 
        sentences = get_sentences(data_df) 
        # then tokenizes them
        self.tokenized_sentences = [tokenizer(sentence["FORM"], padding='max_length', max_length=512, truncation=True, return_tensors="pt", is_split_into_words=True) for sentence in sentences]
        # then gets all the labels for the tokens
        self.labels = [align_labels_of_tokenized_sentence(tokenized_sentences.word_ids(), sentence["TAG"], labels_to_ids) for tokenized_sentences, sentence in zip(self.tokenized_sentences, sentences)]

        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
            return self.tokenized_sentences[index],  torch.LongTensor(self.labels[index])

# This is the PyTorch Dataset for loading the data and augmenting it for the model from a pandas dataframe that contains sentences 
class NERDatasetFromSentenceDF(torch.utils.data.Dataset):
    # similar to NERDataset here original_df_size is the size of the original sentence data frame, augmention_factor tells us
    # how many times we want to add the augmentation sentences to the original sentence data frame
    def __init__(self, sentences_df: pd.DataFrame, labels_to_ids, original_df_size, augmention_factor):
        
        self.original_df_size = original_df_size
        self.augmention_factor = augmention_factor
        # Here we get the size of augmention data 
        self.augmented_size = int((len(sentences_df) - original_df_size) / augmention_factor)
        self.augmented_df = sentences_df
        
        # Here we store one instance of the original sentences and the augmentation sentences
        self.sentences = sentences_df[0:original_df_size+self.augmented_size].copy()

        # We tokenize the sub data frame created previously
        # Because we tokenize only the sub data frame for fast loadin
        # We have to implement a custom logic when accessing the elements
        self.tokenized_sentences = [tokenizer(eval(sentence["FORM"]), padding='max_length', max_length=512, truncation=True, return_tensors="pt", is_split_into_words=True) for _, sentence in self.sentences.iterrows()]
        
        # We get the labels for the tokenized sentences
        self.labels = [align_labels_of_tokenized_sentence(tokenized_sentences.word_ids(), eval(sentence["TAG"]), labels_to_ids) for tokenized_sentences, (_, sentence) in zip(self.tokenized_sentences, self.sentences.iterrows())]
        self.labels_to_ids = labels_to_ids

        
    def __len__(self):
        return len(self.augmented_df)
    
    # With this getter logic, we don't have to tokenize the whole data frame
    # Just the original and the augmentation part once 
    # And return the corresponding tokenized sentence to the given index
    def __getitem__(self, index):
            if (index < self.original_df_size):
                return self.tokenized_sentences[index], torch.LongTensor(self.labels[index])
            else:
                new_index = self.original_df_size + ((index - self.original_df_size) % self.augmented_size)
                
                return self.tokenized_sentences[new_index], torch.LongTensor(self.labels[new_index])
               

# This code loads and returns the train, devel and test data frames
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

# This function returns the NER lables from the train, devel and test data frames
def get_labels(dfs):
    combined_df = pd.DataFrame()
    for df_key in dfs:
        combined_df = pd.concat([combined_df, dfs[df_key]])
    labels = combined_df["CONLL:NER"].unique()
    return labels

# This is our custom huBERT model
class NERModel(torch.nn.Module):
    
    def __init__(self, num_labels, loss_fn=nn.CrossEntropyLoss(), token_to_filter=-100):
        super(NERModel, self).__init__()
        self.num_labels = num_labels
        self.loss_fn = loss_fn
        self.token_to_filter = token_to_filter

        # We add the base huBERT layer
        self.bert = AutoModel.from_pretrained("SZTAKI-HLT/hubert-base-cc")

        # Here we add the Dropout layer
        self.dropout1 = nn.Dropout(0.1)

        # Here we add a Linear layer
        self.linear1 = nn.Linear(in_features=768, out_features=512)

        # Here we add the ReLU activation layer
        self.relu1 =  nn.ReLU()

        # Here we add another Linear layer
        self.linear2 = nn.Linear(in_features=512, out_features=num_labels)

        # Here we add the Sigmoid activation layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, labels) :
        # Here we feed the input through the model
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout1(x[0])
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        
        # Then we calculate the loss
        loss = self.calculate_loss(x, labels, attention_mask)
        return loss, x

    def calculate_loss(self, predicted_labels: torch.Tensor, actual_labels: torch.Tensor, attention_mask: torch.Tensor)-> torch.Tensor:
        #Code from: https://github.com/abhishekkrthakur/bert-entity-extraction/tree/master
        # Get the mask for tokens where the attention is set to 1
        mask = attention_mask.view(-1) == 1
        
        # Change the shape so it contains the the outputed label value for each word sequentially 
        active_logits = predicted_labels.view(-1, self.num_labels)
       
        # Get the actual labels for each word
        actual_labels_with_ignore_index = torch.where(mask,
        actual_labels.view(-1),
        torch.tensor(self.loss_fn.ignore_index).type_as(actual_labels))

        # Calculate the loss
        loss = self.loss_fn(active_logits, actual_labels_with_ignore_index)
        return loss




#Training code made with help from: https://towardsdatascience.com/custom-named-entity-recognition-with-bert-cf1fd4510804
def train_loop(model, train_df, devel_df, test_df, writer, labels_to_ids):

    original_train_size = len(train_df)
    train_augmentation_factor = 10
    train_df = get_augmented_train_df(train_augmentation_factor)    

    # Initialize the datasets
    train_dataset, devel_dataset, test_dataset = NERDatasetFromSentenceDF(train_df, labels_to_ids, original_train_size, train_augmentation_factor), NERDataset(devel_df, labels_to_ids), NERDataset(test_df, labels_to_ids)

    # Set the hyper parameters
    batch_size = 64
    epoch_num = 10
    learning_rate = 0.0001

    # Initialize the data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    devel_dataloader = DataLoader(devel_dataset, batch_size=batch_size)

    # Initialize the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # Use GPU if available
    is_cuda_available = torch.cuda.is_available()
    device  = "cuda" if is_cuda_available else "cpu"
    if is_cuda_available:
        model.to(device)

    # Turn of learning for the base huBERT layer
    model.bert.requires_grad_(False)
    
    # Training loop
    for epoch in range(epoch_num):
        # Set the model to train mode
        model.train()
        # Iterate over the sentences
        for tokenized_sentence, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}"):
            # Put the sentences through the model
            train_labels: torch.Tensor = labels.to(device)
            attention_mask = tokenized_sentence['attention_mask'].squeeze(1).to(device)
            input_ids = tokenized_sentence['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask, train_labels)

            
            # Then start the backpropagation
            loss.backward()
            optimizer.step()
        # After the sentence iteration, we set the model to eval mode 
        model.eval()
        # We get the accuracy for the train and devel datasets
        train_accuracy, total_matches_train = get_accuracy_on_dataset(model, train_dataloader, device, "train")
        devel_accuracy, total_matches_devel = get_accuracy_on_dataset(model, devel_dataloader, device, "devel")
        # We print the results
        print(f"Epoch: {epoch+1} Train Accuracy: {train_accuracy * 100:.2f}% Devel Accuracy: {devel_accuracy*100:.2f}%")
        print(f"Train matches: Got {total_matches_train} matches out of {len(train_df)}")
        print(f"Devel matches: Got {total_matches_devel} matches out of {len(devel_df)}")
        # Then add them to Tensorboard
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Accuracy/devel", devel_accuracy, epoch)

        # At the end of the epoch we save the model
        save_model(model, f"epoch_{epoch + 1}_val_acc_{devel_accuracy:.2f}_train_acc_{train_accuracy:.2f}.pt")


# This function returns the accuracy of a model on a dataset
def get_accuracy_on_dataset(model: NERModel, dataset_loader: DataLoader, device, dataset_name):
  model.eval()
  total_matches = 0
  total_tokens = 0
  # It iterates over the sentences and counts how many of the tokens where predicted correctly
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
# This function returns the number of times the predicted label mathces the actual label
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

# This function saves the model
def save_model(model, file_name):
    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, file_name))

# This function loads a saved model
def load_model(model: NERModel, file_name: str) -> NERModel:
    model.load_state_dict(torch.load(
        os.path.join(MODEL_SAVE_DIR, file_name)))
    return model


# This function appends the augmentation sentences to the original sentences the given amount of times
def get_augmented_train_df(add_no_o_tag_sentences_num_times = 1): 
  train_df =  pd.read_csv(os.path.join(train_test_devel_data_path, "train", "original_sentences.csv"))
  train_sentences_removed_o_tag_df =  pd.read_csv(os.path.join(train_test_devel_data_path, "train", "removed_o_tag_sentences.csv"))
  for i in range(add_no_o_tag_sentences_num_times):
      train_df = pd.concat([train_df, train_sentences_removed_o_tag_df], ignore_index=True)
  return train_df

# This is the main function of the training, this starts the training loop
def main(writer, labels_to_ids): 
  dfs = get_dfs()
  dfs["train"] = pd.read_csv(os.path.join(train_test_devel_data_path, "train", "original_sentences.csv"))
  model = NERModel(LABEL_COUNT)
  train_loop(model=model, train_df=dfs["train"], devel_df=dfs["devel"], test_df=dfs["test"], writer=writer, labels_to_ids=labels_to_ids)

# Constant for the model save directory
MODEL_SAVE_DIR = "models"
# This is the huBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("SZTAKI-HLT/hubert-base-cc")

# This is a constant for the train, devel, test data root directory path
train_test_devel_data_path = os.path.join("data", "train-devel-test")

# This is a constant for the train, devel, test data sub directories that contain the actual data
train_test_devel_data_dirs = [os.path.join(train_test_devel_data_path, data_dir) for data_dir in os.listdir(train_test_devel_data_path) if os.path.isdir(os.path.join(train_test_devel_data_path, data_dir))]

# This is a constant for the .csv files that contain the content of the whole genre
csv_file_pattern = re.compile(".*_full.csv") 

if __name__ == '__main__':
    # We get the data frames
    dfs = get_dfs()
    # We get the labels
    labels=get_labels(dfs)
    # We create the id to label mapping
    ids_to_labels = {k: v for k, v in enumerate(sorted(labels)) }
    # We create the label to id mapping 
    labels_to_ids = {v: k for k, v in enumerate(sorted(labels)) }
    # This is a constant for the label count
    LABEL_COUNT = len(labels)
    # Creates the model save dir if it does not exist already
    if(not os.path.exists(MODEL_SAVE_DIR)):
        os.mkdir(MODEL_SAVE_DIR)
    # Then we open a writer for TensorBoard and it's time to start the training
    with SummaryWriter() as writer:
        main(writer,labels_to_ids)