import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data_path = "/work/aashfaq/datasets/deap/data_preprocessed_python/"
# data_path = "/Users/ashatabak/cmu/datasets/deap/data_preprocessed_python/"
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
deap_df_dict = {
    "labels" : [],
    "data" : [],
    "user":[],
    "trial":[]
}
count = 0
for root, subdirs, files in os.walk(data_path):
    for file in files:
        # print(root, subdirs, files)
        if len(subdirs) == 0:
            filepath = root+ "/" + file
            print(filepath)
            user1_physio = pickle.load(open(filepath, 'rb'), encoding='bytes')
            user = file.split(".")[0]
            for trial_id in range(len(user1_physio[b'labels'])):
                deap_df_dict["user"].append(user)
                deap_df_dict["trial"].append(trial_id)
                deap_df_dict["labels"].append(user1_physio[b'labels'][trial_id])
                deap_df_dict["data"].append(user1_physio[b'data'][trial_id])


deap_df = pd.DataFrame.from_dict(deap_df_dict)
print(len(deap_df.loc[:,["trial","user"]]))
shuffle = True
batch_size = 16


def read_batches(data_df, batch_size, shuffle ):
    batch_num = int(np.ceil(len(data_df) / batch_size))
    index_array = list(range(len(data_df)))
    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        effective_batch_size = len(indices)
        batch_data = [data_df.iloc[idx,:] for idx in indices]
        batch_labels = np.array([np.round(row['labels'][0:1]-1).astype(np.int) for row in batch_data])
        batch_features = np.array([np.reshape(row['data'], -1).astype(np.float32) for row in batch_data])
        # print(batch_labels)
        # print(batch_features)
        batch_features, batch_labels = torch.from_numpy(batch_features), torch.from_numpy(batch_labels)

        batch_features = batch_features.view(effective_batch_size, 40, -1)

        yield batch_features.to(device), batch_labels.to(device)


# for f, l in read_batches(deap_df, 16, True):
#     print(f.shape)
#     print(l.shape, l)
#     break


class CNNClassifier(nn.Module):
    def __init__(self, embed_dim,  in_channels, n_filter, filter_size, num_labels, dropout_prob):
        super(CNNClassifier, self).__init__()
        self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=n_filter,
                                kernel_size=filter_size)
        self.linear = nn.Linear(n_filter, num_labels)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X):
        X_in  = X#.unsqueeze(1)
        after_conv = F.relu(self.cnn(X_in))
        after_pool = F.max_pool1d(after_conv, after_conv.shape[2]).squeeze(2)
        ht = self.dropout(after_pool)
        out = self.linear(ht)
        return out

    def predict(self, X):
        with torch.no_grad():
            out = self.forward(X)
            predicted_class = torch.argmax(out, dim=-1).cpu().numpy()
            return out, predicted_class


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    return sum([x == y for x,y in zip(preds,y.view(-1).cpu().numpy())])


def evaluate(model, eval_dataset, batch_size, criterion):
    # print("eval called")
    model = model.eval()
    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0
    for batch_features, batch_labels in read_batches(eval_dataset, batch_size, False):
        out, pred_class = model.predict(batch_features)
        acc = categorical_accuracy(pred_class, batch_labels)
        loss = criterion(out, batch_labels.view(-1))
        epoch_loss += loss.item()
        epoch_acc += acc
        num_batches += len(batch_features)
        # print(num_batches)
    return epoch_loss / num_batches, epoch_acc / num_batches


def train(train_data, eval_data, test_data, model, batch_size):
    model = model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    best_valid_loss = float('inf')
    best_eval_acc = 0.0
    for epoch_id in range(200):
        for batch_features, batch_labels in read_batches(train_data, batch_size, True):
            optimizer.zero_grad()
            out = model(batch_features)
            # print(out.shape, batch_labels.shape)
            loss = criterion(
                out,  # (batch_size , num_classes)
                batch_labels.view(-1)  # (batch_size * 1)
            )
            loss.backward()
            optimizer.step()

        dev_loss, dev_acc = evaluate(model, eval_data, batch_size, criterion)
        print(f'Epoch {epoch_id} | dev. accuracy={dev_acc} | dev_loss={dev_loss}')
        if dev_acc > best_eval_acc:
            best_eval_acc = dev_acc
        if dev_loss < best_valid_loss:
            best_valid_loss = dev_loss
            torch.save(model.state_dict(), 'cnn_model.pt')
    dev_loss, dev_acc = evaluate(model, test_data, batch_size, criterion)
    print(f'FINAL TEST  | test. accuracy={dev_acc} | test_loss={dev_loss}')

users = list(deap_df["user"].unique())
model =  CNNClassifier(embed_dim = 8064,  in_channels = 40, n_filter = 400, filter_size = 200, num_labels= 9, dropout_prob=0.0)
model = model.to(device)
train_users, valid_users, test_users = users[:int(0.8*len(users))], users[int(0.8*len(users)):int(0.9*len(users))], users[int(0.9*len(users)):]
train_data = deap_df[deap_df.user.isin(train_users)]
eval_data = deap_df[deap_df.user.isin(test_users)]
test_data = deap_df[deap_df.user.isin(test_users)]
print(len(test_data),len(eval_data), len(train_data))
train(train_data,eval_data , test_data, model, 16)
