import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from copy import deepcopy as dc
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
# print(deap_df)
personal_vector = np.zeros(len(deap_df["user"].unique()))

def append_personal(user, features):
    feature_one_hot = dc(personal_vector)
    user_num = int(user.strip("s"))
    feature_one_hot[user_num-1] = 1
    return np.concatenate(features, feature_one_hot)

def read_batches(data_df, batch_size, shuffle ):
    batch_num = int(np.ceil(len(data_df) / batch_size))
    index_array = list(range(len(data_df)))
    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        effective_batch_size = len(indices)
        batch_data = [data_df.iloc[idx,:] for idx in indices]
        batch_labels = np.array([np.round((row['labels'][0:1])/9.0).astype(np.int) for row in batch_data])
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
    def __init__(self, embed_dim,  in_channels, n_filters, filter_sizes, num_label_arr, dropout_prob):
        super(CNNClassifier, self).__init__()
        self.cnns = nn.ModuleList([nn.Conv1d(in_channels=in_channels, out_channels=n_filters_x,
                                            kernel_size=filter_size_x)
                                    for n_filters_x, filter_size_x in zip(n_filters, filter_sizes) ])

        self.linear_arr = nn.ModuleList([nn.Linear(sum(n_filters), num_labels) for num_labels in num_label_arr])
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X):
        X_in  = X#.unsqueeze(1)
        after_convs = [F.relu(cnn(X_in)) for cnn in self.cnns]
        after_pools = [F.max_pool1d(after_conv, after_conv.shape[2]).squeeze(2) for after_conv in after_convs]
        joined_pool = torch.cat(after_pools, dim=1)
        ht = self.dropout(joined_pool)
        outs = [fc(ht) for fc in self.linear_arr]
        return outs

    def predict(self, X):
        with torch.no_grad():
            outs = self.forward(X)
            predicted_classes = [torch.argmax(out, dim=-1).cpu().numpy() for out in outs]
            return outs, predicted_classes

def categorical_accuracy(preds, y):
    """
    preds :
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    return sum([x == y for x,y in zip(preds,y.view(-1).cpu().numpy())])

def evaluate(model, eval_dataset, batch_size, criterion):
    # print("eval called")
    model = model.eval()
    epoch_loss = 0
    epoch_acc_dict = { i:0 for i in range(4)}
    num_batches = 0
    for batch_features, batch_labels in read_batches(eval_dataset, batch_size, False):
        outs, pred_classes = model.predict(batch_features)
        losses = []
        for i, out in enumerate(outs):
            losses.append(criterion(
                out,  # (batch_size , num_classes)
                batch_labels[:, i]  # (batch_size * 1)
            ))
            epoch_acc_dict[i] += categorical_accuracy(pred_classes[i], batch_labels[:, i])
        loss = sum(losses)
        epoch_loss += loss.item()

        num_batches += len(batch_features)
        # print(num_batches)
    epoch_acc_dict = {k:v/num_batches for k,v in epoch_acc_dict.items()}
    return epoch_loss / num_batches, epoch_acc_dict

def train(train_data, eval_data, test_data, model, batch_size):
    model = model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.33, cooldown=1, min_lr=1e-6)
    best_valid_loss = float('inf')
    best_eval_acc = 0.0
    for epoch_id in range(50):
        for batch_features, batch_labels in read_batches(train_data, batch_size, True):
            optimizer.zero_grad()
            outs = model(batch_features)
            # print(out.shape, batch_labels.shape)
            losses = []
            for i,out in enumerate(outs):
                losses.append(criterion(
                    out,  # (batch_size , num_classes)
                    batch_labels[:,i]  # (batch_size * 1)
                ))
            loss = sum(losses)
            loss.backward()
            optimizer.step()

        dev_loss, dev_acc = evaluate(model, eval_data, batch_size, criterion)
        scheduler.step(dev_loss)
        print(f'Epoch {epoch_id} | dev. accuracy={dev_acc} | dev_loss={dev_loss}')
        if dev_loss < best_valid_loss:
            best_valid_loss = dev_loss
            torch.save(model.state_dict(), 'cnn_model.pt')
    model.load_state_dict(torch.load('cnn_model.pt'))
    dev_loss, dev_acc = evaluate(model, test_data, batch_size, criterion)
    print(f'FINAL TEST  | test. accuracy={dev_acc} | test_loss={dev_loss}')

users = list(deap_df["user"].unique())
model = CNNClassifier(embed_dim=8064,  in_channels=40, n_filters=[50,50,100,100,50], filter_sizes=[32,64,128,512,1024], num_label_arr=[2], dropout_prob=0.25)
model = model.to(device)
split_type = "participant"
if split_type == "participant":
    train_users, valid_users, test_users = users[:int(0.8*len(users))], users[int(0.8*len(users)):int(0.9*len(users))], users[int(0.9*len(users)):]
    train_data = deap_df[deap_df.user.isin(train_users)]
    eval_data = deap_df[deap_df.user.isin(test_users)]
    test_data = deap_df[deap_df.user.isin(test_users)]
else:
    train_users,  test_users = users[:int(0.1 * len(users))], users[int(0.1 * len(users)):]
    test_data = deap_df[deap_df.user.isin(train_users)]
    train_data_pre = deap_df[deap_df.user.isin(test_users)]
    train_data, eval_data = train_test_split(train_data_pre, test_size = 0.85, random_state = 42)
print(len(test_data),len(eval_data), len(train_data))
train(train_data,eval_data , test_data, model, 32)
