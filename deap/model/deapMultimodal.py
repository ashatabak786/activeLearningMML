import torch
from utils import read_video_data, read_phys_data, read_batches
from constants import data_path, data_path_atlas, video_features_path, video_features_path_atlas
# from deap.model.utils import read_video_data, read_phys_data, read_batches
# from deap.model.constants import data_path, data_path_atlas, video_features_path, video_features_path_atlas

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ModeFusion(nn.Module):
    def __init__(self, video_model, phys_model, out1_dim, out2_dim, num_label):
        super(ModeFusion, self).__init__()
        self.video_model = video_model
        self.phys_model = phys_model
        self.combine = nn.Linear(out1_dim + out2_dim, num_label)

    def forward(self, vid_X, phys_X):
        out_vid = self.video_model(vid_X)
        out_phys = self.phys_model(phys_X)
        out = torch.cat([out_vid,out_phys],1)
        out = self.combine(out)
        return out

    def predict(self, vid_X, phys_X):
        with torch.no_grad():
            out = self.forward(vid_X, phys_X)
            predicted_classes = torch.argmax(out, dim=-1).cpu().numpy()
            return out, predicted_classes

class LSTMCNNClassifierVideo(nn.Module):
    def __init__(self, embed_dim,  n_filter, filter_size, hidden_dim, num_label, dropout_prob):
        super(LSTMCNNClassifierVideo, self).__init__()

        self.cnn = nn.Conv2d(in_channels=1, out_channels=n_filter, stride=1,
                                             kernel_size=(filter_size, embed_dim))


        self.lstm = nn.LSTM(
            input_size=n_filter, hidden_size=hidden_dim,
            num_layers=1, bidirectional=True,
            batch_first=True)  # let it know we are batching things

        self.linear = nn.Linear(n_filter + hidden_dim * 2, num_label)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X):
        X_in  = X.unsqueeze(1)
        after_conv = F.relu(self.cnn(X_in)).squeeze(3)
        #short cut
        after_pool = F.max_pool1d(after_conv, after_conv.shape[2]).squeeze(2)
        lstm_out, (ht, ct) = self.lstm(after_conv.permute(0,2,1))
        ht = torch.cat((ht[0, :, :], ht[1, :, :], after_pool), 1)
        ht = self.dropout(ht)
        out = self.linear(ht)
        return out

    def predict(self, X):
        with torch.no_grad():
            out = self.forward(X)
            # print(out)
            predicted_classes = torch.argmax(out, dim=-1).cpu().numpy()
            return out, predicted_classes


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("main")






def categorical_accuracy(preds, y):
    """
    preds :
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # print(preds)
    return sum([x == y for x,y in zip(preds,y.view(-1).cpu().numpy())])


def evaluate(model, eval_dataset, eval_dataset_video, batch_size, criterion):
    # print("eval called")
    model = model.eval()
    epoch_loss = 0
    epoch_acc_dict = { i:0 for i in range(1)}
    num_batches = 0
    for batch_features, video_batch_features, batch_labels in read_batches(eval_dataset, eval_dataset_video, batch_size, False, device):
        batch_features = batch_features.permute((0, 2, 1))
        out, pred_classes = model.predict(video_batch_features, batch_features)
        outs = [out]
        losses = []
        for i, out in enumerate(outs):
            losses.append(criterion(
                out,  # (batch_size , num_classes)
                batch_labels.view(-1)  # (batch_size * 1)
            ))
            epoch_acc_dict[i] += categorical_accuracy(pred_classes, batch_labels)
        loss = sum(losses)
        epoch_loss += loss.item()

        num_batches += len(batch_features)
        # print(num_batches)
    epoch_acc_dict = {k:v/num_batches for k,v in epoch_acc_dict.items()}
    return epoch_loss / num_batches, epoch_acc_dict


def train(train_data_phys, video_train_data, eval_data, video_eval_data, test_data, video_test_data, model, batch_size):
    model = model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.33, cooldown=1, min_lr=1e-6, verbose=True)
    best_valid_loss = float('inf')
    best_eval_acc = 0.0
    for epoch_id in range(40):
        model = model.train()
        for batch_features, video_batch_features, batch_labels in read_batches(train_data_phys, video_train_data, batch_size, True, device):
            optimizer.zero_grad()
            batch_features = batch_features.permute((0,2,1))
            out = model(video_batch_features, batch_features)
            loss = criterion(
                out,  # (batch_size , num_classes)
                batch_labels.view(-1)  # (batch_size * 1)
            )

            loss.backward()
            optimizer.step()
        train_loss, train_acc = evaluate(model, train_data, video_train_data, batch_size, criterion)
        dev_loss, dev_acc = evaluate(model, eval_data, video_eval_data, batch_size, criterion)
        scheduler.step(dev_loss)
        print(f'Epoch {epoch_id} | train accuracy={train_acc} | train_loss={train_loss}')
        print(f'Epoch {epoch_id} | dev. accuracy={dev_acc} | dev_loss={dev_loss}')
        if dev_loss < best_valid_loss:
            best_valid_loss = dev_loss
            torch.save(model.state_dict(), 'cnn_model.pt')
    model.load_state_dict(torch.load('cnn_model.pt'))
    dev_loss, dev_acc = evaluate(model, test_data, video_test_data, batch_size, criterion)
    print(f'FINAL TEST  | test. accuracy={dev_acc} | test_loss={dev_loss}')

batch_size = 16
is_debug = True
model = LSTMCNNClassifierVideo( 714,  100, 10, 50, 2, 0.0)
model_phys = LSTMCNNClassifierVideo( embed_dim = 40,  n_filter=10, filter_size=100, hidden_dim=5, num_label=2, dropout_prob=0.0)
model = model.to(device)
model_phys = model_phys.to(device)

combine_model = ModeFusion(model, model_phys, 2, 2, 2)
combine_model = combine_model.to(device)

if is_debug:
    deap_df = read_phys_data(data_path)
    print("read phys data")
    video_df = read_video_data(video_features_path, is_debug = True)
    print("video data read")
else:
    deap_df = read_phys_data(data_path_atlas)
    video_df = read_video_data(video_features_path_atlas, is_debug=False)

print(len(video_df))
users = list(deap_df["user"].unique())
users = [x for x in users if x<"s23"]
print(users)
train_users, valid_users, test_users = users[:int(0.8 * len(users))], users[int(0.8 * len(users)):int(0.9 * len(users))], users[int(0.9 * len(users)):]
train_data = deap_df[deap_df.user.isin(train_users)]
eval_data = deap_df[deap_df.user.isin(valid_users)]
test_data = deap_df[deap_df.user.isin(test_users)]
video_train_data = video_df[video_df.user.isin(train_users)]
video_eval_data = video_df[video_df.user.isin(valid_users)]
video_test_data = video_df[video_df.user.isin(test_users)]

print(len(test_data),len(eval_data), len(train_data))
train(train_data, video_train_data, eval_data, video_eval_data, test_data, video_test_data, combine_model, batch_size)
