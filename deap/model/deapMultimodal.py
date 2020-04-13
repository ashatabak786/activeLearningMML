import torch
from utils import read_video_data, read_phys_data, read_batches, ff
from constants import data_path, data_path_atlas, video_features_path, video_features_path_atlas
# from deap.model.utils import read_video_data, read_phys_data, read_batches
# from deap.model.constants import data_path, data_path_atlas, video_features_path, video_features_path_atlas
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix

class ModeFusion3(nn.Module):
    def __init__(self, video_model, phys_model1, phys_model2, out1_dim, out2_dim, out3_dim, num_label):
        super(ModeFusion3, self).__init__()
        self.video_model = video_model
        self.phys_model1 = phys_model1
        self.phys_model2 = phys_model2
        self.combine = nn.Linear(out1_dim + out2_dim + out3_dim, num_label)

    def forward(self, vid_X, phys_X):
        phys_X1 = phys_X[:,:, :32]
        phys_X2 = phys_X[:, :, 32:]
        out_vid = self.video_model(vid_X)
        out_phys1 = self.phys_model1(phys_X1)
        out_phys2 = self.phys_model2(phys_X2)
        out = torch.cat([out_vid,out_phys1, out_phys2],1)
        out = self.combine(out)
        return out

    def predict(self, vid_X, phys_X):
        with torch.no_grad():
            out = self.forward(vid_X, phys_X)
            predicted_classes = torch.argmax(out, dim=-1).cpu().numpy()
            return out, predicted_classes


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

        self.linear = nn.Linear(n_filter + 2*hidden_dim , num_label)
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


def conf_matrix(preds, y):
    y_cpu = y.view(-1).cpu().numpy()
    try:
        tn, fp, fn, tp = confusion_matrix(y_cpu, preds).ravel()
        return tn, fp, fn, tp
    except Exception as e:
        # print(e, Counter(preds), Counter(y_cpu))
        if 0 in set(preds):
            return len(preds), 0, 0 ,0
        else:
            return 0, 0, 0, len(preds)


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
    all_tp, all_fp, all_tn, all_fn = 0, 0, 0, 0
    num_batches = 0
    all_labels = []
    all_preds = []
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
        tn, fp, fn, tp = conf_matrix(pred_classes, batch_labels)
        all_tp += tp
        all_tn += tn
        all_fp += fp
        all_fn += fn
        num_batches += len(batch_features)
        all_labels += list(batch_labels.view(-1).cpu().numpy())
        all_preds += list(pred_classes)
    # print("pred, label", Counter(all_preds), Counter(all_labels))
    epoch_acc_dict = {k:v/num_batches for k,v in epoch_acc_dict.items()}
    bal_acc = ((all_tn / (all_tn + all_fp)) + (all_tp / (all_tp + all_fn))) / 2
    recall = all_tp / (all_tp + all_fn)
    if all_tp + all_fp > 0:
        precision = all_tp / (all_tp + all_fp)
        fscore = (2 * precision * recall) / (precision + recall)
    else:
        precision = -1
        fscore = -1
    return epoch_loss / num_batches, ff(epoch_acc_dict[0]), ff(bal_acc), ff(fscore)


def train(train_data_phys, video_train_data, eval_data, video_eval_data, test_data, video_test_data, model, batch_size, num_epochs, model_name):
    model = model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=0, factor=0.33, cooldown=0, min_lr=1e-6, verbose=True)
    best_valid_loss = float('inf')
    best_eval_acc = 0.0
    for epoch_id in range(num_epochs):
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
        train_loss, train_acc, train_bal, train_f = evaluate(model, train_data, video_train_data, batch_size, criterion)
        dev_loss, dev_acc, dev_bal, dev_f = evaluate(model, eval_data, video_eval_data, batch_size, criterion)
        scheduler.step(dev_loss)
        print(f'TRAIN Epoch {epoch_id} | balanced train. accuracy={train_bal} | train fscore={train_f} | train_loss={ff(train_loss,n=8)} | train. accuracy={train_acc} |')
        print(f'Epoch {epoch_id} | balanced dev. accuracy={dev_bal} | dev fscore={dev_f} | dev_loss={ff(dev_loss, n=8)} | dev. accuracy={dev_acc} |')
        if dev_loss < best_valid_loss:
            best_valid_loss = dev_loss
            torch.save(model.state_dict(), model_name+'.pt')
    model.load_state_dict(torch.load(model_name+'.pt'))
    dev_loss, dev_acc, dev_bal, dev_f = evaluate(model, test_data, video_test_data, batch_size, criterion)
    print(f'TEST - balanced test. accuracy={dev_bal} | test fscore={dev_f} | test_loss={ff(dev_loss, n=8)} | test acc={dev_acc} |')

num_epochs = 2
batch_size = 16
is_debug = False
num_video_features = 714
num_phys_features1 = 32
num_phys_features2 = 8
num_video_filters = 10
video_filter_size = 10
video_lstm_h = 5
out_size_video = 2
num_phys_filters = 10
phys_filter_size = 100
phys_lstm_h = 5
phys_out_size = 2
dropout = 0.0
model_name = "twoModeWitNoShortcutClass0"
if not is_debug:
    num_epochs = 10
    num_video_filters = 400
    num_phys_filters = 400
    video_lstm_h = 128
    phys_lstm_h = 128
    phys_out_size = 10
    out_size_video = 10
    dropout = 0.0
    phys_filter_size = 128

model = LSTMCNNClassifierVideo(embed_dim=num_video_features,  n_filter=num_video_filters,
                               filter_size=video_filter_size, hidden_dim=video_lstm_h,
                               num_label=out_size_video, dropout_prob=dropout)
model_phys1 = LSTMCNNClassifierVideo( embed_dim = num_phys_features1,  n_filter=num_phys_filters,
                                     filter_size=phys_filter_size, hidden_dim=phys_lstm_h,
                                     num_label=phys_out_size, dropout_prob=dropout)
model_phys2 = LSTMCNNClassifierVideo( embed_dim = num_phys_features2,  n_filter=num_phys_filters,
                                     filter_size=phys_filter_size, hidden_dim=phys_lstm_h,
                                     num_label=phys_out_size, dropout_prob=dropout)
model = model.to(device)
model_phys1 = model_phys1.to(device)
model_phys2 = model_phys2.to(device)
combine_model = ModeFusion3(model, model_phys1, model_phys2, out_size_video, phys_out_size, phys_out_size, 2)
combine_model = combine_model.to(device)

if is_debug:
    deap_df = read_phys_data(data_path)
    print("read phys data")
    video_df = read_video_data(video_features_path, is_debug = True)
    print("video data read")
else:
    deap_df = read_phys_data(data_path_atlas)
    video_df = read_video_data(video_features_path_atlas, is_debug=False, redo = False)

print(len(video_df))
users = list(deap_df["user"].unique())
users = [x for x in users if x<"s23"]
print(len(users))
train_users, valid_users, test_users = users[:int(0.8 * len(users))], users[int(0.8 * len(users)):int(0.9 * len(users))], users[int(0.9 * len(users)):]
print("train, val, test users", len(train_users), len(valid_users), len(test_users))
train_data = deap_df[deap_df.user.isin(train_users)]
eval_data = deap_df[deap_df.user.isin(valid_users)]
test_data = deap_df[deap_df.user.isin(test_users)]
video_train_data = video_df[video_df.user.isin(train_users)]
video_eval_data = video_df[video_df.user.isin(valid_users)]
video_test_data = video_df[video_df.user.isin(test_users)]
if is_debug:
    video_eval_data = video_test_data
    eval_data = test_data

print("train-test-eval - phys", len(train_data), len(test_data),len(eval_data))
print("train-test-eval - video", len(video_train_data), len(video_test_data),len(video_eval_data))

train(train_data, video_train_data, eval_data, video_eval_data, test_data, video_test_data, combine_model, batch_size, num_epochs, model_name)
