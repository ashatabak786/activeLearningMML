import torch
from utils import read_video_data, read_phys_data, read_batches, ff, plot_losses
from constants import data_path, data_path_atlas, video_features_path, video_features_path_atlas
# from deap.model.utils import read_video_data, read_phys_data, read_batches
# from deap.model.constants import data_path, data_path_atlas, video_features_path, video_features_path_atlas
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("main")

class BaselineVideo(nn.Module):



def conf_matrix(preds, y):
    y_cpu = y.view(-1).cpu().numpy()
    try:
        tn, fp, fn, tp = confusion_matrix(y_cpu, preds).ravel()
        return tn, fp, fn, tp
    except Exception as e:
        print(e, Counter(preds), Counter(y_cpu))
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
    nc = 2
    epoch_acc_dict = { i:0 for i in range(nc)}
    all_tp, all_fp, all_tn, all_fn =  { i:0 for i in range(nc)},  { i:0 for i in range(nc)},  { i:0 for i in range(nc)},  { i:0 for i in range(nc)}
    num_batches = 0
    all_labels = { i:[] for i in range(nc)}
    all_preds = { i:[] for i in range(nc)}
    for batch_features, video_batch_features, batch_labels in read_batches(eval_dataset, eval_dataset_video, batch_size, False, device, is_multi=True):
        batch_features = batch_features.permute((0, 2, 1))
        outs, pred_classes = model.predict(video_batch_features, batch_features)
        losses = []
        for i, tup in enumerate(zip(outs, pred_classes)):
            out, pred_class = tup
            losses.append(criterion(
                out,  # (batch_size , num_classes)
                batch_labels[:,i:i+1].view(-1)  # (batch_size * 1)
            ))
            epoch_acc_dict[i] += categorical_accuracy(pred_class, batch_labels[:,i:i+1])
            tn, fp, fn, tp = conf_matrix(pred_class, batch_labels[:,i:i+1])
            all_tp[i] += tp
            all_tn[i] += tn
            all_fp[i] += fp
            all_fn[i] += fn
            all_labels[i] += list(batch_labels[:,i:i+1].view(-1).cpu().numpy())
            all_preds[i] += list(pred_class)
        loss = sum(losses)
        epoch_loss += loss.item()
        num_batches += len(batch_features)
    bal_acc_dict = {}
    fscore_dict = {}
    for i in range(nc):
        print(i,"pred, label", Counter(all_preds[i]), Counter(all_labels[i]))

        bal_acc = ((all_tn[i] / (all_tn[i] + all_fp[i])) + (all_tp[i] / (all_tp[i] + all_fn[i]))) / 2
        recall = all_tp[i] / (all_tp[i] + all_fn[i])
        if all_tp[i] + all_fp[i] > 0:
            precision = all_tp[i] / (all_tp[i] + all_fp[i])
            fscore = (2 * precision * recall) / (precision + recall)
        else:
            precision = -1
            fscore = -1
        bal_acc_dict[i] = ff(bal_acc)
        fscore_dict[i] = ff(fscore)
        epoch_acc_dict[i] = ff(epoch_acc_dict[i]/num_batches)
    return epoch_loss / num_batches, epoch_acc_dict, fscore_dict, epoch_acc_dict


def train(train_data, video_train_data, eval_data, video_eval_data, test_data, video_test_data, model, batch_size, num_epochs, model_name):
    model = model.train()
    weights = [680/261, 680/419]
    class_weights = torch.tensor(weights, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1000)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.33, cooldown=0, min_lr=1e-6, verbose=True)
    best_valid_loss = float('inf')
    best_eval_acc = 0.0
    eval_pat = 10
    dev_loss_list = []
    train_loss_list = []
    train_acc_list = []
    dev_acc_list = []
    step_num = 0
    for epoch_id in range(num_epochs):
        for batch_features, video_batch_features, batch_labels in read_batches(train_data, video_train_data, batch_size, True, device, is_multi=True):
            model = model.train()
            optimizer.zero_grad()
            step_num+=1
            batch_features = batch_features.permute((0,2,1))
            outs = model(video_batch_features, batch_features)
            losses = []
            for i, out in enumerate(outs):
                losses.append(criterion(
                    out,  # (batch_size , num_classes)
                    batch_labels[:, i:i + 1].view(-1)  # (batch_size * 1)
                ))
            loss = sum(losses)
            loss.backward()
            optimizer.step()
            if step_num % eval_pat == 0:
                # print("plot now")
                train_loss, train_acc, train_bal, train_f = evaluate(model, train_data, video_train_data, batch_size, criterion)
                dev_loss, dev_acc, dev_bal, dev_f = evaluate(model, eval_data, video_eval_data, batch_size, criterion)
                train_loss_list.append(train_loss)
                dev_loss_list.append(dev_loss)
                train_acc_list.append(train_bal)
                dev_acc_list.append(dev_bal)
                plot_losses(model_name, train_loss_list, dev_loss_list, train_acc_list, dev_acc_list, step_num)
        train_loss, train_acc, train_bal, train_f = evaluate(model, train_data, video_train_data, batch_size, criterion)
        dev_loss, dev_acc, dev_bal, dev_f = evaluate(model, eval_data, video_eval_data, batch_size, criterion)
        scheduler.step(dev_loss)
        print()
        print(f'TRAIN Epoch {epoch_id} | balanced train. accuracy={train_bal} | train fscore={train_f} | train_loss={ff(train_loss,n=8)} | train. accuracy={train_acc} |')
        print(f'Epoch {epoch_id} | balanced dev. accuracy={dev_bal} | dev fscore={dev_f} | dev_loss={ff(dev_loss, n=8)} | dev. accuracy={dev_acc} |')
        if dev_loss < best_valid_loss:
            best_valid_loss = dev_loss
            torch.save(model.state_dict(), model_name+'.pt')
    model.load_state_dict(torch.load(model_name+'.pt'))
    dev_loss, dev_acc, dev_bal, dev_f = evaluate(model, test_data, video_test_data, batch_size, criterion)
    print(f'TEST - balanced test. accuracy={dev_bal} | test fscore={dev_f} | test_loss={ff(dev_loss, n=8)} | test acc={dev_acc} |')
    plot_losses(model_name, train_loss_list, dev_loss_list, train_acc_list, dev_acc_list, step_num, test_acc=dev_bal, test_loss=dev_loss)



is_debug = True

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

