import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class EFLSTM(nn.Module):
    def __init__(self, d, h, output_dim, dropout):  # , n_layers, bidirectional, dropout):
        super(EFLSTM, self).__init__()
        self.h = h
        self.lstm = nn.LSTMCell(d, h)
        self.fc1 = nn.Linear(h, h)
        self.fc2 = nn.Linear(h, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is batch_size * seq_len * dim_features
        x = x.transpose(0, 1)
        # x is t x n x d
        t = x.shape[0]
        n = x.shape[1]
        self.hx = torch.zeros(n, self.h).to(device)  # .cuda()
        self.cx = torch.zeros(n, self.h).to(device)  # .cuda()
        all_hs = []
        all_cs = []
        for i in range(t):
            self.hx, self.cx = self.lstm(x[i], (self.hx, self.cx))
            all_hs.append(self.hx)
            all_cs.append(self.cx)
        # last hidden layer last_hs is n x h
        last_hs = all_hs[-1]
        pre_output = F.relu(self.fc1(last_hs))
        output = self.dropout(pre_output)
        output = self.fc2(output)
        return output, pre_output

    def predict(self, X_test):
        self.eval()
        with torch.no_grad():
            batch_X = torch.Tensor(X_test).to(device)
            predictions, pre_predictions = self.forward(batch_X)
            predictions = predictions.squeeze(1)
            pre_predictions = pre_predictions#.view(-1)
            predictions, pre_predictions = predictions.cpu().data.numpy(), pre_predictions.cpu().data.numpy()
        return predictions, pre_predictions