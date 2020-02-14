from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
seed = 123
np.random.seed(seed)


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class ActiveEnvSingleState():
    """
    dataset environment simulator
    actions in {0: dont label, 1: label}
    """

    def __init__(self, base_model, train_data, train_labels, budget, valid_data, valid_labels):
        self.base_model = base_model
        self.train_data = train_data
        self.train_labels = train_labels
        self.budget = budget
        self.valid_data = valid_data
        self.valid_labels = valid_labels
        self.predicted_labels = {}
        self.default_reward = -0.99
        self.idx_known = []
        self.idx_unknown = list(range(len(self.train_data)))
        self.idx_predicted = []
        torch.save(self.base_model.state_dict(), './models/first_base_model.pt')
        self.memory = {}
        # idx state
        self.state = None
        # softmax/observed state used for DQN
        self.observed_state = None
        self.action_space = [0, 1]

        # set optimizer and criterion for base model - input later!
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=0.001)
        self.criterion = nn.L1Loss()
        self.criterion = self.criterion.to(device)

        self.snapshot()
        self._before_first_action(20)

    def begin_valid(self):
        self.idx_known = list(range(len(self.train_data)))
        self.train_data = self.valid_data
        self.train_labels = self.valid_labels
        self.idx_unknown = [x for x in range(len(self.train_data)) if x not in self.idx_known]
        self.idx_predicted = []
        self.state = None
        #set a new state from this valid data
        self._get_curr_state()


    def _before_first_action(self, k):
        # add k points to the labelled dataset random
        # train the base model
        state = self._get_curr_state()
        print("current state is ", state, self.idx_unknown)
        for i in range(k):
            # label 20 points to start with
            self._label_current(1)
            self._set_next_state()
        print("init done with 20 points")
        self.retrain_base(k)
        print("init model trained")

    #     1/0

    def _set_next_state(self):
        # randomly pick next state where state is index in original dataset
        select_idx = np.random.randint(len(self.idx_unknown))
        self.state = self.idx_unknown[select_idx]

    def get_observed_state(self):
        data_mat = self.train_data[self.state:self.state+1]
        predictions, pre_predictions = self.base_model.predict(data_mat)
        return pre_predictions  # TODO: add more features

    def _get_curr_state(self):
        if self.state == None:
            self._set_next_state()
        return self.state

    def _label_current(self, action):
        # add point to known or unknown set
        # currently adds pointwise
        # return reward
        reward = self.default_reward
        if action == 1:
            # add subtract label.
            # print(self.idx_known, self.idx_unknown)
            self.idx_known.append(self.state)
            self.idx_unknown.remove(self.state)
            print("labelling happening")
        elif action == 0:
            print("active learning happening")
            data_mat = self.train_data[self.state:self.state+1]
            predictions, pre_predictions = self.base_model.predict(data_mat)

            self.predicted_labels[self.state] = predictions[0]
            self.idx_predicted.append(self.state)

            reward = -(abs(predictions[0] - self.train_labels[self.state])/0.5)**2
            # if predictions[0] == self.train_labels[self.state]:
            #     reward = 1
            # else:
            #     reward = -1

        return reward

    def step(self, action):
        # return next state, reward, done
        reward = self._label_current(action)
        done = False
        observation = None
        try:
            # refresh next state randomly - stochastic environment
            self._set_next_state()
            observation = self.get_observed_state()
        except:
            done = True
        if len(self.idx_known) >= self.budget:
            done = True
        self.retrain_base(8)
        return observation, reward, done

    def retrain_base(self, batchsize):
        self.base_model.load_state_dict(torch.load('./models/first_base_model.pt'))
        # use the dev set to train the best generalizable model! use that as the best model

        all_known_labels = self.idx_known + self.idx_predicted
        label_copy = deepcopy(self.train_labels)
        for idx in self.idx_predicted:
            label_copy[idx] = self.predicted_labels[idx]

        X_train = self.train_data[all_known_labels]
        y_train = label_copy[all_known_labels]
        # reset labels according to predicted by our agent!

        split = int(0.8 * len(X_train))
        X_train_train, X_train_dev = X_train[:split], X_train[split:]
        y_train_train, y_train_dev = y_train[:split], y_train[split:]

        self.base_model.train()

        total_n = X_train_train.shape[0]
        if batchsize > total_n:
            batchsize = total_n
        num_batches = int(total_n / batchsize)
        best_valid_loss = float('inf')
        for epoch_id in range(20):
            epoch_loss = 0
            # print("starting epoch ", epoch_id)
            for batch in range(num_batches):
                start = batch * batchsize
                end = (batch + 1) * batchsize
                self.optimizer.zero_grad()
                # print(device)
                batch_X = torch.Tensor(X_train_train[start:end]).to(device)  # .cuda()
                batch_y = torch.Tensor(y_train_train[start:end]).to(device)  # .cuda()
                predictions, pre_predictions = self.base_model.forward(batch_X)
                predictions = predictions.squeeze(1)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            epoch_loss = epoch_loss / num_batches
            eval_loss = self.evaluate(X_train_dev, y_train_dev)
            if eval_loss < best_valid_loss:
                best_valid_loss = eval_loss
                torch.save(self.base_model.state_dict(), './models/base_model.pt')

        # reload the best model over the epochs
        self.base_model.load_state_dict(torch.load('./models/base_model.pt'))
        # print("retrained baswe model")


    def evaluate(self, X_valid, y_valid, verbose=False):
        epoch_loss = 0

        self.base_model.eval()
        with torch.no_grad():
            batch_X = torch.Tensor(X_valid).to(device)
            batch_y = torch.Tensor(y_valid).to(device)
            predictions, pre_predictions = self.base_model.forward(batch_X)
            epoch_loss = self.criterion(predictions.view(-1), batch_y).item()
            predictions = predictions.squeeze(1)
            # print(predictions, y_valid)
            predictions = predictions.cpu().data.numpy()
            mae = np.mean(np.absolute(predictions - y_valid))
            # print("mae: ", mae)
            if verbose:
                mult = round(sum(np.round(predictions) == np.round(y_valid)) / float(len(y_valid)), 5)
                print("mult_acc: ", mult)
                true_label = (y_valid >= 0)
                predicted_label = (predictions >= 0)
                print("Confusion Matrix :")
                print(confusion_matrix(true_label, predicted_label))
                print("Classification Report :")
                print(classification_report(true_label, predicted_label, digits=5))
                print("Accuracy ", accuracy_score(true_label, predicted_label))

        return mae

    def snapshot(self):
        self.memory["idx_known"] = deepcopy(self.idx_known)
        self.memory["idx_unknown"] = deepcopy(self.idx_unknown)

    def reset(self, from_snapshot=False):
        if from_snapshot:
            self.idx_known = deepcopy(self.memory["idx_known"])
            self.idx_unknown = deepcopy(self.memory["idx_known"])
        else:
            self.idx_known = []
            self.idx_unknown = list(range(len(self.train_data)))
        self._get_curr_state()
        return self.get_observed_state()
