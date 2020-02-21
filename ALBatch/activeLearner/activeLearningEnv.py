from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
import random
seed = 123
np.random.seed(seed)


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class ActiveEnvMultitate():
    """
    dataset environment simulator
    actions in {0: dont label, 1: label}
    """

    def __init__(self, base_model, train_data, train_labels, budget, valid_data, valid_labels, batch_size):
        self.base_model = base_model
        self.batch_size = batch_size
        self.train_data = train_data
        self.train_labels = train_labels
        self.budget = budget
        self.valid_data = valid_data
        self.valid_labels = valid_labels
        self.predicted_labels = {}
        self.default_reward = -2
        self.idx_known = []
        self.idx_unknown = list(range(len(self.train_data)))
        self.idx_predicted = []
        torch.save(self.base_model.state_dict(), './models/first_base_model.pt')
        self.memory = {}
        # idx state - set a batch of random sample points
        self.state = None
        # softmax/observed state used for DQN
        self.observed_state = None
        self.action_space = [0, 1]

        # set optimizer and criterion for base model - input later!
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=0.001)
        self.criterion = nn.L1Loss()
        self.criterion = self.criterion.to(device)
        self.past_states = set()
        self.snapshot()
        self._before_first_action(20)

    def begin_valid(self):
        """
        Set known, predicted(None) and unknown(all) index for the experiment.
        :return: None
        """
        self.idx_known = list(range(len(self.train_data)))
        self.train_data = self.valid_data
        self.train_labels = self.valid_labels
        self.idx_unknown = [x for x in range(len(self.train_data)) if x not in self.idx_known]
        self.idx_predicted = []
        self.state = None
        #set a new state from this valid data
        self._get_curr_state()


    def _before_first_action(self, k):
        """Initialize base state by labelling k points"""
        # add k points to the labelled dataset random
        # train the base model
        state = self._get_curr_state()
        print("current state is ", state, self.idx_unknown)
        for i in range(int(k/self.batch_size)):
            # label 20 points to start with
            self._label_current(np.ones(self.batch_size))
            self._set_next_state()
        print("init done with 20 points")
        self.retrain_base(k)
        print("init model trained")

    #     1/0

    def _set_next_state(self):
        # randomly pick next state where state is index in original dataset
        # need without repalcement
        if self.state is not None:
            self.past_states.update(self.state)
        if len(self.idx_unknown) < self.batch_size:
            self.state = None
            return
        options = list(range(len(self.idx_unknown)))
        random.shuffle(options)
        select_idx_batch = options[:self.batch_size]
        # select_idx_batch = np.random.choice(len(self.idx_unknown), size = self.batch_size)
        assert(len(list(select_idx_batch)) == len(set(select_idx_batch)))
        self.state = np.array([self.idx_unknown[select_idx] for select_idx in select_idx_batch])
        assert(all([x not in self.past_states for x in list(self.state)]))

    def get_observed_state(self):
        data_mat = self.train_data[self.state]
        predictions, pre_predictions = self.base_model.predict(data_mat)
        return pre_predictions  # TODO: add more features

    def _get_curr_state(self):
        if self.state is None:
            self._set_next_state()
        return self.state

    def _print_labeling_state(self):
        print("Num labelled :", len(self.idx_known), "Num predicted :", len(self.idx_predicted))

    def _label_current(self, action):
        """
        take action for each observation and return reward.
        :param action: batch sized action space
        :return: reward for each action
        """
        reward = np.zeros(action.shape[0])
        state_idx_action_1 = [i for i,state_idx in enumerate(self.state) if action[i]==1]
        state_idx_action_0 = [i for i, state_idx in enumerate(self.state) if action[i] == 0]
        if len(state_idx_action_1) > 0:
            # add subtract label.
            # print(self.idx_known, self.idx_unknown)
            self.idx_known+=list(self.state[state_idx_action_1])

            reward[state_idx_action_1] = [self.default_reward]*len(state_idx_action_1)
        if len(state_idx_action_0) > 0:
            state_action_0 = self.state[state_idx_action_0]
            data_mat = self.train_data[state_action_0]
            predictions, pre_predictions = self.base_model.predict(data_mat)
            for idx, state_idx in enumerate(state_action_0):
                self.predicted_labels[state_idx] = predictions[idx]
            [self.idx_predicted.append(state_idx) for state_idx in state_action_0]
            reward[state_idx_action_0] = -(abs(predictions - self.train_labels[state_action_0]))**2
        [self.idx_unknown.remove(state_idx) for state_idx in list(self.state)]

        self._print_labeling_state()
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
        if len(self.idx_known) - len(self.idx_predicted) >= self.budget:
            done = True
        mae = self.retrain_base(16)
        return observation, reward, done, mae

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
        for epoch_id in range(5):
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
                # mae = np.mean(np.absolute(predictions - y_test))
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
        eval_loss = self.evaluate(self.valid_data, self.valid_labels)
        return eval_loss

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
        self.past_states = set()
        self._get_curr_state()
        self.idx_predicted = []
        return self.get_observed_state()
