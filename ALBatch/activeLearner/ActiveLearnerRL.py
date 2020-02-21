import torch
import torch.optim as optim
from ALBatch.activeLearner.BaseRLAgent import BaseAgent
import numpy as np
from ALBatch.activeLearner.DQNModel import ExperienceReplayMemory, DQN
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



class SimpleAgent(BaseAgent):
    def __init__(self, static_policy=False, env=None, config=None, log_dir='./logs'):
        super(SimpleAgent, self).__init__(env=env, config=config, log_dir=log_dir)
        self.device = config.device
        self.learning_history_loss = []
        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ

        self.static_policy = static_policy

        self.num_feats = env.get_observed_state().shape[1]
        self.num_actions = len(env.action_space)
        self.env = env

        # define model, target model
        self.declare_networks()

        # copy model params to target model
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        # are we evaluating or training??
        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        # define replay memory
        self.declare_memory()

    def declare_networks(self):
        self.model = DQN(self.num_feats, self.num_actions)
        self.target_model = DQN(self.num_feats, self.num_actions)

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)

    def append_to_replay(self, s, a, r, s_):

        # store SARS' in memory?
        self.memory.push((s, a, r, s_))

    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        # print("what is transition? - ",
        #       batch_state.shape, batch_action.shape,
        #       batch_reward.shape, batch_next_state.shape)

        shape = (-1, self.num_feats)

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)

        # check if any of the states are final? - Might not be applicable to our case?
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device,
                                      dtype=torch.bool)
        try:  # sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device,
                                                 dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        # estimate
        current_q_values = self.model(batch_state).gather(1, batch_action)

        # target
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
            expected_q_values = batch_reward + self.gamma * max_next_q_values

        diff = (expected_q_values - current_q_values)
        #todo: bring to cpu?
        loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, frame=0):
        if self.static_policy:
            return None

        self.append_to_replay(s, a, r, s_)

        if frame < self.learn_start or frame % self.update_freq != 0:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)
        self.learning_history_loss.append(loss.item())
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        self.save_td(loss.item(), frame)
        self.save_sigma_param_magnitudes(frame)

    def get_action(self, s, eps=0.1):
        # epsilon greedy action selection
        with torch.no_grad():
            if np.random.random() >= eps :
                X = torch.tensor(s, device=self.device, dtype=torch.float)
                a = self.model(X).max(1)[1]#.view(1, 1)
                # state -> action!
                return a
            else:
                return np.random.randint(0, self.num_actions, size = len(s))
                    #np.random.randint(0, self.num_actions)

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        pass

    def reset_hx(self):
        pass

    def make_static(self):
        self.static_policy = True
        self.model.eval()
        self.target_model.eval()

    def make_dynamic(self):
        self.static_policy = False
        self.model.train()
        self.target_model.train()

    def MSE(self, x):
        return 0.5 * x.pow(2)