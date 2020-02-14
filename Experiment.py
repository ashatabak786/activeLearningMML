from activeLearner.ActiveLearnerRL import SimpleAgent
from utils.RLhyperparams import Config
from baseLearners.DataLoaders import  load_saved_data
from baseLearners.MosiEarlyFusionLSTM import EFLSTM
import torch
import math
from activeLearner.activeLearningEnv import ActiveEnvSingleState
import matplotlib.pyplot as plt

def get_config():
    config = Config()

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # epsilon variables
    config.epsilon_start = 1.0
    config.epsilon_final = 0.01
    config.epsilon_decay = 500
    config.epsilon_by_frame = lambda frame_idx: config.epsilon_final + (
                config.epsilon_start - config.epsilon_final) * math.exp(-1. * frame_idx / config.epsilon_decay)
    # misc agent variables
    config.GAMMA = 0.9
    config.LR = 1e-3

    # memory
    config.TARGET_NET_UPDATE_FREQ = 10
    config.EXP_REPLAY_SIZE = 10000
    config.BATCH_SIZE = 32

    # Learning control variables
    config.LEARN_START = 50
    config.MAX_FRAMES = 1000000
    config.UPDATE_FREQ = 1

    # data logging parameters
    config.ACTION_SELECTION_COUNT_FREQUENCY = 100
    return config


if __name__ == "__main__":
    config = get_config()
    base_path = "/Users/ashatabak/cmu/secondSem/active/code/MFN/data/"
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_saved_data(base_path)
    budget = 50
    sample = budget + 20
    train_data = X_train[:sample]
    train_labels = y_train[:sample]
    valid_data = X_train
    valid_labels = y_train

    # define base model
    d = train_data.shape[2]
    h = 5
    t = train_data.shape[1]
    output_dim = 1
    dropout = 0.25
    base_model = EFLSTM(d, h, output_dim, dropout)
    base_model = base_model.to(config.device)

    # define env
    env = ActiveEnvSingleState(base_model, train_data, train_labels, budget, valid_data, valid_labels)

    # define the active learning agent
    agent  = SimpleAgent(env=env, config=config)

    # start the training loop!!
    episode_reward = 0
    epsilon = 0.2
    is_train_RL = True
    if is_train_RL:
        observation = env.reset()
        env._before_first_action(10)
        env.snapshot()
        for frame_idx in range(1, min(1000,config.MAX_FRAMES + 1)):
            epsilon = config.epsilon_by_frame(frame_idx)
            action = agent.get_action(observation, epsilon)
            agent.save_action(action, frame_idx) #log action selection

            # send the action to env
            prev_observation = observation
            observation, reward, done = env.step(action)
            observation = None if done else observation

            agent.update(prev_observation, action, reward, observation, frame_idx)
            episode_reward += reward

            if done:
                print("Epdisode restart!!!!!!!!!!!!!!!!!!!")
                agent.finish_nstep()
                agent.reset_hx()
                observation = env.reset()
                env._before_first_action(10)
                agent.save_reward(episode_reward)
                episode_reward = 0
    plt.plot(agent.learning_history_loss)
    plt.title("RL loss history")
    plt.show()
    # test our agent on the valid set
    maes = []
    agent.make_static()
    env.begin_valid()
    observation = env.get_observed_state()
    for extra_label_c in range(200):
        mae = env.evaluate(X_test, y_test)
        maes.append(mae)
        action = agent.get_action(observation, 0.1)

        observation, reward, done = env.step(action)

    plt.plot(maes)
    plt.title("LAL-RL")
    plt.show()

    maes = []
    # agent.make_static()
    env.reset(from_snapshot=True)
    env.begin_valid()
    observation = env.get_observed_state()
    for extra_label_c in range(200):
        mae = env.evaluate(X_test, y_test)
        maes.append(mae)
        action = agent.get_action(observation, 0.1)

        observation, reward, done = env.step(action)

    plt.plot(maes)
    plt.title("random")
    plt.show()





