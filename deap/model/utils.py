import os
import pickle
import pandas as pd
import numpy as np
import pdb
import torch
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from videoFeatureExtraction import categorize_columns
data_path_atlas = "/work/aashfaq/datasets/deap/data_preprocessed_python/"
data_path = "/Users/ashatabak/cmu/datasets/deap/data_preprocessed_python/"
video_features_path = "/Users/ashatabak/cmu/datasets/deap/video/"
video_features_path_chkpt = "/Users/ashatabak/cmu/datasets/deap/video/video.csv"
video_features_path_chkpt_atlas= "/work/aashfaq/datasets/deap/openFaceFeatures/processed/video.csv.chk"
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def ff(f, n=4):
    return ("{:4."+str(n)+"f}").format(f)




def read_video_data(video_features_path, is_debug=False, redo = False):
    chk_path = video_features_path_chkpt
    if not is_debug:
        chk_path = video_features_path_chkpt_atlas
    try:
        if not redo:
            final_df = pd.read_csv(chk_path)
            # tODO: check weird unnamed column error
            final_df = final_df.iloc[:, 1:]
            return final_df
    except Exception as e:
        print("try reading from individual files", e)
    timestamp_period = 0.1
    all_features_df = []
    for root, subdirs, files in os.walk(video_features_path):
        for file in files:

            filepath = root + "/" + file

            if filepath.endswith(".csv") and "trial" in filepath:
                print(filepath)
                arr = file.split("_")
                user = arr[0]
                trial = arr[1].split(".")[0]
                try:
                    features = pd.read_csv(filepath)
                except:
                    print("cant read:", filepath)
                    continue
                normalize_cols = categorize_columns(features)
                features_subsampler = features.apply(lambda row: 100 / 1 * row[" timestamp"] % 100 == 0, axis=1)
                features[normalize_cols] = features[normalize_cols] - features[normalize_cols].mean()
                # pdb.set_trace()
                features[" face_id"] = features.apply(lambda row: int(user.strip("s")), axis = 1)
                features["user"] = features.apply(lambda row: user, axis=1)
                features["trial"] = features.apply(lambda row: int(trial[-2:]), axis=1)

                features = features[features_subsampler]
                print("after filtering", len(features))
                # print("adding feature size =", len(features))
                # print(features)
                # pdb.set_trace()
                all_features_df.append(features)
    final_df = pd.concat(all_features_df, axis=0)
    print(len(final_df))
    final_df.to_csv(chk_path)
    return final_df

def read_phys_data(data_path):
    deap_df_dict = {
        "labels": [],
        "data": [],
        "user": [],
        "trial": []
    }
    count = 0
    for root, subdirs, files in os.walk(data_path):
        for file in files:
            # print(root, subdirs, files)
            if len(subdirs) == 0:
                filepath = root + "/" + file
                # print(filepath)
                user1_physio = pickle.load(open(filepath, 'rb'), encoding='bytes')
                user = file.split(".")[0]
                for trial_id in range(len(user1_physio[b'labels'])):
                    deap_df_dict["user"].append(user)
                    deap_df_dict["trial"].append(1+trial_id)
                    deap_df_dict["labels"].append(user1_physio[b'labels'][trial_id])
                    deap_df_dict["data"].append(user1_physio[b'data'][trial_id])

    deap_df = pd.DataFrame.from_dict(deap_df_dict)
    print(len(deap_df.loc[:, ["trial", "user"]]))
    return deap_df

def sample_video_df(video_data_df, video_keys):
    """return second level frames features 60*714 features for each image"""
    feature_columns = list(video_data_df.columns)[:-2]
    output_features =[]
    missing_keys = 0
    missing_key_vals = []
    for user, trial in video_keys:
        user_sample = video_data_df["user"] == user
        trial_sample = video_data_df["trial"] == trial
        one_data_point_df = video_data_df[(user_sample) & (trial_sample)]
        if len(one_data_point_df) > 0 :
            # print()
            features = one_data_point_df[feature_columns]
            one_data_np = features.values.astype(np.float32)
        else:
            #TODO: infer the shape from inputs
            missing_keys+=1
            missing_key_vals.append((user, trial))
            one_data_np = np.zeros((60,714)).astype(np.float32)
        output_features.append(one_data_np)
    if missing_keys > 0:
        # print("pct keys missing = ", missing_keys, len(video_keys), missing_key_vals)
        pass
    else:
        pass
        # print("NO KEYS MISSING")
    return np.array(output_features)


def read_batches(data_df, video_data_df, batch_size, shuffle, device, is_multi = False ):
    batch_num = int(np.ceil(len(data_df) / batch_size))
    index_array = list(range(len(data_df)))
    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        effective_batch_size = len(indices)
        batch_data = [data_df.iloc[idx,:] for idx in indices]
        if is_multi:
            batch_labels = np.array([np.round(row['labels'][:2] / 9).astype(np.int) for row in batch_data])
        else:
            batch_labels = np.array([np.round(row['labels'][0:1]/9).astype(np.int) for row in batch_data])
        # batch_labels = np.array([np.array(row['labels'][0:1]).astype(np.float32) for row in batch_data])
        video_keys = [(x["user"], x["trial"]) for x in batch_data]
        video_batch_features = sample_video_df(video_data_df, video_keys)
        batch_features = np.array([np.reshape(row['data'], -1).astype(np.float32) for row in batch_data])
        # print(batch_labels)
        # print(batch_features)
        batch_features, batch_labels = torch.from_numpy(batch_features), torch.from_numpy(batch_labels)
        video_batch_features = torch.from_numpy(video_batch_features)
        batch_features = batch_features.view(effective_batch_size, 40, -1)
        assert batch_features.shape[0] == video_batch_features.shape[0]
        yield batch_features.to(device), video_batch_features.to(device), batch_labels.to(device)


def read_batches_multimodal(data_df, batch_size, shuffle, device):
    for batch_features, video_batch_features, batch_labels in read_batches(data_df, batch_size, shuffle, device):
        yield batch_features[:, :32, :].shape, batch_features[:, 32:, :].shape, video_batch_features, batch_labels

def plot_losses(model_name, train_loss, dev_loss, train_acc, dev_acc, epoch, test_acc=None, test_loss=None):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle('Bal accuracy/loss curves')
    ax1.plot(train_loss, label = "train_loss")
    ax1.plot(dev_loss, label = "dev_loss")
    ax1.set_xlabel("epoch num")
    ax1.set_ylabel("CE loss")
    if test_loss is not None:
        ax1.set_title("test loss=" + str(test_loss))
    else:
        ax1.set_title("Loss curve after epoch number " + str(epoch))
    ax1.legend()

    for c in train_acc[0]:
        train_acc_c = [float(x[c]) for x in train_acc]
        dev_acc_c = [float(x[c]) for x in dev_acc]
        ax2.plot(train_acc_c, label = "train_acc_"+str(c))
        ax2.plot(dev_acc_c, label = "dev_accs_"+str(c))
    ax2.set_xlabel("epoch num")
    ax2.set_ylabel("Balanced accuracy")
    if test_acc is not None:
        ax2.set_title("test acc=" + str(test_acc))
    else:
        ax2.set_title("Bal acc curve after epoch number " + str(epoch))

    ax2.legend()
    fig.savefig("./plots/"+model_name+".png")
    fig.clf()
    plt.close('all')

if __name__ == "__main__":
    print("main")
    output_df = read_video_data(video_features_path)
    print(len(output_df))
    data_df = read_phys_data(data_path)
    for batch_features, batch_labels in read_batches(data_df, output_df, 16, True, device):
        print(len(batch_features))


