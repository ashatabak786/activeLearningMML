import pandas as pd
import os
import numpy as np
from constants import data_path, data_path_atlas, video_features_path_atlas
import pickle
from collections import Counter
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
video_features_path = "/Users/ashatabak/cmu/datasets/deap/video/"
video_features_path_chkpt = "/Users/ashatabak/cmu/datasets/deap/video/video_logreg.csv"
video_features_path_chkpt_atlas= "/work/aashfaq/datasets/deap/openFaceFeatures/processed/video_logred.csv.chk"


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


def path_distance(x_features, freq):
    x_features_diff = x_features.diff()[1:].pow(2).rolling(freq).sum()[freq-1:]
    x_diff_max = x_features_diff.max().rename(lambda x: x + "_diff_max")
    x_diff_mean = x_features_diff.mean().rename(lambda x: x + "_diff_mean")
    return pd.concat([x_diff_max, x_diff_mean])


def categorize_columns(features):
    auc_presence_features = [x for x in features.columns if "AU" in x and "_c" in x]
    auc_intensity_features = [x for x in features.columns if "AU" in x and "_r" in x]
    head_features = [x for x in features.columns if "pose_T" in x]
    gaze_features = [x for x in features.columns if "gaze_" in x and "angle" not in x]
    gaze_angle_features = [x for x in features.columns if "gaze_" in x and "angle" in x]
    eye_landmarks = [x for x in features.columns if "eye_lmk" in x]
    face_landmarks_features = [x for x in features.columns if ("x_" in x or "y_" in x) and "eye" not in x]
    face_landmarks_3dfeatures = [x for x in features.columns if ("X_" in x or "Y_" in x) and "eye" not in x]

    #mean features
    normalize_features = face_landmarks_features + face_landmarks_3dfeatures + eye_landmarks + gaze_features + head_features
    return normalize_features

def aucFeatures(features):
    auc_intensity_features = [x for x in features.columns if "AU" in x and "_r" in x]
    auc_intense = features[auc_intensity_features]
    auc_intense_mean = auc_intense.mean().rename(lambda x: x + "_mean")
    auc_intense_max = auc_intense.max().rename(lambda x: x + "_max")
    auc_intense_min = auc_intense.min().rename(lambda x: x + "_min")
    auc_intense_sd = auc_intense.std().rename(lambda x: x + "_std")

    auc_presence_features = [x for x in features.columns if "AU" in x and "_c" in x]
    auc_presence = features[auc_presence_features]
    auc_presence_sum = auc_presence.sum().rename(lambda x: x + "_sum")
    auc_presence_max = auc_presence.max().rename(lambda x: x + "_max")

    return pd.concat([auc_presence_sum, auc_presence_max, auc_intense_mean, auc_intense_max, auc_intense_min, auc_intense_sd])


def headFeatures(features):
    head_features = [x for x in features.columns if "pose_T" in x]
    head_feat = features[head_features] - features[head_features].mean()
    head_feat_max = head_feat.max().rename(lambda x: x + "_max")
    head_feat_min = head_feat.min().rename(lambda x: x + "_min")
    head_feat_std = head_feat.std().rename(lambda x: x + "_std")
    head_diff = path_distance(head_feat, 100)
    head_feats = pd.concat([head_feat_max, head_feat_std, head_feat_min, head_diff])
    return head_feats


def gaze_features(features):
    gaze_features = [x for x in features.columns if "gaze_" in x and "angle"  in x]
    gaze_angle_features = features[gaze_features] - features[gaze_features].mean()
    gaze_angle_max = gaze_angle_features.max().rename(lambda x: x + "_max")
    gaze_angle_min = gaze_angle_features.min().rename(lambda x: x + "_min")
    gaze_angle_std = gaze_angle_features.std().rename(lambda x: x + "_std")
    gaze_diff = path_distance(gaze_angle_features, 100)
    gaze_feats = pd.concat([gaze_angle_max, gaze_angle_min, gaze_angle_std, gaze_diff])
    return gaze_feats


def extract_features(features):
    return pd.concat([aucFeatures(features), gaze_features(features), headFeatures(features)])

def read_video_data_manual(video_features_path, is_debug=False, redo = False):
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
                features[normalize_cols] = features[normalize_cols] - features[normalize_cols].mean()
                output_features = extract_features(features)
                output_features.set_value("user", user)
                output_features.set_value("trial", int(trial[-2:]))
                output_features.set_value(" face_id", int(user.strip("s")))
                all_features_df.append(output_features)
    final_df = pd.concat(all_features_df, axis=1).transpose()
    print(len(final_df))
    final_df.to_csv(chk_path)
    return final_df

def evaluate(clf, X_train, X_test, y_train, y_test):
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    train_bal = balanced_accuracy_score(y_train, train_predict)
    test_bal = balanced_accuracy_score(y_test, test_predict)
    print("results", train_bal, test_bal)
    print(Counter(test_predict))
    return test_bal



if __name__ == "__main__":
    is_debug = False
    deap_path = data_path_atlas
    n_iter = 500
    vid_path = video_features_path_atlas
    if is_debug:
        n_iter = 10
        deap_path = data_path
        vid_path = video_features_path_atlas
    deap_df = read_phys_data(deap_path)
    video_df = read_video_data_manual(vid_path, is_debug=is_debug, redo=False)
    vid_features = list(video_df.columns)
    vid_features.remove("user")
    # vid_features.remove("trial")
    final_features = vid_features + ["label_1"]

    combined_features = deap_df.merge(video_df, on=["user", "trial"])
    combined_features["label_1"] = combined_features.apply(lambda row: np.round(row["labels"][0]/9),axis=1)
    users = list(deap_df["user"].unique())
    users = [x for x in users if x < "s23"]
    print(len(users))
    train_users, test_users = users[:int(0.8 * len(users))], users[int(0.8 * len(users)):]
    train_df = combined_features[combined_features.user.isin(train_users)][final_features]
    test_df = combined_features[combined_features.user.isin(test_users)][final_features]
    print("train, test users", len(train_users), len(test_users))
    X_train = train_df[vid_features]
    y_train = train_df["label_1"]
    X_test = test_df[vid_features]
    y_test = test_df["label_1"]
    clf = LogisticRegression(random_state=42, penalty='l1', C = 2.0).fit(X_train, y_train)
    print("default", evaluate(clf, X_train, X_test, y_train, y_test))
    random_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100, 1000000],
        'class_weight': ['balanced', {0:0.5,1:0.5}]
    }

    # rf = Ra(random_state=42)
    rf_clf = RandomForestClassifier(random_state=42)
    rf_grid = {'bootstrap': [True, False],
     'max_depth': [5, 10, 20, 30, 40, 50, 60, 100, None],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10],
     'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 2000]}


    clf_random = RandomizedSearchCV(estimator=rf_clf, param_distributions=rf_grid, n_iter=n_iter, cv=5, verbose=0,
                                   random_state=42, n_jobs=-1, scoring='balanced_accuracy')

    clf_random.fit(X_train, y_train)
    clf_best = clf_random.best_estimator_
    print("optimized", evaluate(clf_best, X_train, X_test, y_train, y_test))
    print("best_params", clf_random.best_params_, "val score:", clf_random.best_score_)
    feature_imp = clf_best.feature_importances_
    importance = sorted([(imp, feature) for feature, imp in zip(X_train.columns, feature_imp)], key = lambda x: -1*x[0])
    print("feature importnaces = ", [x for x in importance][:20])

    print()





