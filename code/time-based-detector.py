# %%
# Import necessary libraries and helper functions
from sklearn.metrics import confusion_matrix
import numpy as np
import scipy.stats
from sklearn.covariance import EllipticEnvelope
import pandas as pd
import os
from helper_functions import make_can_df, add_time_diff_per_aid_col, add_actual_attack_col, add_kde_val_col, add_gauss_val_col


# %%
def load_data(directory, exclude=[], file_condition=lambda file_name: True):
    print("Loading data from directory: " + directory)
    df_aggregation = []

    for file_name in os.listdir(directory):
        if file_condition(file_name) and not any(excl in file_name for excl in exclude):
            print("Loading file: " + file_name)
            df = make_can_df(os.path.join(directory, file_name))
            df = add_time_diff_per_aid_col(df)
            df_aggregation.append(df)

    return df_aggregation


def load_and_save_training_data(directory):
    if os.path.exists(directory + 'training_data.csv'):
        print("Training data already exists. Loading training data from: " +
              directory + 'training_data.csv')
        return pd.read_csv(directory + 'training_data.csv')

    df_aggregation = load_data(
        directory, file_condition=lambda file_name: "dyno" in file_name)
    # Concatenate all training datasets on the dyno
    df_training = pd.concat(df_aggregation)
    training_data = df_training[["time", "aid", "time_diffs"]]
    save_file = 'training_data.csv'
    print("Saving training data to: " + save_file)
    training_data.to_csv(directory + save_file)
    return training_data


# %%


def preprocess(df, aid):
    """
    Preprocesses the data by removing outliers.
    """
    time_diffs = df[df.aid == aid].time_diffs.values
    print("before: ", len(time_diffs))

    # identify outliers in the dataset
    # support_fraction=0.99
    ee = EllipticEnvelope(contamination=0.0001, support_fraction=0.999)
    inliers = ee.fit_predict(time_diffs.reshape(-1, 1))

    # select all rows that are not outliers
    mask = inliers != -1
    outliers = sum(mask == False)
    print("outliers: ", outliers, 100*outliers/len(time_diffs))

    time_diffs = time_diffs[mask]
    # summarize the shape of the updated training dataset
    print("after: ", len(time_diffs))

    return time_diffs


def calculate_statistics(time_diffs):
    """
    Returns a dictionary including the mean of its time_diffs, standard deviation of its time_diffs
    and KDE of its time_diffs
    """
    aid_dict = {'mu': time_diffs.mean(), 'std': time_diffs.std(), 'kde': scipy.stats.gaussian_kde(
        time_diffs), 'gauss': scipy.stats.norm(loc=time_diffs.mean(), scale=time_diffs.std())}
    return aid_dict


def calculate_statistics_for_each_aid(data):
    # Get a list of unique aids in the data
    unique_aids = data['aid'].unique()
    # Preprocess the data and calculate statistics for each unique aid
    stats = {aid: calculate_statistics(
        preprocess(data, aid)) for aid in unique_aids}
    data = add_kde_val_col(data, stats)
    data = add_gauss_val_col(data, stats)

    return data


# %%
def annotate_attack_data(attack_data, injection_intervals):
    """
    Annotates the attack data based on the injection intervals.
    """
    for index, row in injection_intervals.iterrows():
        aid = row['aid']
        payload = row['payload']
        intervals = [(row['start_time'], row['end_time'])]
        attack_data = add_actual_attack_col(
            attack_data, intervals, aid, payload)
    return attack_data


def load_and_annotate_attack_data(directory, metadata_file):
    # Load the attack data
    df_aggregation = load_data(
        directory, exclude=['masquerade', 'accelerator', 'metadata', metadata_file])

    # Load the injection intervals from the metadata file
    with open(os.path.join(directory, metadata_file), "r") as read_file:
        attack_dict = json.load(read_file)

    attack_metadata = []
    count = 0  # Initialize count here
    for file_name in os.listdir(directory):
        file_base = file_name[:-4]
        if file_base not in attack_dict:
            continue
        if "masquerade" in file_name or "accelerator" in file_name:
            continue

        metadata = attack_dict[file_base]
        if metadata["injection_id"] != "XXX":
            injection_id = int(metadata["injection_id"], 16)
        else:
            injection_id = "XXX"

        # From metadata file
        attack_metadata.append([tuple(metadata["injection_interval"])])

        # Add column to each attack dataframe to indicate attack (True) or non-attack (False) for each signal
        df_aggregation[count] = add_actual_attack_col(
            df_aggregation[count], attack_metadata[count], injection_id, metadata["injection_data_str"])

        count += 1  # Increment count here, inside the loop where you add items to your lists

    return df_aggregation


# %%


def get_results_binning(attack_list, D, n=6):
    """
    Simplified binning detection method that returns the results directly
    """
    results_binning = {}

    for i, attack in enumerate(attack_list):
        confusion_matrix_ = alert_by_bin(attack, D, n)
        precision = confusion_matrix_[
            1, 1] / (confusion_matrix_[1, 1] + confusion_matrix_[0, 1])
        recall = confusion_matrix_[
            1, 1] / (confusion_matrix_[1, 1] + confusion_matrix_[1, 0])
        false_positive = confusion_matrix_[
            0, 1] / (confusion_matrix_[0, 1] + confusion_matrix_[0, 0])

        results_binning[i+1] = {
            'cm': confusion_matrix_,
            'prec': precision,
            'recall': recall,
            'false_pos': false_positive
        }

    return results_binning


def alert_by_bin(df, D, n=6):
    """
    Checks for time windows of length mu*4 (where mu is average time_diff for aid) with 6 or more signals
    """
    cm = np.array([[0, 0], [0, 0]])

    for aid in df.aid.unique():
        df_test = df[df.aid == aid]
        df_test['predicted_attack'] = df_test.time_diffs.rolling(
            n).sum() <= D[aid]['mu']*4

        cm += confusion_matrix(df_test['actual_attack'],
                               df_test['predicted_attack'], labels=[0, 1])

    return cm

# %%


def detect_anomalies(models, attack_data, detection_method):
    if detection_method == 'Binning':
        return detect_anomalies_binning(models, attack_data)
    # elif detection_method == 'Gaussian':
    #     return detect_anomalies_gaussian(models, attack_data)
    else:
        raise ValueError(f"Unknown detection method: {detection_method}")


# %%

# Function to visualize the results
def visualize_results(results):
    # Add your visualization code here
    pass


# %%

training_data = load_and_save_training_data(
    '/Users/jamescourson/Documents/GAN_play/can-time-based-ids-benchmark/data/ambient/')

# %%
print(training_data.head())

# %%
training_data_stats = calculate_statistics_for_each_aid(training_data)

# %%
print(len(training_data_stats))

# %%
attack_data = load_and_annotate_attack_data(
    '../data/attacks', 'capture_metadata.json')
# results = detect_and_save_results('results', models, attack_data, 'mean') # Detect anomalies and save the results
# visualize_results(results)

# %%
print(attack_data[0].head())
# print length of attack datra that have actual attack
print(len(attack_data[0][attack_data[0].actual_attack == True]))
print(len(attack_data[0][attack_data[0].actual_attack == False]))

# now total number of actual attacks for all 16 attacks
print(len(attack_data[0][attack_data[0].actual_attack == True]) + len(attack_data[1][attack_data[1].actual_attack == True]) + len(attack_data[2][attack_data[2].actual_attack == True]) + len(attack_data[3][attack_data[3].actual_attack == True]) + len(attack_data[4][attack_data[4].actual_attack == True]) + len(attack_data[5][attack_data[5].actual_attack == True]) + len(attack_data[6][attack_data[6].actual_attack == True]) + len(attack_data[7][attack_data[7].actual_attack == True]) +
      len(attack_data[8][attack_data[8].actual_attack == True]) + len(attack_data[9][attack_data[9].actual_attack == True]) + len(attack_data[10][attack_data[10].actual_attack == True]) + len(attack_data[11][attack_data[11].actual_attack == True]) + len(attack_data[12][attack_data[12].actual_attack == True]) + len(attack_data[13][attack_data[13].actual_attack == True]) + len(attack_data[14][attack_data[14].actual_attack == True]) + len(attack_data[15][attack_data[15].actual_attack == True]))


# %%
# Detect anomalies and save the results
results = detect_and_save_results(
    'results', training_data_stats, attack_data, 'Binning')
# visualize_results(results)
