import pandas as pd
import os
from helper_functions import make_can_df, add_time_diff_per_aid_col, add_actual_attack_col, add_kde_val_col, add_gauss_val_col, get_results_binning, unpickle
import json
import tqdm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import confusion_matrix

import scipy.stats
# from IPython.display import display



class Logger:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def log(self, message):
        if self.verbose:
            print(message)

class CanIDS():
    '''

    '''

    def __init__(self, path_to_training_data, LiveAttackDetectorClass=None, offline_detection_method=None, path_to_attack_data=None, meta_data_filename=None, verbose=False):
        self.logger = Logger(verbose)
        
        self.logger.log("loading and saving normal traffic:")
        self.training_data = self._load_and_save_training_data(path_to_training_data)
        self.logger.log(self.training_data)

        self.logger.log('"training" model: calculating stats:')
        self.training_data_stats = self._calculate_statistics_for_each_aid(self.training_data)
        self.logger.log(self.training_data_stats)

        if LiveAttackDetectorClass:
            # Maybe a check here to determine if its a valid Attack Detector
            self.AttackDetector = LiveAttackDetectorClass(self.training_data_stats)

        if offline_detection_method:
            # ensure path_to_attack_data and meta_data_filename specified
            if not (path_to_attack_data or meta_data_filename):
                raise BaseException("Ensure 'path_to_attack_data' and 'meta_data_filename' are specified")

            self.logger.log("loading and annotating attack data:")
            self.attack_data = self._load_and_annotate_attack_data(path_to_attack_data, meta_data_filename)
            self.logger.log(self.attack_data)

            self.logger.log("loading and annotating attack data:")
            self.detection_results = offline_detection_method(self.attack_data, self.training_data_stats)
            self.logger.log(self.detection_results)

    def get_detection_results(self):
        if not self.detection_results:
            raise BaseException("self.detection_results not defined")
        return self.detection_results

    def process_frame(self, frame):
        if not self.AttackDetector:
            raise BaseException("self.AttackDetector not defined")
        return self.AttackDetector.detect_frame(frame)


    def _load_data(self, directory, exclude=[], file_condition=lambda file_name: True):
        self.logger.log("Loading data from directory: " + directory)
        df_aggregation = []

        for file_name in os.listdir(directory):
            if file_condition(file_name) and not any(excl in file_name for excl in exclude):
                self.logger.log("Loading file: " + file_name)
                df = make_can_df(os.path.join(directory, file_name))
                df = add_time_diff_per_aid_col(df)
                df_aggregation.append(df)
                
        return df_aggregation


    def _load_and_save_training_data(self, directory):
        if os.path.exists(directory + 'training_data.csv'):
            self.logger.log("Training data already exists. Loading training data from: " + directory + 'training_data.csv')
            return pd.read_csv(directory + 'training_data.csv')
            
        df_aggregation = self._load_data(directory, file_condition=lambda file_name: "dyno" in file_name)
        # Concatenate all training datasets on the dyno
        df_training = pd.concat(df_aggregation)
        training_data = df_training[["time", "aid", "time_diffs"]]
        save_file = 'training_data.csv'
        self.logger.log("Saving training data to: " + save_file)
        training_data.to_csv(directory + save_file)
        
        return training_data

    def _preprocess(self, df, aid):
        """
        preprocesses the data by removing outliers.
        """
        time_diffs = df[df.aid==aid].time_diffs.values
        self.logger.log("before: " + str(len(time_diffs)))

        # identify outliers in the dataset
        ee = EllipticEnvelope(contamination=0.0001, support_fraction=0.999) # support_fraction=0.99
        inliers = ee.fit_predict(time_diffs.reshape(-1, 1))

        # select all rows that are not outliers
        mask = inliers != -1
        outliers = sum(mask == False)
        self.logger.log("outliers: " + str(outliers) + str(100*outliers/len(time_diffs)))

        time_diffs = time_diffs[mask]
        # summarize the shape of the updated training dataset
        self.logger.log("after: " + str(len(time_diffs)))

        return time_diffs


    def _calculate_statistics(self, time_diffs):
        """
        Returns a dictionary including the mean of its time_diffs, standard deviation of its time_diffs
        and KDE of its time_diffs
        """
        aid_dict = {'mu': time_diffs.mean(), 'std': time_diffs.std(), 'kde': scipy.stats.gaussian_kde(time_diffs), 'gauss': scipy.stats.norm(loc = time_diffs.mean(), scale = time_diffs.std())}
        aid_dict["y_thresholds_kde"] = {}
        aid_dict["y_thresholds_gauss"] = {}
        
        return aid_dict


    def _calculate_statistics_for_each_aid(self, data):
        # Get a list of unique aids in the data
        unique_aids = data['aid'].unique()
        # _preprocess the data and calculate statistics for each unique aid
        stats = {aid: self._calculate_statistics(self._preprocess(data, aid)) for aid in unique_aids}
        # data = [add_kde_val_col(data[i], stats) for i in range(len(data))]
        # data = [add_gauss_val_col(data[i], stats) for i in range(len(data))]
        
        return stats

    def _annotate_attack_data(self, attack_data, injection_intervals):
        """
        Annotates the attack data based on the injection intervals.
        """
        for index, row in injection_intervals.iterrows():
            aid = row['aid']
            payload = row['payload']
            intervals = [(row['start_time'], row['end_time'])]
            attack_data = add_actual_attack_col(attack_data, intervals, aid, payload)
        return attack_data

    def _load_and_annotate_attack_data(self, directory, metadata_file):
        # Load the attack data
        df_aggregation = self._load_data(directory, exclude=['masquerade', 'accelerator', 'metadata', metadata_file])

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
            df_aggregation[count] = add_actual_attack_col(df_aggregation[count], attack_metadata[count], injection_id, metadata["injection_data_str"])

            count += 1  # Increment count here, inside the loop where you add items to your lists

        return df_aggregation


## Offline Method
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def get_results_binning(attack_list, D, n=6):
    """
    Simplified binning detection method that returns the results directly
    """
    results_binning = {}

    for i, attack in enumerate(attack_list):
        confusion_matrix_ = alert_by_bin(attack, D, n)
        precision = confusion_matrix_[1,1] / (confusion_matrix_[1,1] + confusion_matrix_[0,1])
        recall = confusion_matrix_[1,1] / (confusion_matrix_[1,1] + confusion_matrix_[1,0])
        false_positive = confusion_matrix_[0,1] / (confusion_matrix_[0,1] + confusion_matrix_[0,0])

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
    cm = np.array([[0,0], [0,0]])
    
    for aid in df.aid.unique():
        df_test = df[df.aid == aid]
        df_test['predicted_attack'] = df_test.time_diffs.rolling(n).sum() <= D[aid]['mu']*4

        cm += confusion_matrix(df_test['actual_attack'], df_test['predicted_attack'], labels = [0,1])
    
    return cm


'''
    -1: not enough frames
    0: not attack
    1: attack detected
'''
class BinningAttackDetector:
    """
    This class implements a binning strategy for anomaly detection, where each bin corresponds 
    to a time window of length mu*4. If a bin contains n or more messages, it's considered 
    anomalous and marked as a potential attack.
    """
    def __init__(self, d, n=6):
        self.d = d
        self.n = n
        self.frames = {}

    def process_frame(self, frame):
        aid = frame['aid']
        time_diff = frame['time_diffs']

        # Initialize memory for this aid if it doesn't exist
        if aid not in self.frames:
            self.frames[aid] = []

        # Add this frame to memory
        self.frames[aid].append(time_diff)

        # If we don't have enough frames yet, return 'not enough frames'
        if len(self.frames[aid]) < self.n:
            return -1

        # If we have more than n frames, discard the oldest one
        if len(self.frames[aid]) > self.n:
            self.frames[aid].pop(0)

        # Check if the sum of time_diffs for the last n frames is less than or equal to mu*4
        if sum(self.frames[aid]) <= self.d[aid]['mu']*4:
            return 1
        else:
            return 0


ids = CanIDS(path_to_training_data='/home/jbc0071/Documents/can-time-based-ids-benchmark/data/ambient/', 
             LiveAttackDetectorClass = BinningAttackDetector,
             offline_detection_method=get_results_binning,
             path_to_attack_data='../data/attacks', 
             meta_data_filename='capture_metadata.json',
             verbose=True)
# print(ids.get_detection_results())

