import numpy as np
import pandas as pd
import csv
import fnmatch
import os
from scipy import signal

COLUMN_FORMAT = ['%.10f'] * 9

def mobiact(input_path, output_path):
    """MobiAct Datasets
    Vavoulas, G., Chatzaki, C., Malliotakis, T., Pediaditis, M. and Tsiknakis, M.,
    The MobiAct Dataset: Recognition of Activities of Daily Living using Smartphones.,
    In Proceedings of the International Conference on Information and
    Communication Technologies for Ageing Well and e-Health (ICT4AWE 2016), pages 143-151
    https://www.dropbox.com/s/sp8hrmrc2g2cy0u/MobiAct_Dataset.zip?dl=0
    """

    def _filter(activity, acc_path, gyro_path, ori_path, output_path):
        print('Opening datasets: [', acc_path, ', ', gyro_path, ']', sep='')
        raw_acc = np.loadtxt(acc_path, delimiter=',', skiprows=16)
        raw_gyr = np.loadtxt(gyro_path, delimiter=',', skiprows=16)
        raw_ori = np.loadtxt(ori_path, delimiter=',', skiprows=16)

        num_sample = _class_num_sample(activity)
        resampled_acc = signal.resample(raw_acc[:, 1:4], num_sample)
        resampled_gyro = signal.resample(raw_gyr[:, 1:4], num_sample)
        resampled_ori = signal.resample(raw_ori[:, 1:4], num_sample)
        activity_id = np.full((num_sample, 1), _class_id(activity))

        dataset = np.concatenate([resampled_acc, resampled_gyro, resampled_ori], 1)

        # header = str(num_sample) + ',6'
        np.savetxt(output_path, dataset, COLUMN_FORMAT, delimiter=',', comments='')
        print('Saved to:', output_path)

    def _class_id(activity_class):
        ret = -1
        if activity_class == 'STD':
            ret = 0
        elif activity_class == 'SCH':
            ret = 1
        elif activity_class == 'WAL':
            ret = 2
        elif activity_class == 'JOG':
            ret = 3
        elif activity_class == 'STU':
            ret = 4
        elif activity_class == 'STN':
            ret = 5

        return ret

    def _class_num_sample(activity_class):
        if activity_class == 'STD':
            ret = 5 * 60 * 50
        elif activity_class == 'SCH':
            ret = 6 * 50
        elif activity_class == 'WAL':
            ret = 5 * 60 * 50
        elif activity_class == 'JOG':
            ret = 30 * 50
        elif activity_class == 'STU':
            ret = 10 * 50
        elif activity_class == 'STN':
            ret = 10 * 50

        return ret

    def _filter_activity(activity, input_path, output_path):
        std_list = os.listdir(os.path.join(input_path, activity))

        std_acc_list = fnmatch.filter(std_list, activity + '_acc*')
        std_gyro_list = fnmatch.filter(std_list, activity + '_gyro*')
        std_ori_list = fnmatch.filter(std_list, activity + '_ori*')

        std_acc_list.sort()
        std_gyro_list.sort()
        std_ori_list.sort()

        activity_dir = os.path.join(output_path, 'mobiact', activity)
        os.makedirs(activity_dir, exist_ok=True)

        for i, _ in enumerate(std_acc_list):
            input_acc = os.path.join(input_path, activity, std_acc_list[i])
            input_gyro = os.path.join(input_path, activity, std_gyro_list[i])
            input_ori = os.path.join(input_path, activity, std_ori_list[i])
            output = os.path.join(activity_dir,
                                  std_acc_list[i][:4] + std_acc_list[i][8:-4] + '.csv')
            _filter(activity, input_acc, input_gyro, input_ori, output)

    _filter_activity('WAL', input_path, output_path)
    _filter_activity('JOG', input_path, output_path)
    _filter_activity('STU', input_path, output_path)
    _filter_activity('STN', input_path, output_path)


# input_path = '/home/dadafly/program/SensePrivacy/MobiAct/MobiAct_Dataset/'
# output_path = '/home/dadafly/program/SensePrivacy/MobiAct/Data/'

# mobiact(input_path = input_path, output_path = output_path)

def get_ds_infos(sub_info):
    ## 0:ID, 1:Weight, 2:Height, 3:Age, 4:Gender
    dss = np.genfromtxt(sub_info, delimiter=',')
    dss = dss[1:]
    print("----> Data subjects information is imported.")
    return dss


def creat_time_series(sub_info, data_dir, num_features, num_labels, label_codes, trial_codes):
    dataset_columns = num_features + sum(num_labels.values())
    ds_list = get_ds_infos(sub_info)
    train_data = np.zeros((0, dataset_columns))
    test_data = np.zeros((0, dataset_columns))
    for i, sub_id in enumerate(ds_list[:,0]):
        for j, act in enumerate(label_codes):
            for trial in trial_codes[act]:
                fname = data_dir + '/' + act + '/'+act+'_'+str(int(sub_id))+'_'+str(trial)+'.csv'
                raw_data = pd.read_csv(fname)
                unlabel_data = raw_data.values
                label_data = np.zeros((len(unlabel_data), dataset_columns))

                label_data[:,:-sum(num_labels.values())] = unlabel_data
                label_data[:,label_codes[act]] = 1 # act
                label_data[:,num_features + num_labels['activity'] + int(sub_id)-1] = 1 # id
                label_data[:,num_features + num_labels['activity'] + num_labels['id']-1 + num_labels['weight']] = int(ds_list[i,1]) # weight
                label_data[:,num_features + num_labels['activity'] + num_labels['id']-1 + num_labels['weight'] + num_labels['height']] = int(ds_list[i,2]) # height
                label_data[:,num_features + num_labels['activity'] + num_labels['id']-1 + num_labels['weight'] + num_labels['height'] + num_labels['age']] = int(ds_list[i,3]) # age
                label_data[:,-(num_labels['gender'])] = int(ds_list[i,4]) # gen
                ## We consider long trials as training dataset and short trials as test dataset
                if act == 'STN' or act == 'STU':
                    if trial > 4:
                        test_data = np.append(test_data, label_data, axis = 0)
                    else:
                        train_data = np.append(train_data, label_data, axis = 0)
                if act == 'JOG':
                    if trial > 2:
                        test_data = np.append(test_data, label_data, axis = 0)
                    else:
                        train_data = np.append(train_data, label_data, axis = 0)
                if act == 'WAL':
                    size = len(label_data)
                    test_data = np.append(test_data, label_data[:int(size/4)], axis = 0)
                    train_data = np.append(train_data, label_data[int(size/4):], axis = 0)

    return train_data , test_data

## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"


def time_series_to_section(dataset, num_labels, sliding_window_size, step_size_of_sliding_window,
                           standardize=False, num_features, **options):
    data = dataset[:, 0:-sum(num_labels.values())]

    act_labels = dataset[:, num_features : num_features + num_labels['activity']]
    id_labels  = dataset[:, num_features + num_labels['activity'] : num_features + num_labels['activity'] + num_labels['id']]
    wei_labels = dataset[:, num_features + num_labels['activity'] + num_labels['id'] : num_features + num_labels['activity'] + num_labels['id'] + num_labels['weight']]
    hei_labels = dataset[:, num_features + num_labels['activity'] + num_labels['id'] + num_labels['weight'] : num_features + num_labels['activity'] + num_labels['id'] + num_labels['weight'] + num_labels['height']]
    age_labels = dataset[:, num_features + num_labels['activity'] + num_labels['id'] + num_labels['weight'] + num_labels['height'] : num_features + num_labels['activity'] + num_labels['id'] + num_labels['weight'] + num_labels['height'] + num_labels['age']]
    gen_labels = dataset[:, -(num_labels['gender'])]

    mean = 0
    std = 1

    if standardize:
        ## Standardize each sensor’s data to have a zero mean and unity standard deviation.
        ## As usual, we normalize test dataset by training dataset's parameters
        if options:
            mean = options.get("mean")
            std = options.get("std")
            print("----> Test Data has been standardized")
        else:
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            print("----> Training Data has been standardized:\n the mean is = ", str(mean.mean()),
                  " ; and the std is = ", str(std.mean()))

        data -= mean
        data /= std
    else:
        print("----> Without Standardization.....")

    ## We want the Rows of matrices show each Feature and the Columns show time points.
    data = data.T

    size_features = data.shape[0]
    size_data = data.shape[1]
    number_of_secs = round(((size_data - sliding_window_size) / step_size_of_sliding_window))

    ##  Create a 3D matrix for Storing Snapshots
    secs_data = np.zeros((number_of_secs, size_features, sliding_window_size))
    act_secs_labels = np.zeros((number_of_secs, num_labels['activity']))
    gen_secs_labels = np.zeros(number_of_secs)
    hei_secs_labels = np.zeros(number_of_secs)
    wei_secs_labels = np.zeros(number_of_secs)
    age_secs_labels = np.zeros(number_of_secs)
    id_secs_labels  = np.zeros((number_of_secs, num_labels['id']))

    k = 0
    for i in range(0, (size_data) - sliding_window_size, step_size_of_sliding_window):
        j = i // step_size_of_sliding_window
        if (j >= number_of_secs):
            break
        if (gen_labels[i] != gen_labels[i + sliding_window_size - 1]):
            continue
        if (hei_labels[i] != hei_labels[i + sliding_window_size - 1]):
            continue
        if (wei_labels[i] != wei_labels[i + sliding_window_size - 1]):
            continue
        if (age_labels[i] != age_labels[i + sliding_window_size - 1]):
            continue
        if (not (act_labels[i] == act_labels[i + sliding_window_size - 1]).all()):
            continue
        if (not (id_labels[i] == id_labels[i + sliding_window_size - 1]).all()):
            continue
        secs_data[k] = data[0:size_features, i:i + sliding_window_size]
        act_secs_labels[k] = act_labels[i].astype(int)
        id_secs_labels[k]  = id_labels[i].astype(int)
        wei_secs_labels[k] = wei_labels[i].astype(int)
        hei_secs_labels[k] = hei_labels[i].astype(int)
        age_secs_labels[k] = age_labels[i].astype(int)
        gen_secs_labels[k] = gen_labels[i].astype(int)
        k = k + 1
    secs_data = secs_data[0:k]
    act_secs_labels = act_secs_labels[0:k]
    gen_secs_labels = gen_secs_labels[0:k]
    id_secs_labels  = id_secs_labels[0:k]
    wei_secs_labels = wei_secs_labels[0:k]
    hei_secs_labels = hei_secs_labels[0:k]
    age_secs_labels = age_secs_labels[0:k]

    return secs_data, act_secs_labels, gen_secs_labels, id_secs_labels, wei_secs_labels, hei_secs_labels, age_secs_labels, mean, std
