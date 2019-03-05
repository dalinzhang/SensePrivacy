import numpy as np
import pandas as pd

#################################################################################
# load data 
#################################################################################


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
                fname = data_dir + act + '_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                raw_data = pd.read_csv(fname)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
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
                if trial > 10:
                    test_data = np.append(test_data, label_data, axis = 0)
                else:
                    train_data = np.append(train_data, label_data, axis = 0)
    return train_data , test_data


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
