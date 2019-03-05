import numpy as np
import pandas as pd
##_____________________________

#################################################################################
# load data 
#################################################################################

def get_ds_infos():
    ## 0:Code, 1:Weight, 2:Height, 3:Age, 4:Gender
    dss = np.genfromtxt("/home/dadafly/program/SensePrivacy/motion-sense/data/data_subjects_info.csv",delimiter=',')
    dss = dss[1:]
    print("----> Data subjects information is imported.")
    return dss
##____________

def creat_time_series(num_features, num_act_labels, num_gen_labels, label_codes, trial_codes, num_weight_labels=1, num_height_labels=1, num_id_labels=24, num_age_labels=1):
    dataset_columns = num_features+num_act_labels+num_gen_labels + num_height_labels + num_weight_labels + num_id_labels + num_age_labels
    ds_list = get_ds_infos()
    train_data = np.zeros((0,dataset_columns))
    test_data = np.zeros((0,dataset_columns))
    for i, sub_id in enumerate(ds_list[:,0]):
        for j, act in enumerate(label_codes):
            for trial in trial_codes[act]:
                fname = '/home/dadafly/program/SensePrivacy/motion-sense/data/A_DeviceMotion_data/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                raw_data = pd.read_csv(fname)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                unlabel_data = raw_data.values
                label_data = np.zeros((len(unlabel_data), dataset_columns))

                label_data[:,:-(num_act_labels + num_gen_labels + num_height_labels + num_weight_labels + num_id_labels + num_age_labels)] = unlabel_data
                label_data[:,label_codes[act]] = 1 # act
                label_data[:,num_features + num_act_labels + int(sub_id)-1] = 1 # id
                label_data[:,num_features + num_act_labels + num_id_labels -1 + num_weight_labels] = int(ds_list[i,1]) # weight
                label_data[:,num_features + num_act_labels + num_id_labels -1 + num_weight_labels + num_height_labels] = int(ds_list[i,2]) # height
                label_data[:,num_features + num_act_labels + num_id_labels -1 + num_weight_labels + num_height_labels + num_age_labels] = int(ds_list[i,3]) # age
                label_data[:,-(num_gen_labels)] = int(ds_list[i,4]) # gen
                ## We consider long trials as training dataset and short trials as test dataset
                if trial > 10:
                    test_data = np.append(test_data, label_data, axis = 0)
                else:
                    train_data = np.append(train_data, label_data, axis = 0)
    return train_data , test_data

#________________________________


print("--> Start...")

## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"
num_features = 12 # attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
num_act_labels = 4 # dws, ups, wlk, jog
num_gen_labels = 1 # 0/1(female/male)
num_id_labels = 24
num_wei_labels = 1
num_hei_labels = 1
num_age_labels = 1

label_codes = {"dws":num_features, "ups":num_features+1, "wlk":num_features+2, "jog":num_features+3}
trial_codes = {"dws":[1,2,11], "ups":[3,4,12], "wlk":[7,8,15], "jog":[9,16]}
## Calling 'creat_time_series()' to build time-series
print("--> Building Training and Test Datasets...")
train_ts, test_ts = creat_time_series(num_features, num_act_labels, num_gen_labels, label_codes, trial_codes)
print("--> Shape of Training Time-Seires:", train_ts.shape)
print("--> Shape of Test Time-Series:", test_ts.shape)


def time_series_to_section(dataset, num_act_labels, num_gen_labels, sliding_window_size, step_size_of_sliding_window,
                           standardize=False, num_features=12, num_weight_labels=1, num_height_labels=1, num_id_labels=24, num_age_labels=1, **options):
    data = dataset[:, 0:-(num_act_labels + num_gen_labels + num_height_labels + num_weight_labels + num_id_labels + num_age_labels)]

    act_labels = dataset[:, num_features : num_features + num_act_labels]
    id_labels  = dataset[:, num_features + num_act_labels : num_features + num_act_labels + num_id_labels]
    wei_labels = dataset[:, num_features + num_act_labels + num_id_labels : num_features + num_act_labels + num_id_labels + num_weight_labels]
    hei_labels = dataset[:, num_features + num_act_labels + num_id_labels + num_weight_labels : num_features + num_act_labels + num_id_labels + num_weight_labels + num_height_labels]
    age_labels = dataset[:, num_features + num_act_labels + num_id_labels + num_weight_labels + num_height_labels : num_features + num_act_labels + num_id_labels + num_weight_labels + num_height_labels + num_age_labels]
    gen_labels = dataset[:, -(num_gen_labels)]

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
    act_secs_labels = np.zeros((number_of_secs, num_act_labels))
    gen_secs_labels = np.zeros(number_of_secs)
    hei_secs_labels = np.zeros(number_of_secs)
    wei_secs_labels = np.zeros(number_of_secs)
    age_secs_labels = np.zeros(number_of_secs)
    id_secs_labels  = np.zeros((number_of_secs, num_id_labels))

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


##________________________________________________________________


## This Variable Defines the Size of Sliding Window
## ( e.g. 100 means in each snapshot we just consider 100 consecutive observations of each sensor)
sliding_window_size = 50  # 50 Equals to 1 second for MotionSense Dataset (it is on 50Hz samplig rate)
## Here We Choose Step Size for Building Diffrent Snapshots from Time-Series Data
## ( smaller step size will increase the amount of the instances and higher computational cost may be incurred )
step_size_of_sliding_window = 10

print("--> Sectioning Training and Test datasets: shape of each section will be: (", num_features, "x",
      sliding_window_size, ")")

train_data, \
act_train_labels, \
gen_train_labels, \
id_train_labels, \
wei_train_labels, \
hei_train_labels, \
age_train_labels, \
train_mean, \
train_std = time_series_to_section(train_ts.copy(),
                                   num_act_labels,
                                   num_gen_labels,
                                   sliding_window_size,
                                   step_size_of_sliding_window,
                                   standardize = True,
                                   num_features = num_features)




test_data, \
act_test_labels, \
gen_test_labels, \
id_test_labels, \
wei_test_labels, \
hei_test_labels, \
age_test_labels, \
test_mean, test_std = time_series_to_section(test_ts.copy(),
                                             num_act_labels,
                                             num_gen_labels,
                                             sliding_window_size,
                                             step_size_of_sliding_window,
                                             standardize = True,
                                             num_features = num_features,
                                             mean = train_mean,
                                             std = train_std)


print("--> Shape of Training Sections:", train_data.shape)
print("--> Shape of Test Sections:", test_data.shape)

train_data = np.expand_dims(train_data,axis=3)
test_data = np.expand_dims(test_data,axis=3)

num_train, height, width, channel = train_data.shape

#######################################################################################3
from keras import backend as K
from keras.models import load_model


transnet = load_model('./saved_model/style_weight_0/transnet_10.h5')
transnet.trainable = False

#######################################################################################3
test_trans = transnet.predict([test_data])

from scipy.io import savemat

savemat('./data/test_data.mat', {'test_data': test_data})
savemat('./data/test_trans.mat', {'test_trans': test_trans})


