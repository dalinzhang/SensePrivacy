import numpy as np
import pandas as pd
import load_mobiact
import load_motionsense

print("--> Start...")
print("--> Building Training and Test Datasets...")

#############################################################################################################
# Load MobiAct dataset
#############################################################################################################
sub_info = "/home/dadafly/program/SensePrivacy/data/mobiact/data_subjects_info.csv"
data_dir = '/home/dadafly/program/SensePrivacy/data/mobiact/'
save_dir = '/home/dadafly/program/SensePrivacy/mobiact_model/'

num_features = 9
num_labels = {'activity': 4,
			  'gender':	  1,
			  'weight':   1, 
			  'height':   1,
			  'id':      44, 
			  'age':      1}



label_codes = {"STN":num_features, "STU":num_features+1, "WAL":num_features+2, "JOG":num_features+3}
trial_codes = {"STN":[1, 2, 3, 4, 5, 6], "STU":[1, 2, 3, 4, 5, 6], "WAL":[1], "JOG":[1, 2, 3]}
#________________________________

train_ts, test_ts = load_mobiact.creat_time_series(sub_info, data_dir, num_features, num_labels, label_codes, trial_codes)

#############################################################################################################
# Load MotionSense dataset
#############################################################################################################
# sub_info = "/home/dadafly/program/SensePrivacy/data/motionsense/data_subjects_info.csv"
# data_dir = '/home/dadafly/program/SensePrivacy/data/motionsense/'
# save_dir = '/home/dadafly/program/SensePrivacy/motionsense_model/'
# 
# num_features = 12 # attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
# num_labels = {'activity': 4,
# 			  'gender':	  1,
# 			  'weight':   1, 
# 			  'height':   1,
# 			  'id':      24, 
# 			  'age':      1}
# 
# 
# 
# label_codes = {"dws":num_features, "ups":num_features+1, "wlk":num_features+2, "jog":num_features+3}
# trial_codes = {"dws":[1,2,11], "ups":[3,4,12], "wlk":[7,8,15], "jog":[9,16]}
# 
# train_ts, test_ts = load_motionsense.creat_time_series(sub_info, data_dir, num_features, num_labels, label_codes, trial_codes)
# 
#________________________________

print("--> Shape of Training Time-Seires:", train_ts.shape)
print("--> Shape of Test Time-Series:", test_ts.shape)
#############################################################################################################
# process loaded data
#############################################################################################################

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
train_std = load_mobiact.time_series_to_section(train_ts.copy(),
                                   num_labels,
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
test_mean, test_std = load_mobiact.time_series_to_section(test_ts.copy(),
                                             num_labels,
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

###########################################################################################
# build network
###########################################################################################


from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.utils import np_utils

metrics = ['acc']
re_metrics = ['mse', 'mae']
## Activity Recognition
act_last_layer_dim = num_act_labels
act_loss_func = "categorical_crossentropy"
act_activation_func = 'softmax'
## Gender Classification
gen_last_layer_dim = num_gen_labels
gen_loss_func = "binary_crossentropy"
gen_activation_func = 'sigmoid'
## id Classification
id_last_layer_dim = num_id_labels
id_loss_func = "categorical_crossentropy"
id_activation_func = 'softmax'
## height regression
hei_last_layer_dim = num_wei_labels
hei_loss_func = "mean_squared_error"
hei_activation_func = 'linear'
## weight regression
wei_last_layer_dim = num_wei_labels
wei_loss_func = "mean_squared_error"
wei_activation_func = 'linear'
## age regression
age_last_layer_dim = num_wei_labels
age_loss_func = "mean_squared_error"
age_activation_func = 'linear'
## Training Phase
batch_size = 64
num_of_epochs = 10
verbosity = 2
## MTCNN
kernel_size = 3

pool_size = 2

conv_depth_1 = 16
conv_depth_2 = 32

drop_prob = 0.5

hidden_size = 400

from keras import optimizers
adam = optimizers.adam(lr=0.001)
'''
lossnet
'''
## Note that: because each section of time-series is a matrix, we use Convolution2D.
## On the other side: because each row of the matrix correspond to one feature of
##   time-series, so we use a (1,k) kernel to convolve the data points of each row with
##   just that row's data points

# inp = Input(shape=(height, width,1))
# conv1_1 = Convolution2D(conv_depth_1, (1 , kernel_size), padding='same', activation='relu')(inp)
# conv1_2 = Convolution2D(conv_depth_1, (1 , kernel_size), padding='same', activation='relu')(conv1_1)
# 
# pool_1 = MaxPooling2D(pool_size=(1, pool_size))(conv1_2)
# drop_1 = Dropout(drop_prob)(pool_1)
# 
# conv2_1 = Convolution2D(conv_depth_2, (1 , kernel_size), padding='same', activation='relu')(drop_1)
# conv2_2 = Convolution2D(conv_depth_2, (1 , kernel_size), padding='same', activation='relu')(conv2_1)
# pool_2 = MaxPooling2D(pool_size=(1, pool_size))(conv2_2)
# drop_2 = Dropout(drop_prob)(pool_2)
# 
# flat = Flatten()(drop_2)
# hidden = Dense(hidden_size, activation='relu')(flat)
# drop_3 = Dropout(drop_prob)(hidden)
# 
# out1 = Dense(act_last_layer_dim, activation= act_activation_func, name = "ACT")(drop_3)
# 
# lossnet_model = Model(inputs=inp, outputs=[out1])
# 
# lossnet_model.compile(loss=[act_loss_func],
#           optimizer=adam,
#           metrics=metrics)
# 
# for i in range(10):
# 	print(i+1,"/",10)
# 	history = lossnet_model.fit(train_data, [act_train_labels],
# 	              batch_size = batch_size,
# 	              epochs = 1,
# 	              verbose = verbosity)
# 
# 
# 	print("--> Evaluation on Test Dataset:")
# 	results_1 = lossnet_model.evaluate(test_data, [act_test_labels],
# 	                                 verbose = verbosity)
# 	print("**** Accuracy for Activity Recognition task is: ", results_1[1])
# 
# lossnet_model.save(save_dir + 'lossnet.h5')

'''
gender_net
'''
inp = Input(shape=(height, width, 1))
conv1_1 = Convolution2D(conv_depth_1, (1 , kernel_size), padding='same', activation='relu')(inp)
conv1_2 = Convolution2D(conv_depth_1, (1 , kernel_size), padding='same', activation='relu')(conv1_1)

pool_1 = MaxPooling2D(pool_size=(1, pool_size))(conv1_2)
drop_1 = Dropout(drop_prob)(pool_1)

conv2_1 = Convolution2D(conv_depth_2, (1 , kernel_size), padding='same', activation='relu')(drop_1)
conv2_2 = Convolution2D(conv_depth_2, (1 , kernel_size), padding='same', activation='relu')(conv2_1)
pool_2 = MaxPooling2D(pool_size=(1, pool_size))(conv2_2)
drop_2 = Dropout(drop_prob)(pool_2)

flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob)(hidden)

out1 = Dense(gen_last_layer_dim, activation= gen_activation_func, name = "ACT")(drop_3)

gender_model = Model(inputs=inp, outputs=[out1])

gender_model.compile(loss=[gen_loss_func],
          optimizer=adam,
          metrics=metrics)

for i in range(5):
	print(i+1,"/",5)
	history = gender_model.fit(train_data, [gen_train_labels],
	              batch_size = batch_size,
	              epochs = 1,
	              verbose = verbosity)


	print("--> Evaluation on Test Dataset:")
	results_1 = gender_model.evaluate(test_data, [gen_test_labels],
	                                 verbose = verbosity)
	print("**** Accuracy for gender Recognition task is: ", results_1[1])

gender_model.save(save_dir + 'gender_net.h5')


'''
id_net
'''
# inp = Input(shape=(height, width, 1))
# conv1_1 = Convolution2D(conv_depth_1, (1 , kernel_size), padding='same', activation='relu')(inp)
# conv1_2 = Convolution2D(conv_depth_1, (1 , kernel_size), padding='same', activation='relu')(conv1_1)
# 
# pool_1 = MaxPooling2D(pool_size=(1, pool_size))(conv1_2)
# drop_1 = Dropout(drop_prob)(pool_1)
# 
# conv2_1 = Convolution2D(conv_depth_2, (1 , kernel_size), padding='same', activation='relu')(drop_1)
# conv2_2 = Convolution2D(conv_depth_2, (1 , kernel_size), padding='same', activation='relu')(conv2_1)
# pool_2 = MaxPooling2D(pool_size=(1, pool_size))(conv2_2)
# drop_2 = Dropout(drop_prob)(pool_2)
# 
# flat = Flatten()(drop_2)
# hidden = Dense(hidden_size, activation='relu')(flat)
# drop_3 = Dropout(drop_prob)(hidden)
# 
# out1 = Dense(id_last_layer_dim, activation= id_activation_func, name = "ACT")(drop_3)
# 
# id_model = Model(inputs=inp, outputs=[out1])
# 
# id_model.compile(loss=[id_loss_func],
#           optimizer=adam,
#           metrics=metrics)
# 
# for i in range(25):
# 	print(i+1,"/",50)
# 	history = id_model.fit(train_data, [id_train_labels],
# 	              batch_size = batch_size,
# 	              epochs = 1,
# 	              verbose = verbosity)
# 
# 	print("--> Evaluation on Test Dataset:")
# 	results_1 = id_model.evaluate(test_data, [id_test_labels],
# 	                                 verbose = verbosity)
# 	print("**** Accuracy for id Recognition task is: ", results_1[1])
# 
# id_model.save(save_dir + 'id_net.h5')
# 
# 
# '''
# weight_net
# '''
# inp = Input(shape=(height, width, 1))
# conv1_1 = Convolution2D(conv_depth_1, (1 , kernel_size), padding='same', activation='relu')(inp)
# conv1_2 = Convolution2D(conv_depth_1, (1 , kernel_size), padding='same', activation='relu')(conv1_1)
# 
# pool_1 = MaxPooling2D(pool_size=(1, pool_size))(conv1_2)
# drop_1 = Dropout(drop_prob)(pool_1)
# 
# conv2_1 = Convolution2D(conv_depth_2, (1 , kernel_size), padding='same', activation='relu')(drop_1)
# conv2_2 = Convolution2D(conv_depth_2, (1 , kernel_size), padding='same', activation='relu')(conv2_1)
# pool_2 = MaxPooling2D(pool_size=(1, pool_size))(conv2_2)
# drop_2 = Dropout(drop_prob)(pool_2)
# 
# flat = Flatten()(drop_2)
# hidden = Dense(hidden_size, activation='relu')(flat)
# drop_3 = Dropout(drop_prob)(hidden)
# 
# out1 = Dense(wei_last_layer_dim, activation= wei_activation_func, name = "ACT")(drop_3)
# 
# wei_model = Model(inputs=inp, outputs=[out1])
# 
# wei_model.compile(loss=[wei_loss_func],
#           optimizer='adam',
#           metrics=re_metrics)
# 
# for i in range(18):
# 	print(i+1,"/",18)
# 	history = wei_model.fit(train_data, [wei_train_labels],
# 	              batch_size = batch_size,
# 	              epochs = 1,
# 	              verbose = verbosity)
# 
# 
# 	print("--> Evaluation on Test Dataset:")
# 	results_1 = wei_model.evaluate(test_data, [wei_test_labels],
# 	                                 verbose = verbosity)
# 	print("**** mse for weight Recognition task is: ", results_1[1])
# 	print("**** mae for weight Recognition task is: ", results_1[2])
# 
# wei_model.save(save_dir + 'wei_net.h5')


'''
height_net
'''
# inp = Input(shape=(height, width, 1))
# conv1_1 = Convolution2D(conv_depth_1, (1 , kernel_size), padding='same', activation='relu')(inp)
# conv1_2 = Convolution2D(conv_depth_1, (1 , kernel_size), padding='same', activation='relu')(conv1_1)
# 
# pool_1 = MaxPooling2D(pool_size=(1, pool_size))(conv1_2)
# drop_1 = Dropout(drop_prob)(pool_1)
# 
# conv2_1 = Convolution2D(conv_depth_2, (1 , kernel_size), padding='same', activation='relu')(drop_1)
# conv2_2 = Convolution2D(conv_depth_2, (1 , kernel_size), padding='same', activation='relu')(conv2_1)
# pool_2 = MaxPooling2D(pool_size=(1, pool_size))(conv2_2)
# drop_2 = Dropout(drop_prob)(pool_2)
# 
# flat = Flatten()(drop_2)
# hidden = Dense(hidden_size, activation='relu')(flat)
# drop_3 = Dropout(drop_prob)(hidden)
# 
# out1 = Dense(hei_last_layer_dim, activation= hei_activation_func, name = "ACT")(drop_3)
# 
# hei_model = Model(inputs=inp, outputs=[out1])
# 
# hei_model.compile(loss=[hei_loss_func],
#           optimizer=adam,
#           metrics=re_metrics)
# 
# for i in range(10):
# 	print(i+1,"/",10)
# 	history = hei_model.fit(train_data, [hei_train_labels],
# 	              batch_size = batch_size,
# 	              epochs = 1,
# 	              verbose = verbosity)
# 
# 
# 	print("--> Evaluation on Test Dataset:")
# 	results_1 = hei_model.evaluate(test_data, [hei_test_labels],
# 	                                 verbose = verbosity)
# 	print("**** mse for height Recognition task is: ", results_1[1])
# 	print("**** mae for height Recognition task is: ", results_1[2])
# 
# print("--> Evaluation on Test Dataset:")
# print("**** mse for height Recognition task is: ", results_1[1])
# print("**** mae for height Recognition task is: ", results_1[2])
# 
# hei_model.save(save_dir + 'hei_net.h5')


'''
age_net
'''
# inp = Input(shape=(height, width, 1))
# conv1_1 = Convolution2D(conv_depth_1, (1 , kernel_size), padding='same', activation='relu')(inp)
# conv1_2 = Convolution2D(conv_depth_1, (1 , kernel_size), padding='same', activation='relu')(conv1_1)
# 
# pool_1 = MaxPooling2D(pool_size=(1, pool_size))(conv1_2)
# drop_1 = Dropout(drop_prob)(pool_1)
# 
# conv2_1 = Convolution2D(conv_depth_2, (1 , kernel_size), padding='same', activation='relu')(drop_1)
# conv2_2 = Convolution2D(conv_depth_2, (1 , kernel_size), padding='same', activation='relu')(conv2_1)
# pool_2 = MaxPooling2D(pool_size=(1, pool_size))(conv2_2)
# drop_2 = Dropout(drop_prob)(pool_2)
# 
# flat = Flatten()(drop_2)
# hidden = Dense(hidden_size, activation='relu')(flat)
# drop_3 = Dropout(drop_prob)(hidden)
# 
# out1 = Dense(age_last_layer_dim, activation= age_activation_func, name = "ACT")(drop_3)
# 
# age_model = Model(inputs=inp, outputs=[out1])
# 
# age_model.compile(loss=[age_loss_func],
#           optimizer=adam,
#           metrics=re_metrics)
# 
# for i in range(15):
# 	print(i+1,"/",15)
# 	history = age_model.fit(train_data, [age_train_labels],
# 	              			batch_size = batch_size,
# 	              			epochs = 1,
# 	              			verbose = verbosity)
# 
# 
# 	print("--> Evaluation on Test Dataset:")
# 	results_1 = age_model.evaluate(test_data, [age_test_labels],
# 	                                 verbose = verbosity)
# 	print("**** mse for age Recognition task is: ", results_1[1])
# 	print("**** mae for age Recognition task is: ", results_1[2])
# 
# print("--> Evaluation on Test Dataset:")
# print("**** mse for age Recognition task is: ", results_1[1])
# print("**** mae for age Recognition task is: ", results_1[2])
# 
# age_model.save(save_dir + 'age_net.h5')

# preliminary study
# use feature of loss net to train an one layer MLP 


'''
from keras import backend as K
from keras.models import load_model

lossnet_model = load_model(save_dir + 'lossnet.h5')

lossnet_model.trainable = False

from keras import backend as K
inp = lossnet_model.input
outputs = [layer.output for layer in lossnet_model.layers]
functors = [K.function([inp], [out]) for out in outputs]

test_layer_outs = [func([test_data]) for func in functors]
train_layer_outs = [func([train_data]) for func in functors]

# layer_outs[0][0] input layer
# layer_outs[1][0] conv1_1
# layer_outs[2][0] conv1_2
# layer_outs[3][0] pool_1
# layer_outs[4][0] dropout_1
# layer_outs[5][0] conv2_1
# layer_outs[6][0] conv2_2
# layer_outs[7][0] pool_2
# layer_outs[8][0] dropout_2
# layer_outs[9][0] flat
# layer_outs[10][0] dense
# layer_outs[11][0] dropout_3
# layer_outs[12][0] output layer


X_test = test_layer_outs[1][0].reshape(test_data.shape[0], -1)
X_train = train_layer_outs[1][0].reshape(train_data.shape[0], -1)


assert X_train.shape[1] == X_test.shape[1]

inp_size = X_train.shape[1]

inp_pre = Input(shape=(inp_size,))
output_pre = Dense(num_act_labels, activation=act_activation_func)(inp_pre)

pre_model = Model(inputs=inp_pre, outputs=[output_pre])

pre_model.compile(loss = [act_loss_func], optimizer=adam, metrics = metrics)

history_pre = pre_model.fit(X_train, [act_train_labels], 
							batch_size=batch_size,
							epochs = num_of_epochs,
							verbose = verbosity)

results_pre = pre_model.evaluate(X_test, [act_test_labels], verbose=verbosity)

print("**** Accuracy for Activity Recognition task of conv1_1 is: ", results_pre[1])
'''


