import numpy as np
import pandas as pd
from loss import *
from load_data import *

print("--> Start...")
print("--> Building Training and Test Datasets...")

#############################################################################################################
# Load dataset
#############################################################################################################
sub_info = project_dir + "/Motionsense_dataset/data_subjects_info.csv"
data_dir = project_dir + '/Motionsense_dataset/A_DeviceMotion_data/'
save_dir = project_dir + '/Motionsense_model/'
num_features = 12 # attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
num_labels = {'activity':   4,
		'gender':   1,
		'weight':   1, 
		'height':   1,
		'id':      24, 
		'age':      1}



label_codes = {"dws":num_features, "ups":num_features+1, "wlk":num_features+2, "jog":num_features+3}
trial_codes = {"dws":[1,2,11], "ups":[3,4,12], "wlk":[7,8,15], "jog":[9,16]}
#________________________________

train_ts, test_ts = creat_time_series(sub_info, data_dir, num_features, num_labels, label_codes, trial_codes)

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
train_std = time_series_to_section(train_ts.copy(),
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
test_mean, test_std = time_series_to_section(test_ts.copy(),
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
#################################################################################
# generate noise data 
#################################################################################
np.random.seed(33)

noise_data = np.random.uniform(-20, 20, [1, train_data.shape[1], train_data.shape[2], 1])
#################################################################################
# load model
#################################################################################

from keras import backend as K
from keras.models import load_model

lossnet = load_model(project_dir+'/motionsense_model/lossnet.h5')
lossnet.trainable = False

lossnet_layers = dict([(layer.name, layer) for layer in lossnet.layers])
lossnet_out_func = dict([(layer.name, K.function([lossnet.input], [lossnet_layers[layer.name].output])) for layer in lossnet.layers])


gennet = load_model(project_dir+'/motionsense_model/gender_net.h5')
gennet.trainable = False
heinet = load_model(project_dir+'/motionsense_model/hei_net.h5')
heinet.trainable = False
weinet = load_model(project_dir+'/motionsense_model/wei_net.h5')
weinet.trainable = False
agenet = load_model(project_dir+'/motionsense_model/age_net.h5')
agenet.trainable = False
idnet = load_model(project_dir+'/motionsense_model/id_net.h5')
idnet.trainable = False
verbosity = 2

id_results  = idnet.evaluate(test_data, [id_test_labels], verbose = verbosity)
hei_results = heinet.evaluate(test_data, [hei_test_labels], verbose = verbosity)
wei_results = weinet.evaluate(test_data, [wei_test_labels], verbose = verbosity)
age_results = agenet.evaluate(test_data, [age_test_labels], verbose = verbosity)
gen_results = gennet.evaluate(test_data, [gen_test_labels], verbose = verbosity)

act_results = lossnet.evaluate(test_data, [act_test_labels], verbose = verbosity)

print("--> Evaluation on Raw Test Dataset:")
print("**** Accuracy for activity Recognition task is: ", act_results[1])

print("**** Accuracy for id Recognition task is: ", id_results[1])

print("**** Accuracy for gender Recognition task is: ", gen_results[1])

print("**** mse for weight Recognition task is: ", wei_results[1])
print("**** mae for weight Recognition task is: ", wei_results[2])

print("**** mse for height Recognition task is: ", hei_results[1])
print("**** mae for height Recognition task is: ", hei_results[2])

print("**** mse for age Recognition task is: ", age_results[1])
print("**** mae for age Recognition task is: ", age_results[2])
#################################################################################
# trans net
#################################################################################

content_weight = 0.35
activity_weight = 0.1
reconstruction_weight = 0.0
style_weight = 0.55

print("**************************************style weight is: ", style_weight)
print("**************************************content weight is: ", content_weight)
print("**************************************activity weight is: ", activity_weight)
print("**************************************reconstruction weight is: ", reconstruction_weight)

num_act_labels=num_labels['activity']

num_of_epochs = 10

kernel_size = 3
pool_size = 2

conv_depth_1 = 16
conv_depth_2 = 32

verbosity = 2
drop_prob = 0.5

hidden_size = 400
act_last_layer_dim=num_act_labels
act_activation_func = 'softmax'

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Reshape, merge, concatenate
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Conv2DTranspose
from keras.utils import np_utils
from keras.losses import categorical_crossentropy
from keras.utils.data_utils import get_file

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))

# define lossnet
def transnet_lossnet_concat(trans_in, trans_out):
	loss_in = Input(shape=(height, width, 1))
	x = concatenate([trans_out, loss_in], axis=0)

	conv1_1 = Convolution2D(conv_depth_1, (1 , kernel_size), padding='same', activation='relu')(x)
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
	
	out = Dense(act_last_layer_dim, activation= act_activation_func, name = "ACT")(drop_3)

	model = Model([trans_in, loss_in], out)

	# load weights
	for i, layer in enumerate(model.layers[-13:]):
		weights = lossnet.layers[i].get_weights()
		layer.set_weights(weights)

	for layer in model.layers[-13:]:
		layer.trainable = False
	
	return model



# build transnet
act_target = Input(shape=(num_act_labels,))
raw_inp = Input(shape=(height, width, 1))

x = Convolution2D(conv_depth_1, (1 , kernel_size), padding='same', activation='relu')(raw_inp)
x = MaxPooling2D(pool_size = (1 , pool_size))(x)

x = Convolution2D(conv_depth_2, (1 , kernel_size), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size = (1 , pool_size),padding='same')(x)

encoder = Convolution2D(conv_depth_1, (1 , kernel_size), padding='same', activation='relu')(x)

x = Convolution2D(conv_depth_2, (1 , kernel_size), padding='same', activation='relu')(x)

x = Conv2DTranspose(conv_depth_2, (1, kernel_size), strides=(1, 2), padding = 'same', activation='relu')(x)
x = Convolution2D(conv_depth_1, (1 , kernel_size), padding='same', activation='relu')(x)

x = Conv2DTranspose(conv_depth_1, (1, kernel_size), strides=(1, 2), padding = 'same', activation='relu')(x)
decoder = Convolution2D(1, (1 , kernel_size), padding='valid', activation='linear')(x)

transnet = Model(inputs = raw_inp, outputs = decoder)

print(transnet.summary())

# build combination net
combine_net = transnet_lossnet_concat(trans_in = transnet.input, trans_out = transnet.output)

# add reconstruction loss
# reconstruction_loss = reconstruction_weight * mean_squared_error(combine_net.input[0], transnet.output)
# combine_net.add_loss(reconstruction_loss)

comnet_layers = dict([(layer.name, layer) for layer in combine_net.layers])

# add content loss
content_layers = ['conv2d_9']
if content_weight !=0:
	for i, layer_name in enumerate(content_layers):
		layer = comnet_layers[layer_name]

		content_loss = FeatureReconstructionRegularizer(
			weight = content_weight)(layer)
		combine_net.add_loss(content_loss)


# add style loss
style_layers = [['conv2d_1','conv2d_7'], ['conv2d_3', 'conv2d_9']]
if style_weight !=0:
	for i, layer_name in enumerate(style_layers):
		layer = comnet_layers[layer_name[1]]
		style_target = lossnet_out_func[layer_name[0]]([noise_data])
		style_loss = StyleReconstructionRegularizer(
			style_feature_target = style_target[0],
			weight = style_weight)(layer)
		combine_net.add_loss(style_loss)


def custom_loss_function(y_true, y_pred):
	# add acctivity loss
	loss=activity_weight * categorical_crossentropy(y_true, y_pred[:1])
	return loss

combine_net.compile(
	optimizer = 'adam',
	loss=custom_loss_function
)


for i in range(num_of_epochs):
	print(i+1,'/',num_of_epochs)
	combine_net.fit(
		[train_data, train_data], act_train_labels,
		epochs = 1,
		batch_size = 1,
	    verbose=verbosity
	)

	#######################################################################################3
	test_trans = transnet.predict([test_data])

	id_results 	= idnet.evaluate(test_trans, [id_test_labels], verbose = verbosity)
	hei_results = heinet.evaluate(test_trans, [hei_test_labels], verbose = verbosity)
	wei_results = weinet.evaluate(test_trans, [wei_test_labels], verbose = verbosity)
	age_results = agenet.evaluate(test_trans, [age_test_labels], verbose = verbosity)
	gen_results = gennet.evaluate(test_trans, [gen_test_labels], verbose = verbosity)
	act_results = lossnet.evaluate(test_trans, [act_test_labels], verbose = verbosity)
	print("--> Evaluation on Trans Test Dataset:")
	print("**** Accuracy for activity Recognition task is: ", act_results[1])
	
	print("**** Accuracy for id Recognition task is: ", id_results[1])
	
	print("**** Accuracy for gender Recognition task is: ", gen_results[1])
	
	print("**** mse for weight Recognition task is: ", wei_results[1])
	print("**** mae for weight Recognition task is: ", wei_results[2])
	
	print("**** mse for height Recognition task is: ", hei_results[1])
	print("**** mae for height Recognition task is: ", hei_results[2])
	
	print("**** mse for age Recognition task is: ", age_results[1])
	print("**** mae for age Recognition task is: ", age_results[2])

	transnet.save(save_dir+'transnet_'+str(i+1)+'.h5')
	#######################################################################################3






