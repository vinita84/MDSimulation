import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.constraints import maxnorm
from sklearn.model_selection import StratifiedKFold
import random

# Python optimisation variables
learning_rate = 0.0001
epochs = 500
dropout_rate = 0.15

filename = 'jcs_paper_data.dat'

#data load
contents = open(filename).read().splitlines()
contents = contents[1:]
random.shuffle(contents)
input_params=[]
output_density=[]
pos_val=[]
error_bars = []
for line in range(1, len(contents)):
            line_arr = contents[line].split(",")
            input_params.append(list(map(float, line_arr[0:5])))
            pos_val.append(list(map(float, line_arr[5:155])))
            output_density.append(list(map(float, line_arr[155:305])))
            error_bars.append(list(map(float, line_arr[305:455])))
train_input,test_input=input_params[0:2432],input_params[2432:]
train_gt,test_gt=output_density[0:2432],output_density[2432:]

#Define the model
model = keras.Sequential([
    keras.layers.Dense(512, input_shape=(5,), kernel_initializer=keras.initializers.glorot_normal(seed=None), kernel_constraint=maxnorm(3), activation="relu"),
    keras.layers.Dropout(dropout_rate),
    keras.layers.Dense(256, kernel_initializer=keras.initializers.glorot_normal(seed=None), kernel_constraint=maxnorm(3), activation="relu"),
    keras.layers.Dropout(dropout_rate),
    keras.layers.Dense(150, kernel_initializer=keras.initializers.glorot_normal(seed=None),activation="linear")
])

#Stochastic GD
sgd = SGD(lr=0.01, momentum=0.9)

#Compile the model
model.compile(optimizer='ada', loss="mean_squared_error", metrics=['accuracy'])

#Stratified K-fold Validation
#skf = StratifiedKFold(n_splits=100, random_state=None, shuffle=False)
#for train_index, test_index in skf.split(np.array(input_params), np.array(output_density)):
#    print("TRAIN:", train_index, "TEST:", test_index)

#train the model
model.fit(train_input, train_gt, validation_data=(test_input, test_gt), batch_size=25, epochs = 5000)

#prediction
prediction = model.predict(test_input)

#Calculating peak densities
predicted_peaks = []
predicted_pos_peaks = []
gt_peaks = []
for i,ele in enumerate(prediction):
    ind = list(ele).index(max(ele))
    predicted_peaks.append(ele[ind])
    predicted_pos_peaks.append(pos_val[i][ind])
    gt_peaks.append(test_gt[i][ind])

#plot the graphs
plt.plot(predicted_pos_peaks, gt_peaks,"g^")
plt.plot(predicted_pos_peaks, predicted_peaks,"ro")

#Calculate Success Rate
""" x = []
for i,ele in enumerate(prediction):
    success = []
    for j in range(0, len(ele)):
        y=append(ele[j] - test_gt[i][j])
        success.append(1 if y<error_bars[i][j] else 0)
         """