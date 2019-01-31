#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 21:34:53 2018

@author: yifeiren
"""

#Original Dataset attributes:
#T-Drive data, Microsoft Beijing Sample
#id, date time, long, lat

import time
import pandas as pd
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Embedding, Dense, merge, SimpleRNN, Merge, Activation, LSTM, GRU, Dropout,Input,TimeDistributed
from keras import optimizers
import tensorflow as tf
from scipy.stats import rankdata
from keras.utils import to_categorical
import glob
import errno
from keras.models import model_from_json
from scipy.spatial import distance


path = '/Users/yifeiren/Desktop/RNNTest/data/01/*.txt'
files = glob.glob(path)
trajs = []
maxrecord = 0
TRAIN_TEST_PART = 0.95
GRID_COUNT = 32

place_dimension = GRID_COUNT*GRID_COUNT
time_dimension=576
pld=50
timek=50
hiddenneurons=50
learningrate=0.001
for name in files:
    try:
        with open(name) as f:
            taxi_trajs = pd.read_csv(name)
            taxi_trajs = taxi_trajs.values
            trajs.append(taxi_trajs)
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

trajsdata = np.concatenate( trajs, axis=0 )

#PATH = '/Users/yifeiren/Desktop/RNNTest/data/01/9754.txt'
#taxi_trajs = pd.read_csv(PATH)
#taxi_trajs = taxi_trajs.values



def time_second(ci_time, form = '%Y-%m-%d %X'):
    st = time.strptime(ci_time, form)
    #mounth = st.tm_mon
    weekday = st.tm_wday
    second = st.tm_min
    hour = st.tm_hour
    if weekday < 6:
        timestamp = hour*12 + int(second/12)
        return timestamp
    else:
        timestamp = 24*12 + hour*12 + int(second/12)
        return timestamp


second = np.zeros(shape=(len(trajsdata),1))
i=0
for row in trajsdata:
    currentSecond = time_second(row[1])
    second[i] = currentSecond
    i=i+1

taxi_data = np.append(trajsdata, second, axis=1)
#usefuldata = taxi_data[:,2:5]



newdata = []
initialtime = taxi_data[0][4]
for row in taxi_data:
    if row[4] != initialtime:
        if ((row[2]!= 0.0) and (row[3]!= 0.0)):
            newdata.append([row[0],row[2],row[3],row[4]])
            initialtime = row[4]

minValueX = 10e10
minValueY = 10e10
maxValueX = -10e10
maxValueY = -10e10 

for item in newdata:
    dimensionXY = ([float(item[1]), float(item[2])])
    if (dimensionXY[0] < minValueX):
        minValueX = dimensionXY[0]
    if (dimensionXY[1] < minValueY):
        minValueY = dimensionXY[1]
    if (dimensionXY[0] > maxValueX):
        maxValueX = dimensionXY[0]
    if (dimensionXY[1] > maxValueY):
        maxValueY = dimensionXY[1]
m_dOriginX = minValueX
m_dOriginY = minValueY
dSizeX = (maxValueX - minValueX) / GRID_COUNT
dSizeY = (maxValueY - minValueY) / GRID_COUNT

center_location_list = []
for i in range(0, GRID_COUNT * GRID_COUNT + 1):
    y_ind = int(i / GRID_COUNT)
    x_ind = i - y_ind * GRID_COUNT
    center_location_list.append((m_dOriginX + x_ind * dSizeX + 0.5 * dSizeX,  + m_dOriginY+ y_ind * dSizeY + 0.5 * dSizeY))
 
finaldata = []
for each in newdata:
    nXCol = int((each[1] - m_dOriginX) / dSizeX)
    nYCol = int((each[2] - m_dOriginY) / dSizeY)
    if nXCol >= GRID_COUNT :
        print ('max X')
        nXCol = GRID_COUNT  - 1
    if nYCol >= GRID_COUNT :
        print ('max Y')
        nYCol = GRID_COUNT  - 1
    iIndex = nYCol * GRID_COUNT  + nXCol
    finaldata.append([each[0],iIndex,each[3]])
    
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(finaldata)
 
users = {}
for row in finaldata:
    if row[0] in users:
        users[row[0]].append(row)
    else:
        users[row[0]] = []
        users[row[0]].append(row)

for user in users.keys():
    if len(users[user]) > maxrecord:
        maxrecord = len(users[user])

    

    
##train test seperate
def geo_dataset_train_test_text(user_feature_sequence, max_record, place_dim = GRID_COUNT*GRID_COUNT,
                              train_test_part=TRAIN_TEST_PART):
    user_index = {}
    i=0
    for user in user_feature_sequence.keys():
        user_index[user]=i
        i=i+1
    user_dim = len(user_feature_sequence.keys())

    all_train_X_pl, all_train_X_time , all_train_X_user, all_train_Y\
        = [],[],[],[]
    all_test_X_pl_1, all_test_X_time_1, all_test_X_user_1, all_test_evl_1 \
        = [],[],[],[]
    all_test_X_pl_2, all_test_X_time_2, all_test_X_user_2, all_test_evl_2\
        = [],[],[],[]
    all_test_X_pl_3, all_test_X_time_3, all_test_X_user_3, all_test_evl_3\
        = [],[],[],[]
    all_test_X_pl_4, all_test_X_time_4, all_test_X_user_4, all_test_evl_4\
        = [],[],[],[]
    all_test_X_pl_5, all_test_X_time_5, all_test_X_user_5, all_test_evl_5\
        = [],[],[],[]

    for user in user_feature_sequence.keys():
        train_pl = []
        train_time = []
        train_y = []
        train_user = []

        test_pl_1 = []
        test_time_1 = []
        test_y_1 = []
        test_user_1 = []
        
        test_pl_2 = []
        test_time_2 = []
        test_y_2 = []
        test_user_2 = []
        
        test_pl_3 = []
        test_time_3 = []
        test_y_3 = []
        test_user_3 = []
        
        test_pl_4 = []
        test_time_4 = []
        test_y_4 = []
        test_user_4 = []
        
        test_pl_5 = []
        test_time_5 = []
        test_y_5 = []
        test_user_5 = []
        
        sequ_features = user_feature_sequence[user]
        train_size = max_record-8
        for sample in range(0, len(sequ_features)-8):
            train_pl.append(sequ_features[sample][1])
            train_time.append(sequ_features[sample][2])
            train_user.append((user_index[user] + 1))
        train_y = train_pl[1:]
        while len(train_pl) < (train_size):
            train_pl.append(0)
            train_time.append(0)
            train_user.append(0)
        train_y.append(sequ_features[-8][1])
        while len(train_y) < (train_size):
            train_y.append(0)
        all_train_X_pl.append(np.array(train_pl))
        all_train_X_time.append(np.array(train_time))
        all_train_X_user.append(np.array(train_user))
        all_train_Y.append(to_categorical(train_y, num_classes=place_dim + 1))
        
        
        ##test 1
        for sample in range(7, 2, -1):
            test_pl_1.append(sequ_features[-sample][1])
            test_time_1.append(sequ_features[-sample][2])
            test_user_1.append((user_index[user] + 1))
        while len(test_pl_1) < (train_size):
            test_pl_1.append(0)
            test_time_1.append(0)
            test_user_1.append(0)
        test_y_1.append(sequ_features[-1][1])
        while len(test_y_1) < (train_size):
            test_y_1.append(0)
        all_test_X_pl_1.append(np.array(test_pl_1))
        all_test_X_time_1.append(np.array(test_time_1))
        all_test_X_user_1.append(np.array(test_user_1))
        #all_test_Y.append(to_categorical(test_y, num_classes=place_dim + 1))
        all_test_evl_1.append(test_y_1)
        


        ##test2
        for sample in range(6, 2, -1):
            test_pl_2.append(sequ_features[-sample][1])
            test_time_2.append(sequ_features[-sample][2])
            test_user_2.append((user_index[user] + 1))
        while len(test_pl_2) < (train_size):
            test_pl_2.append(0)
            test_time_2.append(0)
            test_user_2.append(0)
        test_y_2.append(sequ_features[-1][1])
        while len(test_y_2) < (train_size):
            test_y_2.append(0)
        all_test_X_pl_2.append(np.array(test_pl_2))
        all_test_X_time_2.append(np.array(test_time_2))
        all_test_X_user_2.append(np.array(test_user_2))
        #all_test_Y.append(to_categorical(test_y, num_classes=place_dim + 1))
        all_test_evl_2.append(test_y_2)
        
        ##test 3
        for sample in range(5, 2, -1):
            test_pl_3.append(sequ_features[-sample][1])
            test_time_3.append(sequ_features[-sample][2])
            test_user_3.append((user_index[user] + 1))
        while len(test_pl_3) < (train_size):
            test_pl_3.append(0)
            test_time_3.append(0)
            test_user_3.append(0)
        test_y_3.append(sequ_features[-1][1])
        while len(test_y_3) < (train_size):
            test_y_3.append(0)
        all_test_X_pl_3.append(np.array(test_pl_3))
        all_test_X_time_3.append(np.array(test_time_3))
        all_test_X_user_3.append(np.array(test_user_3))
        #all_test_Y.append(to_categorical(test_y, num_classes=place_dim + 1))
        all_test_evl_3.append(test_y_3)
        
        ##test 4
        for sample in range(4, 2, -1):
            test_pl_4.append(sequ_features[-sample][1])
            test_time_4.append(sequ_features[-sample][2])
            test_user_4.append((user_index[user] + 1))
        while len(test_pl_4) < (train_size):
            test_pl_4.append(0)
            test_time_4.append(0)
            test_user_4.append(0)
        test_y_4.append(sequ_features[-1][1])
        while len(test_y_4) < (train_size):
            test_y_4.append(0)
        all_test_X_pl_4.append(np.array(test_pl_4))
        all_test_X_time_4.append(np.array(test_time_4))
        all_test_X_user_4.append(np.array(test_user_4))
        #all_test_Y.append(to_categorical(test_y, num_classes=place_dim + 1))
        all_test_evl_4.append(test_y_4)

        
        ##test 5
        for sample in range(3, 2, -1):
            test_pl_5.append(sequ_features[-sample][1])
            test_time_5.append(sequ_features[-sample][2])
            test_user_5.append((user_index[user] + 1))
        while len(test_pl_5) < (train_size):
            test_pl_5.append(0)
            test_time_5.append(0)
            test_user_5.append(0)
        test_y_5.append(sequ_features[-1][1])
        while len(test_y_5) < (train_size):
            test_y_5.append(0)
        all_test_X_pl_5.append(np.array(test_pl_5))
        all_test_X_time_5.append(np.array(test_time_5))
        all_test_X_user_5.append(np.array(test_user_5))
        #all_test_Y.append(to_categorical(test_y, num_classes=place_dim + 1))
        all_test_evl_5.append(test_y_5)
        
    
    all_train_X_pl =  np.array(all_train_X_pl)
    all_train_X_time = np.array(all_train_X_time)
    all_train_X_user = np.array(all_train_X_user)
    # all_train_X_text = np.array(all_train_X_text)
    #all_train_Y =  np.array(all_train_Y)
    all_test_X_pl_1 = np.array(all_test_X_pl_1)
    all_test_X_time_1=  np.array(all_test_X_time_1)
    all_test_X_user_1 = np.array(all_test_X_user_1)
    
    all_test_X_pl_2 = np.array(all_test_X_pl_2)
    all_test_X_time_2=  np.array(all_test_X_time_2)
    all_test_X_user_2 = np.array(all_test_X_user_2)
    
    all_test_X_pl_3 = np.array(all_test_X_pl_3)
    all_test_X_time_3=  np.array(all_test_X_time_3)
    all_test_X_user_3 = np.array(all_test_X_user_3)
    
    all_test_X_pl_4 = np.array(all_test_X_pl_4)
    all_test_X_time_4=  np.array(all_test_X_time_4)
    all_test_X_user_4 = np.array(all_test_X_user_4)
    
    all_test_X_pl_5 = np.array(all_test_X_pl_5)
    all_test_X_time_5=  np.array(all_test_X_time_5)
    all_test_X_user_5 = np.array(all_test_X_user_5)

    return [all_train_X_pl,all_train_X_time,all_train_X_user],np.array(all_train_Y),\
           [all_test_X_pl_1, all_test_X_time_1,all_test_X_user_1], all_test_evl_1,\
           [all_test_X_pl_2, all_test_X_time_2,all_test_X_user_2], all_test_evl_2,\
           [all_test_X_pl_3, all_test_X_time_3,all_test_X_user_3], all_test_evl_3,\
           [all_test_X_pl_4, all_test_X_time_4,all_test_X_user_4], all_test_evl_4,\
           [all_test_X_pl_5, all_test_X_time_5,all_test_X_user_5], all_test_evl_5,\
           user_dim

train_X, train_Y, vali_X_1, vali_evl_1,vali_X_2, vali_evl_2,vali_X_3, vali_evl_3,\
vali_X_4, vali_evl_4, vali_X_5, vali_evl_5, user_dim = geo_dataset_train_test_text(users, maxrecord)           
           
           
def geo_lprnn_trainable_text_model(user_dim, max_record, place_dim=place_dimension, time_dim=time_dimension,
                            pl_d=pld, time_k=timek, hidden_neurons=hiddenneurons,
                                   learning_rate=learningrate):
    len = int(max_record-8)
    # RNN model construction
    pl_input = Input(shape=(len,), dtype='int32', name = 'pl_input')
    time_input = Input(shape=(len,), dtype='int32', name = 'time_input')
    user_input = Input(shape=(len,), dtype='int32', name='user_input')


    pl_embedding = Embedding(input_dim=place_dim + 1, output_dim=pl_d, name ='pl_embedding' ,
                              mask_zero=True)(pl_input)
    time_embedding = Embedding(input_dim=time_dim + 1, output_dim=time_k, name='time_embedding',
                               mask_zero=True)(time_input)
    user_embedding = Embedding(input_dim=user_dim + 1, output_dim=place_dim + 1, name='user_embedding',
                               mask_zero=True)(user_input)

    # text_embedding = Embedding(input_dim=word_vec.shape[0],output_dim= TEXT_K,
    #                           weights=[word_vec],name="text_embeddng")(text_input)
    #text_embedding = EmbeddingMatrix(TEXT_K, weights=[word_vec], name="text_embeddng", trainable=True)(text_input)

    attrs_latent = merge([pl_embedding,time_embedding],mode='concat')
    # time_dist = TimeDistributed(Dense(50))
    lstm_out = LSTM(hidden_neurons, return_sequences=True,name='lstm_layer0')(attrs_latent)
    # lstm_out = LSTM(hidden_neurons, return_sequences=True, name='lstm_layer1')(lstm_out)
    # lstm_out = LSTM(hidden_neurons, return_sequences=True, name='lstm_layer2')(lstm_out)
    dense = Dense(place_dim + 1, name='dense')(lstm_out)
    out_vec = merge([dense,user_embedding],mode='sum')
    pred = Activation('softmax')(out_vec)
    model = Model([pl_input,time_input,user_input], pred)

    # model.load_weights('./model/User_RNN_Seg_Epoch_0.3_rmsprop_300.h5')
    # Optimization
    #sgd = optimizers.SGD(lr=learning_rate)
    rmsprop = optimizers.RMSprop(lr=learning_rate)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy')
    model.summary()
    return model           
           
           
model = geo_lprnn_trainable_text_model(user_dim, maxrecord) 
model.fit(train_X,train_Y,epochs = 1000)


def evaluation_last_with_distance(all_output_array, all_test_Y):
    count, all_recall1, all_recall2, all_recall3, all_recall4, all_recall5, alldistance = 0.,0.,0.,0.,0.,0.,0.
    for j in range(len(all_test_Y)):
        y_test = all_test_Y[j]
        output_array = all_output_array[j]
        for i in range(len(y_test)):
            flag = False
            if ((i+1)<len(y_test)):
                if (y_test[i] != 0) & (y_test[i+1]==0):
                    flag = True
            else:
                if y_test[i] != 0:
                    flag =True
            if flag:
                true_pl = y_test[i]-1
                infe_pl = output_array[i]
                #topd = infe_pl[1:].argsort()[-5:][::-1]
                
                tr = center_location_list[true_pl]
                pred = infe_pl[1:].argsort()[-1:]
                dst = distance.euclidean(tr, pred)
                alldistance += dst
                '''
                dd = []
                for k in topd:
                    pred = center_location_list[k]
                    tr = center_location_list[true_pl]
                    d = haversine(pred[0], pred[1], tr[0], tr[1])
                    dd.append(d)
                d = min(dd)
                alldistance += d
                '''
                if true_pl in infe_pl[1:].argsort()[-1:]: all_recall1 += 1
                if true_pl in infe_pl[1:].argsort()[-5:]: all_recall2 += 1
                if true_pl in infe_pl[1:].argsort()[-10:]: all_recall3 += 1
                if true_pl in infe_pl[1:].argsort()[-15:]: all_recall4 += 1
                if true_pl in infe_pl[1:].argsort()[-20:]: all_recall5 += 1
                count += 1
                print(count)
    return [all_recall1 / count, all_recall2 / count,
            all_recall3 / count, all_recall4 / count, all_recall5 / count, alldistance / count]
    
result_1 = model.predict(vali_X_1)
p1_1,p2_1,p3_1,p4_1,p5_1, errdistance_1 = evaluation_last_with_distance(result_1, vali_evl_1)

result_2 = model.predict(vali_X_2)
p1_2,p2_2,p3_2,p4_2,p5_2, errdistance_2 = evaluation_last_with_distance(result_2, vali_evl_2)

result_3 = model.predict(vali_X_3)
p1_3,p2_3,p3_3,p4_3,p5_3, errdistance_3 = evaluation_last_with_distance(result_3, vali_evl_3)

result_4 = model.predict(vali_X_4)
p1_4,p2_4,p3_4,p4_4,p5_4, errdistance_4 = evaluation_last_with_distance(result_4, vali_evl_4)

result_5 = model.predict(vali_X_5)
p1_5,p2_5,p3_5,p4_5,p5_5, errdistance_5 = evaluation_last_with_distance(result_5, vali_evl_5)

    
'''    
length1 = int(0.2*len(result[0]))
length2 = int(0.4*len(result[0]))
length3 = int(0.6*len(result[0]))
length4 = int(0.8*len(result[0]))

a = result[:,0:length1]
b = vali_evl[:0:length1]
    
p1_1,p2_1,p3_1,p4_1,p5_1, errdistance_1 = evaluation_last_with_distance(a, b)
p1_2,p2_2,p3_2,p4_2,p5_2, errdistance_2 = evaluation_last_with_distance(result[:,length1: length2], vali_evl[:,length1: length2])
p1_3,p2_3,p3_3,p4_3,p5_3, errdistance_3 = evaluation_last_with_distance(result[:,length2:length3], vali_evl[:,length2: length3])
p1_4,p2_4,p3_4,p4_4,p5_4, errdistance_4 = evaluation_last_with_distance(result[:,length3: length4], vali_evl[:,length3: length4])
p1_5,p2_5,p3_5,p4_5,p5_5, errdistance_5 = evaluation_last_with_distance(result[:,length4:], vali_evl[:,length4:])

p1 = (p1_1 + p1_2 + p1_3 + p1_4 + p1_5)/5
p2 = (p2_1 + p2_2 + p2_3 + p2_4 + p2_5)/5
p3 = (p3_1 + p3_2 + p3_3 + p3_4 + p3_5)/5
p4 = (p4_1 + p4_2 + p4_3 + p4_4 + p4_5)/5
p5 = (p5_1 + p5_2 + p5_3 + p5_4 + p5_5)/5

errdistance = (errdistance_1+errdistance_2+errdistance_3+errdistance_4+errdistance_5)/5

'''

'''
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")

'''
          
'''
col0 = 0
col1 = 1
spatio = [ finaldata[i][col0] for i in range(len(finaldata)) ]
temporal = [ finaldata[i][col1] for i in range(len(finaldata)) ]

###parameters
place_dim = GRID_COUNT*GRID_COUNT
time_dim=48
pl_d=50
time_k=50
hidden_neurons=50
learning_rate=0.001


###train test apart
train_test_part=0.9
train_pl = []
train_time = []
train_y = []

test_pl = []
test_time = []
test_y = []

train_size = int(len(spatio)*train_test_part)+1

for sample in range(0, train_size):
    train_pl.append(np.array(spatio[sample]))
    train_time.append(np.array(temporal[sample]))
    train_y.append(to_categorical(spatio[sample+1], num_classes=place_dim + 1))

for sample in range(train_size, len(spatio)-1):
    test_pl.append(np.array(spatio[sample]))
    test_time.append(np.array(temporal[sample]))
    test_y.append(to_categorical(spatio[sample+1], num_classes=place_dim + 1))
    

train_pl = np.array(train_pl)
train_time = np.array(train_time)
train_y = np.array(train_y)

test_pl = np.array(test_pl)
test_time = np.array(test_time)
test_y = np.array(test_y)




pl_input = Input(shape=(1,), dtype='float64', name = 'pl_input')
time_input = Input(shape=(1,), dtype='float64', name = 'time_input')

pl_embedding = Embedding(input_dim=place_dim+1, output_dim=pl_d, name ='pl_embedding' ,mask_zero=True)(pl_input)
time_embedding = Embedding(input_dim=time_dim+1 , output_dim=time_k, name='time_embedding',mask_zero=True)(time_input)

attrs_latent = merge([pl_embedding,time_embedding],mode='concat')
    # time_dist = TimeDistributed(Dense(50))
lstm_out = LSTM(hidden_neurons, return_sequences=False,name='lstm_layer0')(attrs_latent)
    # lstm_out = LSTM(hidden_neurons, return_sequences=True, name='lstm_layer1')(lstm_out)
    # lstm_out = LSTM(hidden_neurons, return_sequences=True, name='lstm_layer2')(lstm_out)
dense = Dense(place_dim+1 , name='dense')(lstm_out)
pred = Activation('softmax')(dense)
#model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
model = Model(inputs=[pl_input,time_input], outputs=pred)

    # model.load_weights('./model/User_RNN_Seg_Epoch_0.3_rmsprop_300.h5')
    # Optimization
#sgd = optimizers.SGD(lr=learning_rate)
adam = optimizers.Adam(lr=learning_rate)
model.compile(optimizer=adam, loss='categorical_crossentropy')
model.summary()
model.fit([train_pl,train_time],train_y, epochs = 2000)
result = model.predict([test_pl,test_time])
decision = result.argsort()

###the last five number tell the most possible location index
location10 = decision[:,2491:2501]
spatio13 = spatio[132:145]

'''
