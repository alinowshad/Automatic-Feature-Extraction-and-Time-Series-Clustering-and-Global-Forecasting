from tcn import TCN
import csv
import os
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import SpectralClustering
# %tensorflow_version 1.x
import tensorflow as tf
import keras
import keras.backend as K
from keras import layers
from keras.models import Sequential,Model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import time
print(tf.__version__)
from keras.layers import MultiHeadAttention
from keras.layers import Dense
import gc
from keras.layers import concatenate
import csv
import math
import warnings
import os
# import xgboost as xgb
warnings.filterwarnings('ignore')
# import GPy, GPyOpt
tfkl = tf.keras.layers
tfk = tf.keras
from rstl import STL
from texttable import Texttable
from sklearn.metrics import silhouette_score



def get_dataset_params(dataset_name = 'nn5'):
    suilin_smape = False
    dataset_path = "nothing"
    if dataset_name == 'nn5':
        raw_data = pd.read_csv('/home/anoshad.t1@gigroup.local/workarea/es_analysis/ExperimentLSTM/dataset/nn5.csv',sep='delimeter',header=None)
        features = pd.read_csv("/home/anoshad.t1@gigroup.local/workarea/es_analysis/ExperimentLSTM/features/nn5 lstm/Features_nn5_LSTM_16.csv",sep=',', header=None)


        lag = 150
        look_forward = 56
        batch_size = 1
        epochs = 1
        learning_rate = 0.0001
        suilin_smape = False
        frequency =7

    #------------------------------------------------------------------------------------------------------------#

    sample_overlap = look_forward - 1


    raw_data = raw_data[0].str.split(',', expand=True)

    raw_data = raw_data.to_numpy().astype('float64')
    features = features.to_numpy().astype('float64')
    dataset = []
    for i in range(len(raw_data)):
        dataset.append(raw_data[i][~np.isnan(raw_data[i])])


    return dataset, features, lag, look_forward, sample_overlap, learning_rate, dataset_path, suilin_smape, frequency

def normalize_dataset(dataset, look_forward ):
    data_means = [];
    for index in range(len(dataset)):
    # Mean Noramlization
        series_mean = np.mean(dataset[index][:len(dataset[index]) - look_forward]) # Train Mean: look_forward || Full Mean: Mean: look_forward = 0

        if series_mean == 0:
            series_mean = 0.001

        data_means.append(series_mean)
        dataset[index] = np.divide(dataset[index], series_mean)

        # Log Transformation
        dataset[index] = np.log(dataset[index] + 1)

    return dataset, np.array(data_means)

def rescale_data_to_main_value(data, means, dataset_seasonal = []):

    for index in range(len(data)):
        # Revert Log Transformation
        data[index] = np.e ** data[index]
        data[index] = data[index] - 1

        # Revert Mean Normalization
        data[index] = means[index] * data[index]

        if len(dataset_seasonal) != 0:
            data[index] = data[index] + dataset_seasonal[index]


    return data

def normalize_feature_vectors(features):
    #------------------- Z-score ----------------------#
#     means = features.mean(0)
#     stds = features.std(0)

#     for i in range(len(features)):
#         features[i] = (features[i] - means) / stds

    #--------------------Min - Max---------------------#
    minimum = features.min(0)
    maximum = features.max(0)

    for i in range(len(features)):
        features[i] = (features[i] - minimum) / (maximum - minimum)


    return features

"""![root_mean_square_deviation.svg](attachment:root_mean_square_deviation.svg)"""

#RMSE
def root_mean_squared_error(actual, forecast, method = 'single_value'):
    # Methods = single_value | per_series
    if method == 'single_value':
        #Flatten To One Vector
        actual = actual.flatten()
        forecast = forecast.flatten()

        return np.sqrt(np.mean(np.square(actual - forecast)))
    elif method == 'per_series':
        rmses = []
        for i in range(len(actual)):
            rmses.append(np.sqrt(np.mean(np.square(actual[i] - forecast[i]))))

        return rmses

"""![YIy33.png](attachment:YIy33.png)"""

#SMAPE
def single_point_smape(actual, forecast, suilin_smape = False):
    if suilin_smape == True:
        epsilon = 0.1

        return (np.sum(2 * np.abs(forecast - actual) / max((np.abs(actual) + np.abs(forecast))+ epsilon, 0.5 + epsilon)))
    else:
        return (np.sum(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast))))

def smape(actual, forecast, method = 'single_value', suilin_smape = False):
    # Methods = single_value | per_series
    if method == 'single_value':
        #Flatten To One Vector
        actual = actual.flatten()
        forecast = forecast.flatten()
        sum_smape = 0
        for i in range(len(actual)):
            sum_smape += single_point_smape(actual[i], forecast[i], suilin_smape)
        return 100 * sum_smape / len(actual)

    elif method == 'per_series':
        smapes = []
        for i in range(len(actual)):
            sum_smape = 0
            for j in range(len(actual[i])):
                sum_smape += single_point_smape(actual[i,j], forecast[i,j], suilin_smape)
            smapes.append(100 * sum_smape / len(actual[i]))
        return np.array(smapes)

# Create Samples from DataSet
def create_dataset(sample, look_back, look_forward, sample_overlap, dataset_seasonal):
    if(sample_overlap >= look_forward or sample_overlap < 0): sample_overlap = look_forward - 1
    if(look_forward == 1): sample_overlap = 0

    dataX, dataY, dataY_seasonal = [], [], []
    dataX_means, dataY_means = [], []
    for i in range(0, len(sample) - look_back - look_forward+1, look_forward - sample_overlap):
        dataX.append(sample[i:(i+look_back), 0])
        dataY.append(sample[(i + look_back):(i + look_back + look_forward), 0])

        dataY_seasonal.append(dataset_seasonal[(i + look_back):(i + look_back + look_forward)])


    return np.array(dataX), np.array(dataY), np.array(dataY_seasonal)


#RMSE
def root_mean_squared_error(actual, forecast, method = 'single_value'):
    # Methods = single_value | per_series
    if method == 'single_value':
        #Flatten To One Vector
        actual = actual.flatten()
        forecast = forecast.flatten()

        return np.sqrt(np.mean(np.square(actual - forecast)))
    elif method == 'per_series':
        rmses = []
        for i in range(len(actual)):
            rmses.append(np.sqrt(np.mean(np.square(actual[i] - forecast[i]))))

        return rmses

"""![YIy33.png](attachment:YIy33.png)"""

#SMAPE
def single_point_smape(actual, forecast, suilin_smape = False):
    if suilin_smape == True:
        epsilon = 0.1

        return (np.sum(2 * np.abs(forecast - actual) / max((np.abs(actual) + np.abs(forecast))+ epsilon, 0.5 + epsilon)))
    else:
        return (np.sum(2 * np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast))))

def smape(actual, forecast, method = 'single_value', suilin_smape = False):
    # Methods = single_value | per_series
    if method == 'single_value':
        #Flatten To One Vector
        actual = actual.flatten()
        forecast = forecast.flatten()
        sum_smape = 0
        for i in range(len(actual)):
            sum_smape += single_point_smape(actual[i], forecast[i], suilin_smape)
        return 100 * sum_smape / len(actual)

    elif method == 'per_series':
        smapes = []
        for i in range(len(actual)):
            sum_smape = 0
            for j in range(len(actual[i])):
                sum_smape += single_point_smape(actual[i,j], forecast[i,j], suilin_smape)
            smapes.append(100 * sum_smape / len(actual[i]))
        return np.array(smapes)

# Create Samples from DataSet
def create_dataset(sample, look_back, look_forward, sample_overlap, dataset_seasonal):
    if(sample_overlap >= look_forward or sample_overlap < 0): sample_overlap = look_forward - 1
    if(look_forward == 1): sample_overlap = 0

    dataX, dataY, dataY_seasonal = [], [], []
    dataX_means, dataY_means = [], []
    for i in range(0, len(sample) - look_back - look_forward+1, look_forward - sample_overlap):
        dataX.append(sample[i:(i+look_back), 0])
        dataY.append(sample[(i + look_back):(i + look_back + look_forward), 0])

        dataY_seasonal.append(dataset_seasonal[(i + look_back):(i + look_back + look_forward)])


    return np.array(dataX), np.array(dataY), np.array(dataY_seasonal)

def create_sample(look_forward,sample_seasonal,dataX, dataY, data_mean, dataY_seasonal,frequency):
    test_size=1
    val_size=1

    train_size=(len(dataX)-test_size)

    trainX, testX = dataX[0:train_size,:], dataX[train_size:,:]
    trainY, testY = dataY[0:train_size,:], dataY[train_size:,:]

    valX, valY = trainX[train_size-val_size:train_size,:],trainY[train_size-val_size:train_size, :]

    trainX = np.reshape(trainX, (trainX.shape[0],1, trainX.shape[1]))
    valX = np.reshape(valX, (valX.shape[0],1, valX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0],1, testX.shape[1]))

    val_means = np.full(len(valY), data_mean)
    test_means = np.full(len(testY), data_mean)

    val_seasonal = dataY_seasonal[train_size-val_size:train_size, :]
    

    train=dataY_seasonal[:train_size, :]
    train=train.reshape(-1,1)
    train2=train[:len(train):len(valY[0])]
    
    test2 = []


    
    sample_size = len(sample_seasonal.flatten()) - look_forward

    train3=sample_seasonal[:sample_size].flatten()

    if frequency!=None:
        if len(train3.flatten()) > frequency*2:
            sp = frequency
            fit1 = ExponentialSmoothing(endog=pd.Series(train3.flatten()), seasonal_periods=sp,trend='add', seasonal='add').fit()

        elif len(train3.flatten())<frequency*2 and len(train3.flatten())>frequency :
            sp = int(frequency/2)
            fit1 = ExponentialSmoothing(endog=pd.Series(train3.flatten()), seasonal_periods=sp, trend='add',
                                        seasonal='add').fit()
        else:
            fit1 = ExponentialSmoothing(endog=pd.Series(train3.flatten())).fit()

        preds2 = fit1.forecast(steps=len(valY[0])).values.reshape(1,-1)

        # print("preds",preds2,type(preds2))
    else:
        preds2=np.zeros(look_forward)
    for i in range(0,len(val_seasonal[0])):
        test2.append(val_seasonal[0][len(val_seasonal[0])-1])

    # print("datay_seasonal",dataY_seasonal)

    test_seasonal_y = dataY_seasonal[train_size:,:]
    # print("test",test_seasonal_y)
    # print("train2",train2.flatten())
    # np.savetxt('train.csv', train2.flatten(), delimiter=', ')

    return np.array(trainX),np.array(valX),np.array(testX),np.array(trainY),np.array(valY),np.array(testY), test_means, val_means, val_seasonal,test_seasonal_y, preds2


# Preprocess Data For Sampling
def all_pre_process(all_dataset, lag, look_forward, sample_overlap, data_means, dataset_seasonal,frequency):
    look_back = lag

    trainX = []
    trainY = []

    valX = []
    valY = []

    testX = []
    testY = []

    all_test_means = []
    all_val_means = []

    all_test_seasonals = []
    all_test_seasonals2 = []
    all_val_seasonals = []

    for index in range(len(all_dataset)):
        sample = np.array(all_dataset[index])
        sample = sample.reshape(sample.shape[0], 1)
        # sample_seasonal=np.array(dataset_seasonal[index])

        dataX_s, dataY_s, dataY_seasonal = create_dataset(sample, look_back, look_forward, sample_overlap, dataset_seasonal[index])
        temp_trainX, temp_valX, temp_testX, temp_trainY, temp_valY, temp_testY, test_means, val_means, val_seasonal, test_seasonal,test2 = create_sample(look_forward,dataset_seasonal[index],dataX_s,dataY_s,data_means[index], dataY_seasonal,frequency)

        trainX = trainX + temp_trainX.tolist()
        trainY = trainY + temp_trainY.tolist()

        valX = valX + temp_valX.tolist()
        valY = valY + temp_valY.tolist()

        testX = testX + temp_testX.tolist()
        testY = testY + temp_testY.tolist()

        all_test_means = all_test_means + test_means.tolist()
        all_val_means = all_val_means + val_means.tolist()
        all_test_seasonals = all_test_seasonals +  test_seasonal.tolist()
        all_test_seasonals2 = all_test_seasonals2 +test2.tolist()  # test_seasonal.tolist() #"NOTE"
        all_val_seasonals = all_val_seasonals + val_seasonal.tolist()


    return np.array(trainX), np.array(valX), np.array(testX), np.array(trainY), np.array(valY), np.array(testY), np.array(all_test_means), np.array(all_val_means), np.array(all_val_seasonals), np.array(all_test_seasonals),np.array(all_test_seasonals2)


def save_prediction_result(data, dataset_name = 'cif-6', dataset_path = ''):
    if dataset_name == '':
        filename = dataset_name + '-results.csv'
    else:
        filename = dataset_path + '/' + dataset_name + '-results.csv'


    df = pd.DataFrame(data)
    df.to_csv(filename, sep=',',index=False,header=False)

"""#@main"""

# Main Work & Functionality
def run_model_test(dataset, data_means, dataset_seasonal, dataset_name, cluster_lable, lag, look_forward, sample_overlap, batch_size, epochs, learning_rate, suilin_smape, dataset_path,frequency, use_saved_model = False, save_trained_model = False):
    print("len dataset",len(dataset))
    # Initialize Look Forward & Back
    look_back=lag
    calculations_method = 'per_series' # single_value | per_series

    trainX, valX, testX, trainY, valY, testY, test_means, val_means, val_seasonal, test_seasonal,test_seasonal2 = all_pre_process(dataset, lag, look_forward, sample_overlap, data_means, dataset_seasonal,frequency)



    # Get Model From Local Saved File
    if use_saved_model == True:
        if os.path.exists(dataset_path + '/' + dataset_name + '-model-cluster-' + str(cluster_lable)) == True:
            model = keras.models.load_model(dataset_path + '/' + dataset_name  + '-model-cluster-' + str(cluster_lable))

            val_prediction_results = model.predict([valX],batch_size=16, verbose=0)

            val_RMSE = root_mean_squared_error(valY, val_prediction_results, calculations_method)
            val_SMAPE = smape(valY, val_prediction_results, calculations_method, suilin_smape)

            ######################################################
            test_prediction_results = model.predict([testX],batch_size=16, verbose=0)

            test_RMSE = root_mean_squared_error(testY, test_prediction_results, calculations_method)
            test_SMAPE = smape(testY, test_prediction_results, calculations_method, suilin_smape)

        else:
            use_saved_model = False
            save_trained_model = True

    # Train Model From Scratch
    if use_saved_model == False:

        dense_neuron = 100
        denselayer_activation = 'linear' #None
        output_activation = 'linear' #'linear'

        print("---------------------------------------------------------------------")
        print("lag", lag)
        print("look_forward", look_forward)
        print("sample overlap", sample_overlap)
        print("trainshape", trainX.shape)
        print("valshape", valX.shape)
        print("testshape", testX.shape)
        print(learning_rate, dense_neuron, denselayer_activation, output_activation)



        validation_loss=[]
        test_loss=[]
        iter = 1
        for j in range(iter):
            # #---------------------------------------Input Layer------------------------------------------#
            input_layer = layers.Input(shape = (1, lag,), name = "Input-Layer")



            multi_head_attention_layer = TCN(return_sequences=True,dilations=[1, 2, 4, 8])(input_layer)
            conv = keras.layers.Conv1D(64,
                              strides=2,
                              kernel_size=4,
                              activation=None,
                              padding="same",)(multi_head_attention_layer)#multi_head_attention_layer
            conv2 = keras.layers.Conv1D(16,
                              strides=2,
                              kernel_size=4,
                              activation=None,
                              padding="same",)(conv)
            flatten_layer2=keras.layers.Flatten(name="Flatten-Layer2")(conv2)

            dense_layer1 = Dense(
                dense_neuron,
                activation = denselayer_activation,
                name = "Fully-Connected-Layer")(flatten_layer2)

            dense_layer2 = Dense(
                look_forward,
                activation = None,
                name = "Output-Layer")(dense_layer1)




            # Create Model
            model = Model(inputs = [input_layer], outputs = dense_layer2)

            # Optimizer
            opt=tf.keras.optimizers.Adam(learning_rate=learning_rate)

            # Compile Model
            model.compile(loss="mse",optimizer=opt,metrics=["mse"])

            for k in range(epochs): #epochs
                history = model.fit([trainX], trainY, validation_data=([valX, valY]),
                    verbose = 0,
                    batch_size = batch_size,
            ).history

                val_prediction_results = model.predict([valX],batch_size=16, verbose=0)

                val_RMSE = root_mean_squared_error(valY, val_prediction_results, calculations_method)
                val_SMAPE = smape(valY, val_prediction_results, calculations_method, suilin_smape)

                ######################################################
                test_prediction_results = model.predict([testX],batch_size=16, verbose=0)

                test_RMSE = root_mean_squared_error(testY, test_prediction_results, calculations_method)
                test_SMAPE = smape(testY, test_prediction_results, calculations_method, suilin_smape)

                validation_loss.append(np.mean(val_RMSE))
                test_loss.append(np.mean(test_RMSE))

            K.clear_session()
            gc.collect()

            # Save model to a file if wanted
            if save_trained_model == True:
                model.save(dataset_path + '/' + dataset_name + '-model-cluster-' + str(cluster_lable))

            model=None
            del history


    rescaled_valY = rescale_data_to_main_value(valY, val_means, val_seasonal)
    rescaled_val_prediction_results = rescale_data_to_main_value(val_prediction_results, val_means,val_seasonal) #####
    val_SMAPE = smape(rescaled_valY, rescaled_val_prediction_results, calculations_method, suilin_smape)
    val_RMSE = root_mean_squared_error(rescaled_valY, rescaled_val_prediction_results, calculations_method)


    rescaled_testY = rescale_data_to_main_value(testY, test_means, test_seasonal)
    rescaled_test_prediction_results = rescale_data_to_main_value(test_prediction_results, test_means,test_seasonal2 ) ###,test_seasonal
    test_SMAPE = smape(rescaled_testY, rescaled_test_prediction_results, calculations_method)
    test_RMSE = root_mean_squared_error(rescaled_testY, rescaled_test_prediction_results, calculations_method)
    results = {
            'val_SMAPE': val_SMAPE,
            'val_RMSE': val_RMSE,
            'test_SMAPE': test_SMAPE,
            'test_RMSE': test_RMSE
        }
    #############################################

    return results

def cluster_series(features, number_of_clusters=2):
    clustered = KMedoids(n_clusters=number_of_clusters, init='k-medoids++',random_state=0).fit(features) # Kmedoids init='k-medoids++'
    print("Kmedoids")
    print('silhouette_score -------->', silhouette_score(features, clustered.labels_))
    return clustered.labels_

def stl_decomposition(dataset, frequency):
    seasonal = []
    trend = []
    for index in range(len(dataset)):
        if frequency != None:
            stl = STL(dataset[index], frequency, "periodic")

            seasonal.append(stl.seasonal)
            trend.append(stl.trend)
            dataset[index] = dataset[index] - stl.seasonal
        else:
            seasonal.append(np.zeros((dataset[index].shape)))
            trend.append(np.zeros((dataset[index].shape)))

    return dataset, np.array(seasonal), np.array(trend)

def run_local_models(dataset_name, number_of_clusters=2, AEName='LSTM', Dim=8, epochs = 20, batch =20, use_saved_model = False, save_trained_model = False, run=1):
    import gc
    gc.collect()
    print('dataset: ', dataset_name)
    batch_size = batch
    epochs = epochs
    # Prepare & Read Data
    dataset, features, lag, look_forward, sample_overlap, learning_rate, dataset_path, suilin_smape, frequency = get_dataset_params(dataset_name)

    # Normalize Data
    dataset, data_means = normalize_dataset(dataset, look_forward = 0)

    dataset, seasonal, trend = stl_decomposition(dataset, frequency)

    # Normalize Features
    features = normalize_feature_vectors(features)

    # Cluster Series Based On Feature Vectors (Feature Based Clustering)
    if number_of_clusters == 1:
        clusters = np.zeros(len(features))
    else:
        clusters = cluster_series(features, number_of_clusters)

    dataset = np.array(dataset)

    results = {
        'val_SMAPE': np.array([]),
        'val_RMSE': np.array([]),
        'test_SMAPE': np.array([]),
        'test_RMSE': np.array([])
    }

    # Loop Trough Clusters
    for cluster_lable in range(number_of_clusters):
        idx = [x for x in range(len(clusters)) if clusters[x] == cluster_lable]
        cluster_dataset = np.array(dataset)[idx]
        cluster_dataset_means = data_means[idx]
        cluster_dataset_seasonal = seasonal[idx]

        result = run_model_test(cluster_dataset, cluster_dataset_means, cluster_dataset_seasonal, dataset_name, cluster_lable, lag, look_forward, sample_overlap, batch_size, epochs, learning_rate, suilin_smape, dataset_path, frequency, use_saved_model, save_trained_model)

        results = {
            'val_SMAPE': np.concatenate((results['val_SMAPE'], result['val_SMAPE'])),
            'val_RMSE': np.concatenate((results['val_RMSE'], result['val_RMSE'])),
            'test_SMAPE': np.concatenate((results['test_SMAPE'], result['test_SMAPE'])),
            'test_RMSE': np.concatenate((results['test_RMSE'], result['test_RMSE'])),
        }

    t = Texttable()
    print('\n\n#------------------------------------Scaled------------------------------------#')
    t.add_rows([
        ['Index', 'Mean sMAPE', 'Median sMAPE', 'Mean RMSE', 'Median RMSE'],
        ['Validate', np.mean(results['val_SMAPE']), np.median(results['val_SMAPE']), np.mean(results['val_RMSE']), np.median(results['val_RMSE'])],
        ['Test', np.mean(results['test_SMAPE']), np.median(results['test_SMAPE']), np.mean(results['test_RMSE']), np.median(results['test_RMSE'])]
    ])
    
    print(t.draw())
    t = Texttable()
    t.add_rows([
        ['dataset Name', 'Run', 'N.Epochs', 'Batch Size', 'Auto Encoder', 'Latent Dimension'],
        [dataset_name, run, epochs,batch_size, AEName, Dim]
    ])
    print(t.draw())
    initial_data = [
    ['dataset Name', 'Run', 'N.Epochs', 'Batch Size', 'Auto Encoder', 'Latent Dimension', 'Index', 'Mean sMAPE', 'Median sMAPE', 'Mean RMSE', 'Median RMSE'],
    [dataset_name, run, epochs, batch_size, AEName, Dim, 'Validate', np.mean(results['val_SMAPE']), np.median(results['val_SMAPE']), np.mean(results['val_RMSE']), np.median(results['val_RMSE'])],
    [dataset_name, run, epochs, batch_size, AEName, Dim, 'Test', np.mean(results['test_SMAPE']), np.median(results['test_SMAPE']), np.mean(results['test_RMSE']), np.median(results['test_RMSE'])]
    ]
    return results, initial_data

def nn5(AEName="LSTM", Dim=16, run=1):
    
    ds_names = [AEName, [5, 15, 25, 35, 50], [20, 40, 60, 80, 100], Dim]
    file_mode = 'w'
    csv_file = "NN5_results_"+AEName+"_"+str(Dim)+".csv"
    if os.path.exists(csv_file):
        file_mode = 'a'
    else:
        file_mode = 'w'

    with open(csv_file, mode=file_mode, newline='') as file:
        writer = csv.writer(file)
        for epoch in ds_names[1]:
            for batch in ds_names[2]:
    
                hospital_results, initial_data = run_local_models(dataset_name = 'nn5', number_of_clusters = 2, AEName=ds_names[0], Dim=ds_names[3], epochs = epoch, batch = batch, use_saved_model = False, save_trained_model = False, run=run)
                writer.writerows(initial_data)


for i in range(2):
    print("*" * 50)
    nn5(run = i+3)