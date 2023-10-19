# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 21:46:12 2023

@author: rehan
"""

from imblearn.over_sampling import RandomOverSampler
import numpy as np
np.random.seed(1234)
import random
random.seed(1234)
import pandas as pd

import matplotlib.pyplot as plt
import tqdm
from sklearn.model_selection import train_test_split

import os
import glob
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score, roc_auc_score,f1_score,precision_score,recall_score, roc_curve
from scipy.stats import ttest_rel

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
tf.random.set_seed(1234)
from tensorflow.keras.layers import BatchNormalization, ReLU, GRU, Input, SpatialDropout1D, Bidirectional, MaxPooling2D, MaxPooling1D, Conv1D, Dense, Flatten, Dropout, LSTM, concatenate, Conv2D
from tensorflow.keras.optimizers import Adam,SGD
# from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import pickle

import warnings
warnings.filterwarnings("ignore")

input_fodler = 'whole_data' # Data Folder Path



import keras.backend as K
from keras.layers import Layer

class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res


# class SelfAttention(tf.keras.layers.Layer):
#     def __init__(self, d_model):
#         super(SelfAttention, self).__init__()
#         self.d_model = d_model

#     def build(self, input_shape):
#         self.W_q = self.add_weight(name="W_q", shape=(self.d_model, self.d_model), initializer="uniform")
#         self.W_k = self.add_weight(name="W_k", shape=(self.d_model, self.d_model), initializer="uniform")
#         self.W_v = self.add_weight(name="W_v", shape=(self.d_model, self.d_model), initializer="uniform")

#     def call(self, inputs):
#         q = tf.matmul(inputs[0], self.W_q)
#         k = tf.matmul(inputs[1], self.W_k)
#         v = tf.matmul(inputs[2], self.W_v)

#         attention_scores = tf.matmul(q, k, transpose_b=True)
#         attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.d_model, tf.float32))

#         attention_weights = tf.nn.softmax(attention_scores, axis=-1)
#         output = tf.matmul(attention_weights, v)

#         return output                   


class Models_class__:
    def __init__(self):
        self.epochs = 100
        self.batch_size = 32
        self.verbose = 1
        self.neurons_models = 128
        self.lr = 0.01
        self.lat_layer_activation = 'softmax'
        self.early_Stop = 100
        self.group_rows = 5
        self.chennels = 8
        self.classes = 4
        self.remove_from_start = 0 # keep 0 if nothing to drop from start
        self.remove_from_end = 1 # keep 1 if nothing to drop from end

        self.histories = []
        self.accuracies = {}
        self.accuracies['models'] = ['proposed','Stacked','CNN','LSTM','GRU',
                                      'lstm_lr', 'lstm_svm', 'lstm_dt',
                                      'cnn_lr', 'cnn_svm', 'cnn_dt',
                                      'gru_lr', 'gru_svm', 'gru_dt'

                                      ]

        self.cohen_kappa_score = {}
        self.cohen_kappa_score['models'] = ['proposed','Stacked','CNN','LSTM','GRU',
                                      'lstm_lr', 'lstm_svm', 'lstm_dt',
                                      'cnn_lr', 'cnn_svm', 'cnn_dt',
                                      'gru_lr', 'gru_svm', 'gru_dt'

                                      ]

        self.roc_auc_score = {}
        self.roc_auc_score['models'] = ['proposed','Stacked','CNN','LSTM','GRU',
                                      'lstm_lr', 'lstm_svm', 'lstm_dt',
                                      'cnn_lr', 'cnn_svm', 'cnn_dt',
                                      'gru_lr', 'gru_svm', 'gru_dt'

                                      ]

        self.f1_score = {}
        self.f1_score['models'] = ['proposed','Stacked','CNN','LSTM','GRU',
                                      'lstm_lr', 'lstm_svm', 'lstm_dt',
                                      'cnn_lr', 'cnn_svm', 'cnn_dt',
                                      'gru_lr', 'gru_svm', 'gru_dt'

                                      ]
        self.precision_score = {}
        self.precision_score['models'] = ['proposed','Stacked','CNN','LSTM','GRU',
                                      'lstm_lr', 'lstm_svm', 'lstm_dt',
                                      'cnn_lr', 'cnn_svm', 'cnn_dt',
                                      'gru_lr', 'gru_svm', 'gru_dt'

                                      ]
        self.recall_score = {}
        self.recall_score['models'] = ['proposed','Stacked','CNN','LSTM','GRU',
                                      'lstm_lr', 'lstm_svm', 'lstm_dt',
                                      'cnn_lr', 'cnn_svm', 'cnn_dt',
                                      'gru_lr', 'gru_svm', 'gru_dt'

                                      ]

        self.classes_names = [str(i) for i in range(self.classes)]
        if len(self.classes_names)==2:
            self.Multi_class = False

        elif len(self.classes_names)>2:
            self.Multi_class = True

    def perform_t_test(self,df):

        m = len(df)-1
        proposed = df.iloc[-1,1:-1]
        p_value_thres = 0.05
        models_name = ['proposed','Stacked','CNN','LSTM','Bi-LSTM',
                                      'lstm_lr', 'lstm_svm', 'lstm_dt',
                                      'cnn_lr', 'cnn_svm', 'cnn_dt',
                                      'bi_lstm_lr', 'bi_lstm_svm', 'bi_lstm_dt'

                                      ]
        all_p_values = []
        for i in range(m):
            p_value = ttest_rel(a=proposed, b=df.iloc[i,1:-1]).pvalue
            check_p_value = p_value/m
            # print(models_name[i], p_value,'p/m',check_p_value)
            if check_p_value>p_value_thres:
                print('need to discard',models_name[i])
                m = m-1
            all_p_values.append(p_value)

        all_p_values.append(check_p_value)

        df['ttest_p_value'] = all_p_values

        return df

    def evaluation(self,
                   y_true,y_pred,name):

        auc = []
        auc.append(np.round(accuracy_score(y_true,y_pred),4))

        auc.append(np.round(cohen_kappa_score(y_true,y_pred),4))
        if  self.Multi_class:
            auc.append(np.round(roc_auc_score(y_true,
                                              to_categorical(y_pred,num_classes=self.classes),
                                              multi_class =  'ovr'),4))
            auc.append(np.round(f1_score(y_true,y_pred,
                                         average = 'macro'),4))
            auc.append(np.round(precision_score(y_true,y_pred,
                                                average = 'macro'),4))
            auc.append(np.round(recall_score(y_true,y_pred,
                                             average = 'macro'),4))
        else:
            auc.append(np.round(roc_auc_score(y_true,y_pred),4))
            auc.append(np.round(f1_score(y_true,y_pred),4))
            auc.append(np.round(precision_score(y_true,y_pred),4))
            auc.append(np.round(recall_score(y_true,y_pred),4))

        print(name+' accuracy:',auc[0])

        cm = confusion_matrix(y_true,y_pred)

        disp = ConfusionMatrixDisplay(cm,
                                display_labels=self.classes_names)
        disp.plot()
        plt.title('Confusion matrix',fontsize=20, fontweight='bold')

        plt.xlabel('Predicted Values',fontsize=18, fontweight='bold')
        plt.ylabel('True Values',fontsize=18, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder,name+'_confusion_matrics.png'))
        report = classification_report(y_true,y_pred, target_names=self.classes_names,output_dict=True) #classification report
        df = pd.DataFrame(report).transpose()

        df.insert(0, "", [str(i) for i in range(self.classes)] +['accuracy','macro avg','weighted avg'])
        df.to_csv(os.path.join(self.output_folder,name+'_classification_report.csv'),index = False)

        return auc
    def get_CNN(self, input_size):
        # Initialising the CNN
        model = Sequential()

        model.add(Conv1D(self.neurons_models, kernel_size=2,
                  activation='relu', input_shape=input_size,
                  kernel_regularizer = 'L1L2',
                  bias_regularizer = 'L2',
                  
                  ))
        model.add(Conv1D(self.neurons_models, kernel_size=2, activation='relu',
                          kernel_regularizer = 'L1L2',
                          activity_regularizer = 'L1L2'))
        model.add(SpatialDropout1D(0.2))
        model.add(MaxPooling1D(pool_size=3,data_format='channels_last'))
        # model.add(BatchNormalization())
        # model.add(Conv1D(self.neurons_models, kernel_size=2, activation='relu'))

        model.add(Flatten())
        
        # model.add(Dropout(0.1))
        model.add(Dense(32, activation='relu',kernel_regularizer = 'L1L2',
                        activity_regularizer = 'L1L2'))
        
        model.add(BatchNormalization())
        model.add(Dense(self.classes, activation=self.lat_layer_activation))

        # Compliling the model
        model.compile(optimizer=Adam(learning_rate=self.lr),
                      loss=CategoricalCrossentropy(),
                      metrics=['accuracy'])
        # print(model.summary())
        return model

    def get_LSTM(self, input_size):
        # Initialising the LSTM
        model = Sequential()

        model.add(LSTM(self.neurons_models, input_shape=input_size, return_sequences=True,
                       kernel_regularizer = 'L1L2',
                       bias_regularizer = 'L2',))
        model.add(LSTM(self.neurons_models, return_sequences=True,
                       kernel_regularizer = 'L1L2',
                       activity_regularizer = 'L1L2'))
        model.add(SpatialDropout1D(0.2))
        model.add(MaxPooling1D(pool_size=3))
        # model.add(BatchNormalization())

        model.add(Flatten())
        # model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu',kernel_regularizer = 'L1L2',
                        activity_regularizer = 'L1L2'))
        model.add(BatchNormalization())

        model.add(Dense(self.classes, activation=self.lat_layer_activation))

        # Compliling the model
        model.compile(optimizer=Adam(learning_rate=self.lr),
                      loss=CategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model

    def get_BI_LSTM(self, input_size):
     # Initialising the BI-LSTM
        model = Sequential()

        model.add(GRU(self.neurons_models, input_shape=input_size, return_sequences=True,
                       kernel_regularizer = 'L1L2',
                       bias_regularizer = 'L2',))
        model.add(GRU(self.neurons_models, return_sequences=True,
                       kernel_regularizer = 'L1L2',
                       activity_regularizer = 'L1L2'))
        model.add(SpatialDropout1D(0.2))
        model.add(MaxPooling1D(pool_size=3))
        # model.add(BatchNormalization())


        model.add(Flatten())
        # model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu',kernel_regularizer = 'L1L2',
                        activity_regularizer = 'L1L2'))
        model.add(BatchNormalization())

        model.add(Dense(self.classes, activation=self.lat_layer_activation))

        # Compliling the model
        model.compile(optimizer=Adam(learning_rate=self.lr),
                      loss=CategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model

    def train(self, path='',):

        MC = ModelCheckpoint(
            filepath=path,
            monitor="val_accuracy",
            verbose=0,
            mode="auto",
            save_best_only=True,
        )

        er = EarlyStopping(
            monitor="val_accuracy",
            min_delta=0,
            patience=self.early_Stop,
            verbose = self.verbose,
            mode="auto",
            baseline=None,
            restore_best_weights=True)

        lr = ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.1,
            patience=5,
            verbose = self.verbose,
            mode='auto',
            min_delta=0.01,
            cooldown=0,
            min_lr=0,)

        history = self.model.fit(self.X_train, self.Y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epochs,
                                 verbose=self.verbose,
                                 validation_data=(self.X_test, self.Y_test),
                                 callbacks=[MC, er, lr],
                                 )
        # self.histories.append(history)

        plt.figure(figsize=(15, 10))
        # plt.figure()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy',fontsize=26, fontweight='bold')
        plt.xlabel('Epochs',fontsize=22, fontweight='bold')
        plt.ylabel('Accuracy',fontsize=22, fontweight='bold')
        plt.legend(['train', 'test'], loc='upper left',fontsize=20)
        plt.xticks(fontsize=20, fontweight='bold')
        plt.yticks(fontsize=20, fontweight='bold')
        plt.ylim(0.0, 1.0)

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss',fontsize=26, fontweight='bold')
        plt.xlabel('Epochs',fontsize=22, fontweight='bold')
        plt.ylabel('Loss',fontsize=22, fontweight='bold')
        plt.legend(['train', 'test'], loc='upper left',fontsize=20)
        plt.xticks(fontsize=20, fontweight='bold')
        plt.yticks(fontsize=20, fontweight='bold')
        plt.ylim(0.0, 1.0)

        file_name = os.path.basename(path)[:-3]

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, file_name+'.png'))

        # plt.show()

        plt.close('all')
        # if 'proposed' in file_name:
        #     self.model = load_model(path,custom_objects={'SelfAttention':SelfAttention,
        #                                                  'RBFLayer':RBFLayer})
        # else:
        self.model = load_model(path)

    def create_dataset(self, csv_path):
        df = pd.read_csv(csv_path)
        
        # oxygeneted = [i for i in df.columns if '_O' in i] + [df.columns[-1]]
        # df = df[oxygeneted]

        # data = df.dropna()
# Separate features and labels
        # X = data.iloc[:, :-1]  # Features (exclude the last column)
        # y = data.iloc[:, -1]   # Labels (last column)
# print(y)
# Apply RandomOverSampler to balance the classes
        # ros = RandomOverSampler(random_state=42)
        # X_resampled, y_resampled = ros.fit_resample(X, y)

# Create a new balanced DataFrame
        # balanced_data = pd.concat([X_resampled, y_resampled], axis=1)

# Save the balanced data to a new CSV file
        # balanced_data.to_csv('balanced_dataset.csv', index=False)
        # df = pd.read_csv('balanced_dataset.csv', header=None)
        df = df.dropna()

        df= df.iloc[self.remove_from_start+1:-self.remove_from_end,:]
        # # print(df)
        # df_numpy = df.to_numpy()
        # df_numpy_n = df_numpy[:, :-1]

        # scaler = MinMaxScaler()
        # scaler.fit(df_numpy_n)

        # self.df = pd.DataFrame(scaler.transform(
        #     df.iloc[:, :-1]), index=df.index, columns=None)
        # self.df['labels'] = df.iloc[:, -1]
        self.df = df

    def create_output_folder(self, csv_path):

        self.folder_name = os.path.basename(csv_path)[:-4]
        self.output_folder = os.path.join('output', self.folder_name)

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def model_layers_seting(self, input,model):
        
        model.layers[0].trainable = False
        new_model = model.layers[0](input)
        for layer in model.layers[1:-3]:
            # Freeze the layers
            layer.trainable = False
            new_model = layer(new_model)
        return new_model

    def stacking_model(self, CNN_model, LSTM_model, BI_LSTM_model,
                       input_shape=(2, 20)):

        input = Input(shape=input_shape)
        CNN_model = self.model_layers_seting( input,CNN_model)
        LSTM_model = self.model_layers_seting( input,LSTM_model)
        BI_LSTM_model = self.model_layers_seting( input,BI_LSTM_model)

        x = concatenate([CNN_model, LSTM_model, BI_LSTM_model], axis=1)
        x = Dense(32, activation='relu', name = 'stacking_1',kernel_regularizer = 'L1L2',
                        activity_regularizer = 'L1L2')(x)
        x = Dense(32, activation='relu', name = 'stacking_2',kernel_regularizer = 'L1L2',
                        activity_regularizer = 'L1L2')(x)
        out = Dense(self.classes, activation=self.lat_layer_activation, name='otput_layer')(x)

        model_new = Model(inputs=input, outputs=out)
        model_new.compile(optimizer=Adam(learning_rate=self.lr),
                          loss = CategoricalCrossentropy(),
                          metrics=['accuracy'])
        return model_new
    
    
    def Resnet_stacking_model(self, CNN_model, LSTM_model, BI_LSTM_model,
                       input_shape=(2, 20)):#Radial Basis Function Networks
        
        input = Input(shape=input_shape)
        CNN_model = self.model_layers_seting( input,CNN_model)
        LSTM_model = self.model_layers_seting( input,LSTM_model)
        BI_LSTM_model = self.model_layers_seting( input,BI_LSTM_model)
        
        # x = SelfAttention(128)([CNN_model, LSTM_model, BI_LSTM_model])
        x = Flatten()(input)
        # x = RBFLayer(128,0.3)(x)
        x = Dense(128, activation='gelu',)(x)
        
        CNN_model = tf.keras.layers.Add()([CNN_model,x])
        LSTM_model = tf.keras.layers.Add()([LSTM_model,x])
        BI_LSTM_model = tf.keras.layers.Add()([BI_LSTM_model,x])
        
        x = concatenate([CNN_model, LSTM_model, BI_LSTM_model], axis=1)
        # x = Flatten()(input)
        
        
        # x = RBFLayer(128, 0.1)(x) # to look global patterns
        # x = RBFLayer(64, 0.3)(x) # to look local pattern or close relations.
        # x = RBFLayer(64, 0.5)(x) # to look local pattern or close relations.
        # x = RBFLayer(32, 0.5)(x) # to look local pattern or close relations.
        x = Dense(64, activation='relu', name = 'stacking_1',kernel_regularizer = 'L1L2',
                        activity_regularizer = 'L1L2')(x)
        x = Dense(64, activation='relu', name = 'stacking_2',kernel_regularizer = 'L1L2',
                        activity_regularizer = 'L1L2')(x)
        
        out = Dense(self.classes, activation=self.lat_layer_activation, name='otput_layer')(x)

        model_new = Model(inputs=input, outputs=out)
        # print(model_new.summary())
        model_new.compile(optimizer=Adam(learning_rate=self.lr),
                          loss = CategoricalCrossentropy(),
                          metrics=['accuracy'])
        return model_new
    

    def load_DL_models(self):
        LSTM_model = load_model(os.path.join(self.output_folder, 'LSTM.h5'))
        BI_LSTM_model = load_model(os.path.join(self.output_folder, 'GRU.h5'))
        CNN_model = load_model(os.path.join(self.output_folder, 'CNN.h5'))
        return CNN_model, LSTM_model, BI_LSTM_model

    def remove_last_layer(self,model):
        model_new = Sequential()
        for layer in model.layers[:-3]: # this is where I changed your code
            layer.trainable = False
            model_new.add(layer)
        return model_new

    def get_train_test_feautures(self,model):
        return model.predict(self.X_train,verbose = self.verbose), model.predict(self.X_test,verbose = self.verbose)


    def single_ml_train(self,  model,train_cnn_feats, test_cnn_feats,
                        name=''):

        model.fit(train_cnn_feats,np.argmax(self.Y_train,axis = 1))
        y_pred = model.predict(test_cnn_feats)
        self.predictions.append(model.predict_proba(test_cnn_feats))
        acc = self.evaluation(np.argmax(self.Y_test,axis = 1),y_pred,
                              name = name)

        # save the model to disk
        model_path = os.path.join(self.output_folder, name+'.sav')
        pickle.dump(model, open(model_path, 'wb'))

        return acc

    def get_model_features_with_FFT(self,model):
        model_new = self.remove_last_layer(model)
        train_feats, test_feats = self.get_train_test_feautures(model_new)
        return np.fft.fft(train_feats).real, np.fft.fft(test_feats).real


    def ML_classifier_train(self,model,name):

        model_new = self.remove_last_layer(model)
        train_feats, test_feats = self.get_train_test_feautures(model_new)

        LR = LogisticRegression()
        lr_acc = self.single_ml_train(LR,train_feats, test_feats,
                                      name=name+'_Logistic_regression')


        svm = SVC(kernel='linear',probability=True)
        svm_acc = self.single_ml_train(svm,train_feats, test_feats,
                                      name=name+'_svm')

        # nb = GaussianNB()
        # nb_acc = self.single_ml_train(nb,train_feats, test_feats,
        #                               name=name+'_naive_bayes')

        dt =  tree.DecisionTreeClassifier()
        dt_acc = self.single_ml_train(dt,train_feats, test_feats,
                                      name=name+'_decision_tree')

        return lr_acc, svm_acc, dt_acc

    def train_evaluate(self, name  = 'model', epochs = 0):

        # print(self.model.summary())
        print('Training:',name)
        self.train(path=os.path.join(self.output_folder, name+'.h5'))
        y_pred = self.model.predict(self.X_test,verbose = self.verbose)
        self.predictions.append(y_pred)
        acc = self.evaluation(np.argmax(self.Y_test,axis = 1),np.argmax(y_pred,axis = 1),
                              name = name)
        return acc

    def ROC_curve(self):
        y_test = np.argmax(self.Y_test,axis = 1)
        models_name = ['proposed','Stacked','CNN','LSTM','GRU',
                                      'lstm_lr', 'lstm_svm', 'lstm_dt',
                                      'cnn_lr', 'cnn_svm', 'cnn_dt',
                                      'gru_lr', 'gru_svm', 'gru_dt'

                                      ]

        colors = ['black','brown','red','orange','yellow','limegreen','lime','cyan','teal',
                  'deepskyblue','olive','navy','violet','purple','slategray']

        plt.figure(figsize=(15, 10))

        for i,probs in enumerate(self.predictions[:5]):
            # print(i,len(probs),models_name[i])
            fpr1, tpr1, thresh1 = roc_curve(y_test, probs[:,1], pos_label=1)
            plt.plot(fpr1, tpr1, label=models_name[i],color= colors[i])

        plt.legend(fontsize=15)
        plt.title('ROC curve',fontsize=26, fontweight='bold')
        plt.xlabel('False Positive Rate',fontsize=22, fontweight='bold')
        plt.ylabel('True Positive rate',fontsize=22, fontweight='bold')
        plt.xticks(fontsize=20, fontweight='bold')
        plt.yticks(fontsize=20, fontweight='bold')
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'ROC_Curve.png'))
        plt.close('all')
        # raise Exception

    def run(self, csv_path):

        self.predictions = []
        self.create_output_folder(csv_path)
        self.create_dataset(csv_path)
        groupped = self.df.groupby(self.df.columns[-1])
        # print(self.df)

        all_features = None
        all_labels = None

        for i in self.classes_names:
            i = int(i)
            data = groupped.get_group(i)
            if not len(data) % self.group_rows == 0:
                rt_ = len(data) % self.group_rows
                data = data.iloc[:-rt_, :]

            df_numpy = data.to_numpy()
            df_numpy_n = df_numpy[:, :-1]
            new_df_numpy_n = df_numpy_n.reshape(
                (df_numpy_n.shape[0]//self.group_rows, self.group_rows, self.chennels))
            # print(new_df_numpy_n.shape)
            y = df_numpy[:, -1]
            y = y.reshape((new_df_numpy_n.shape[0], self.group_rows))
            y = y[:, 1]

            if i == 0:
                all_features = new_df_numpy_n
                all_labels = y
            else:
                all_features = np.concatenate(
                    (all_features, new_df_numpy_n), axis=0)
                all_labels = np.concatenate((all_labels, y), axis=0)


        y = to_categorical(all_labels,num_classes=self.classes)
        
        

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            all_features, y, test_size=0.2)
        
        
        # x_train = np.reshape(self.X_train,(*self.X_train.shape[:-2],-1))
        # x_test = np.reshape(self.X_test,(*self.X_test.shape[:-2],-1))
        
        # from minisom import MiniSom
        # som = MiniSom(64,20, 40, sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
        # som.train(x_train, 10) # trains the SOM with 100 iterations
        # print(x_train.shape)
        # s = som.get_weights()
        # print(s.shape)
        # s= list( som.win_map(x_train).values())
        # print(len(s))
        # self.X_train = []
        # for i in s:
        #     self.X_train.append(i[0])
        # self.X_train = np.array(self.X_train)
        
        # print(sn.values())
        # for i in sn:
        #     print(sn[i])
        #     print(sn[i].tolist())
        # # print(list(self.X_train))
        # print('sss')
        # print(self.X_train)

        
        print("Data shape:", self.X_train.shape, self.Y_train.shape)

        """
        CNN + LSTM + Bi-LSTM
        """
        self.model = self.get_CNN(self.X_train.shape[1:])
        cnn_auc = self.train_evaluate( name  = 'CNN')


        self.model = self.get_LSTM(self.X_train.shape[1:])
        lstm_auc = self.train_evaluate( name  = 'LSTM')

        self.model = self.get_BI_LSTM(self.X_train.shape[1:])
        bi_lstm_auc = self.train_evaluate( name  = 'GRU')

        """
        Load all models
        """
        CNN_model, LSTM_model, BI_LSTM_model = self.load_DL_models()

        """
        train all ML models: DL_algos -> ML models
        """

        print('Training: Machine learning algorithms')
        cnn_lr_acc, cnn_svm_acc, cnn_dt_acc = self.ML_classifier_train(CNN_model,'CNN')
        lstm_lr_acc, lstm_svm_acc, lstm_dt_acc = self.ML_classifier_train(CNN_model,'LSTM')
        bi_lstm_lr_acc, bi_lstm_svm_acc, bi_lstm_dt_acc = self.ML_classifier_train(CNN_model,'GRU')

        """
        Stacking model: DL_algos -> stacking
        """
        self.model = self.stacking_model(CNN_model, LSTM_model, BI_LSTM_model,
                        input_shape=self.X_train.shape[1:])
        # plot_model(self.model,to_file = os.path.join(self.output_folder,'stacked.png'))
        stacked_auc = self.train_evaluate( name  = 'stacked_model')

        """
        Proposed method: DL_algos  -> stacking -> RBF
        """
        cnn_fft_train,cnn_fft_test = self.get_model_features_with_FFT(CNN_model)
        lstm_fft_train,lstm_fft_test = self.get_model_features_with_FFT(LSTM_model)
        bi_lstm_fft_train,bi_lstm_fft_test = self.get_model_features_with_FFT(BI_LSTM_model)


        stacked_fft_train = np.hstack((cnn_fft_train, lstm_fft_train,bi_lstm_fft_train))
        train_shape = stacked_fft_train.shape
        self.X_train = stacked_fft_train.reshape((train_shape[0],-1))


        # stacked_fft_test = np.hstack((cnn_fft_test, lstm_fft_test,bi_lstm_fft_test))
        # test_shape = stacked_fft_test.shape
        # self.X_test = stacked_fft_test.reshape((test_shape[0],-1))

        # input_ = Input(shape=self.X_train.shape[1:])
        # x = Dense(32, activation='relu', name = 'stacking_1',kernel_regularizer = 'L1L2',
        #                 activity_regularizer = 'L1L2')(input_)
        # x = Dense(32, activation='relu', name = 'stacking_2',kernel_regularizer = 'L1L2',
        #                 activity_regularizer = 'L1L2')(x)
        # # x = Dense(32, activation='relu', name = 'stacking_3')(x)
        # out = Dense(self.classes, activation=self.lat_layer_activation, name='otput_layer')(x)

        # self.model = Model(inputs=input_, outputs=out)
        # self.model.compile(optimizer=Adam(learning_rate=0.0001),
        #                   loss = CategoricalCrossentropy(),
        #                   metrics=['accuracy'])

        
        self.model = self.Resnet_stacking_model(CNN_model, LSTM_model, BI_LSTM_model,
                        input_shape=self.X_train.shape[1:])
        
        proposed_acc = self.train_evaluate(name = 'proposed')
        self.ROC_curve()

        metric = 0
        self.accuracies[self.folder_name] =  [proposed_acc[metric],stacked_auc[metric],cnn_auc[metric],lstm_auc[metric],bi_lstm_auc[metric],
                                               lstm_lr_acc[metric], lstm_svm_acc[metric], lstm_dt_acc[metric],
                                               cnn_lr_acc[metric], cnn_svm_acc[metric], cnn_dt_acc[metric],
                                               bi_lstm_lr_acc[metric], bi_lstm_svm_acc[metric], bi_lstm_dt_acc[metric]

                                              ]


        metric = 1
        self.cohen_kappa_score[self.folder_name] =  [proposed_acc[metric],stacked_auc[metric],cnn_auc[metric],lstm_auc[metric],bi_lstm_auc[metric],
                                               lstm_lr_acc[metric], lstm_svm_acc[metric], lstm_dt_acc[metric],
                                               cnn_lr_acc[metric], cnn_svm_acc[metric], cnn_dt_acc[metric],
                                               bi_lstm_lr_acc[metric], bi_lstm_svm_acc[metric], bi_lstm_dt_acc[metric]

                                              ]


        metric = 2
        self.roc_auc_score[self.folder_name] =  [proposed_acc[metric],stacked_auc[metric],cnn_auc[metric],lstm_auc[metric],bi_lstm_auc[metric],
                                               lstm_lr_acc[metric], lstm_svm_acc[metric], lstm_dt_acc[metric],
                                               cnn_lr_acc[metric], cnn_svm_acc[metric], cnn_dt_acc[metric],
                                               bi_lstm_lr_acc[metric], bi_lstm_svm_acc[metric], bi_lstm_dt_acc[metric]

                                              ]


        metric = 3
        self.f1_score[self.folder_name] = [proposed_acc[metric],stacked_auc[metric],cnn_auc[metric],lstm_auc[metric],bi_lstm_auc[metric],
                                               lstm_lr_acc[metric], lstm_svm_acc[metric], lstm_dt_acc[metric],
                                               cnn_lr_acc[metric], cnn_svm_acc[metric], cnn_dt_acc[metric],
                                               bi_lstm_lr_acc[metric], bi_lstm_svm_acc[metric], bi_lstm_dt_acc[metric]

                                              ]


        metric = 4
        self.precision_score[self.folder_name] = [proposed_acc[metric],stacked_auc[metric],cnn_auc[metric],lstm_auc[metric],bi_lstm_auc[metric],
                                               lstm_lr_acc[metric], lstm_svm_acc[metric], lstm_dt_acc[metric],
                                               cnn_lr_acc[metric], cnn_svm_acc[metric], cnn_dt_acc[metric],
                                               bi_lstm_lr_acc[metric], bi_lstm_svm_acc[metric], bi_lstm_dt_acc[metric]

                                              ]

        metric = 5
        self.recall_score[self.folder_name] =[proposed_acc[metric],stacked_auc[metric],cnn_auc[metric],lstm_auc[metric],bi_lstm_auc[metric],
                                               lstm_lr_acc[metric], lstm_svm_acc[metric], lstm_dt_acc[metric],
                                               cnn_lr_acc[metric], cnn_svm_acc[metric], cnn_dt_acc[metric],
                                               bi_lstm_lr_acc[metric], bi_lstm_svm_acc[metric], bi_lstm_dt_acc[metric]

                                              ]



m = Models_class__()
files = sorted(glob.glob(os.path.join(input_fodler, '*.csv')))[:10]

# print(files)

for file in tqdm.tqdm(files):
    print(file)
    m.run(file)


# path = os.path.join('output','subjects_avg_acc_loss')
# if not os.path.exists(path):
#     os.makedirs(path)
# for i,model in enumerate(['proposed','Stacked','CNN','LSTM','BI_LSTM']):

#     histories = []
#     for j in range(i,i+(len(m.histories)//5)):
#         # print(len(m.histories),j)
#         histories.append(m.histories[j])

#     acc =[]
#     loss = []
#     val_acc = []
#     val_loss = []

#     for hist in histories:
#         acc.append(hist.history['accuracy'])
#         val_acc.append(hist.history['val_accuracy'])
#         loss.append(hist.history['loss'])
#         val_loss.append(hist.history['val_loss'])

#     acc = np.mean(acc,axis = 0)
#     loss = np.mean(loss,axis = 0)
#     val_acc = np.mean(val_acc,axis = 0)
#     val_loss = np.mean(val_loss,axis = 0)



#     plt.figure(figsize=(20, 15))
#     # plt.figure()
#     plt.subplot(1, 2, 2)
#     plt.plot(acc)
#     plt.plot(val_acc)
#     plt.title(f'{model} Accuracy',fontsize=26, fontweight='bold')
#     plt.xlabel('Epochs',fontsize=22, fontweight='bold')
#     plt.ylabel('Accuracy',fontsize=22, fontweight='bold')
#     plt.legend(['train', 'test'], loc='upper left',fontsize=20)
#     plt.xticks(fontsize=20, fontweight='bold')
#     plt.yticks(fontsize=20, fontweight='bold')
#     plt.ylim(0.0, 1.0)
#     # colors = colors
#     plt.subplot(1, 2, 1)
#     plt.plot(loss)
#     plt.plot(val_loss)
#     plt.title(f'{model} Loss',fontsize=26, fontweight='bold')
#     plt.xlabel('Epochs',fontsize=22, fontweight='bold')
#     plt.ylabel('Loss',fontsize=22, fontweight='bold')
#     plt.legend(['train', 'test'], loc='upper left',fontsize=20)
#     plt.xticks(fontsize=20, fontweight='bold')
#     plt.yticks(fontsize=20, fontweight='bold')
#     plt.ylim(0.0, 1.0)
#     file_path = os.path.join(path,f'{model}.png')
#     plt.tight_layout()
#     plt.savefig(file_path)
#     # plt.show()
#     plt.close('all')



def avgs_std_dev(df,name = ''):
    avgs = []
    dev = []
    for i in range(len(df)):
        avg = np.round(np.mean(df.iloc[i,1:]),4)
        avgs.append(avg)
        dev_ = np.round(np.std(df.iloc[i,1:]),4)
        dev.append(dev_)
    df['Average'] = avgs
    df['standard dev'] = dev

    plt.figure(figsize=(20, 15))
    plt.bar(range(len(df['Average'])),df['Average'], yerr=df['standard dev'] , alpha=0.8,
            color = ['black','brown','red','orange','yellow','limegreen','lime','cyan','teal',
                      'deepskyblue','olive','navy','violet','purple','slategray'],
            
            align='center', ecolor='black', capsize=10)
    
    plt.plot(df['Average'],linewidth=2, markersize=12)
            # ,color = 'cyan')
    # plt.legend(fontsize=15)
    plt.title('Models '+name,fontsize=26, fontweight='bold')
    plt.xlabel('Models',fontsize=22, fontweight='bold')
    plt.ylabel(name,fontsize=22, fontweight='bold')
    plt.xticks(range(len(df['models'])),df['models'],rotation = 45,fontsize=22, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.ylim(0.0, 1.0)
    plt.gca().yaxis.grid(True)
    plt.savefig(os.path.join('output', name+'.png'))
    plt.close('all')

    return df


df = pd.DataFrame.from_dict(m.accuracies)
df = avgs_std_dev(df,name = 'accuracies')
df = m.perform_t_test(df)
df.to_csv(os.path.join('output','accuracies.csv'),index = False)


df = pd.DataFrame.from_dict(m.cohen_kappa_score)
df = avgs_std_dev(df,name = 'cohen_kappa_score')
df = m.perform_t_test(df)
df.to_csv(os.path.join('output','cohen_kappa_score.csv'),index = False)


df = pd.DataFrame.from_dict(m.roc_auc_score)
df = avgs_std_dev(df,name = 'roc_auc_score')
df = m.perform_t_test(df)
df.to_csv(os.path.join('output','roc_auc_score.csv'),index = False)


df = pd.DataFrame.from_dict(m.f1_score)
df = avgs_std_dev(df,name = 'f1_score')
df = m.perform_t_test(df)
df.to_csv(os.path.join('output','f1_score.csv'),index = False)


df = pd.DataFrame.from_dict(m.precision_score)
df = avgs_std_dev(df,name = 'precision_score')
df = m.perform_t_test(df)
df.to_csv(os.path.join('output','precision_score.csv'),index = False)


df = pd.DataFrame.from_dict(m.recall_score)
df = avgs_std_dev(df,name = 'recall_score')
df = m.perform_t_test(df)
df.to_csv(os.path.join('output','recall_score.csv'),index = False)

# !zip -r /content/output.zip /content/output