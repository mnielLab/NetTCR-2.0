import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import time
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, concatenate
from keras.optimizers import Adam
from keras.initializers import glorot_normal
from keras.activations import sigmoid
from sklearn.metrics import roc_auc_score
import utils
import keras.backend as K
from keras.callbacks import EarlyStopping

#Options for Pandas DataFrame printing
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', -1)

from argparse import ArgumentParser

#Args parser 
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-tr", "--trainfile", help="Specify input training file with TCR sequences")
parser.add_argument("-te", "--testfile", help="Specify input testing file with TCR sequences")
parser.add_argument("-o", "--outfile", default=sys.stdout, help="Specify output file")
parser.add_argument("-e", "--epochs", default=100, type=int, help="Specify the number of epochs")
args = parser.parse_args()

EPOCHS = int(args.epochs)

print('Loading and encoding the data..')
train_data = pd.read_csv(args.trainfile)
test_data = pd.read_csv(args.testfile)


# Encode data
encoding = utils.blosum50_20aa

pep_train = utils.enc_list_bl_max_len(train_data.peptide, encoding, 9)
tcra_train = utils.enc_list_bl_max_len(train_data.CDR3a, encoding, 30)
tcrb_train = utils.enc_list_bl_max_len(train_data.CDR3b, encoding, 30)
y_train = np.array(train_data.binder)

pep_test = utils.enc_list_bl_max_len(test_data.peptide, encoding, 9)
tcra_test = utils.enc_list_bl_max_len(test_data.CDR3a, encoding, 30)
tcrb_test = utils.enc_list_bl_max_len(test_data.CDR3b, encoding, 30)

# Network architecture
def nettcr_ab():
   
    pep_in = Input(shape=(9,20))
    cdra_in = Input(shape=(30,20))
    cdrb_in = Input(shape=(30,20))
       
    pep_conv1 = Conv1D(16, 1, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool1 = GlobalMaxPooling1D()(pep_conv1)
    pep_conv3 = Conv1D(16, 3, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool3 = GlobalMaxPooling1D()(pep_conv3)
    pep_conv5 = Conv1D(16, 5, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool5 = GlobalMaxPooling1D()(pep_conv5)
    pep_conv7 = Conv1D(16, 7, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool7 = GlobalMaxPooling1D()(pep_conv7)
    pep_conv9 = Conv1D(16, 9, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(pep_in)
    pep_pool9 = GlobalMaxPooling1D()(pep_conv9)
    
    cdra_conv1 = Conv1D(16, 1, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdra_in)
    cdra_pool1 = GlobalMaxPooling1D()(cdra_conv1)
    cdra_conv3 = Conv1D(16, 3, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdra_in)
    cdra_pool3 = GlobalMaxPooling1D()(cdra_conv3)
    cdra_conv5 = Conv1D(16, 5, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdra_in)
    cdra_pool5 = GlobalMaxPooling1D()(cdra_conv5)
    cdra_conv7 = Conv1D(16, 7, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdra_in)
    cdra_pool7 = GlobalMaxPooling1D()(cdra_conv7)
    cdra_conv9 = Conv1D(16, 9, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdra_in)
    cdra_pool9 = GlobalMaxPooling1D()(cdra_conv9)
    
    cdrb_conv1 = Conv1D(16, 1, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool1 = GlobalMaxPooling1D()(cdrb_conv1)
    cdrb_conv3 = Conv1D(16, 3, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool3 = GlobalMaxPooling1D()(cdrb_conv3)
    cdrb_conv5 = Conv1D(16, 5, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool5 = GlobalMaxPooling1D()(cdrb_conv5)
    cdrb_conv7 = Conv1D(16, 7, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool7 = GlobalMaxPooling1D()(cdrb_conv7)
    cdrb_conv9 = Conv1D(16, 9, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdrb_in)
    cdrb_pool9 = GlobalMaxPooling1D()(cdrb_conv9)
    

    pep_cat = concatenate([pep_pool1, pep_pool3, pep_pool5, pep_pool7, pep_pool9])
    cdra_cat = concatenate([cdra_pool1, cdra_pool3, cdra_pool5, cdra_pool7, cdra_pool9])
    cdrb_cat = concatenate([cdrb_pool1, cdrb_pool3, cdrb_pool5, cdrb_pool7, cdrb_pool9])

    
    cat = concatenate([pep_cat, cdra_cat, cdrb_cat], axis=1)
    
    dense = Dense(32, activation='sigmoid')(cat)
        
    out = Dense(1, activation='sigmoid')(dense)
    
    model = (Model(inputs=[cdra_in, cdrb_in, pep_in],outputs=[out]))
    
    return model


early_stop = EarlyStopping(monitor='loss',min_delta=0,
               patience=10, verbose=0,mode='min',restore_best_weights=True)


# Call and compile the model
mdl = nettcr_ab()
mdl.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001))
print('Training..')
# Train
history = mdl.fit([tcra_train, tcrb_train, pep_train], y_train, 
                  epochs=EPOCHS, batch_size=128, verbose=1, callbacks=[early_stop])
print('Evaluating..')
# Predict on test data
preds = mdl.predict([tcra_test, tcrb_test, pep_test], verbose=0)
pred_df = pd.concat([test_data.CDR3a, test_data.CDR3b, test_data.peptide, pd.Series(np.ravel(preds), name='prediction')], axis=1)

pred_df.to_csv(args.outfile, index=False)