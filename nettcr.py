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
from tensorflow.keras.optimizers import Adam
from keras.initializers import glorot_normal
from keras.activations import sigmoid
from sklearn.metrics import roc_auc_score
import utils
import keras.backend as K
from keras.callbacks import EarlyStopping

from nettcr_architectures import nettcr_ab, nettcr_one_chain 

#Options for Pandas DataFrame printing
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', -1)

from argparse import ArgumentParser

#Args parser 
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-tr", "--trainfile", help="Specify the full path of the training file with TCR sequences")
parser.add_argument("-te", "--testfile", help="Specify the full path of the file with TCR sequences")
parser.add_argument("-c", "--chain", default="ab", help="Specify the chain(s) to use (a, b, ab). Default: ab")
parser.add_argument("-o", "--outfile", default=sys.stdout, help="Specify output file")
parser.add_argument("-e", "--epochs", default=100, type=int, help="Specify the number of epochs")
args = parser.parse_args()

EPOCHS = int(args.epochs)
chain = args.chain
if chain not in ["a","b","ab"]:
    print("Invalid chain. You can select a (alpha), b (beta), ab (alpha+beta)")

print('Loading and encoding the data..')
train_data = pd.read_csv(args.trainfile)
test_data = pd.read_csv(args.testfile)


# Encode data
encoding = utils.blosum50_20aa
early_stop = EarlyStopping(monitor='loss',min_delta=0,
               patience=10, verbose=0,mode='min',restore_best_weights=True)


# Call and compile the model
if chain=='ab':
    pep_train = utils.enc_list_bl_max_len(train_data.peptide, encoding, 9)
    tcra_train = utils.enc_list_bl_max_len(train_data.CDR3a, encoding, 30)
    tcrb_train = utils.enc_list_bl_max_len(train_data.CDR3b, encoding, 30)
    y_train = np.array(train_data.binder)

    pep_test = utils.enc_list_bl_max_len(test_data.peptide, encoding, 9)
    tcra_test = utils.enc_list_bl_max_len(test_data.CDR3a, encoding, 30)
    tcrb_test = utils.enc_list_bl_max_len(test_data.CDR3b, encoding, 30)
    train_inputs = [tcra_train, tcrb_train, pep_train]
    test_inputs = [tcra_test, tcrb_test, pep_test]

    mdl = nettcr_ab()
elif chain=="a":
    pep_train = utils.enc_list_bl_max_len(train_data.peptide, encoding, 9)
    tcra_train = utils.enc_list_bl_max_len(train_data.CDR3a, encoding, 30)
    y_train = np.array(train_data.binder)

    pep_test = utils.enc_list_bl_max_len(test_data.peptide, encoding, 9)
    tcra_test = utils.enc_list_bl_max_len(test_data.CDR3a, encoding, 30)
    train_inputs = [tcra_train, pep_train]
    test_inputs = [tcra_test, pep_test]
    mdl = nettcr_one_chain()
elif chain=="b":
    pep_train = utils.enc_list_bl_max_len(train_data.peptide, encoding, 9)
    tcrb_train = utils.enc_list_bl_max_len(train_data.CDR3b, encoding, 30)
    y_train = np.array(train_data.binder)

    pep_test = utils.enc_list_bl_max_len(test_data.peptide, encoding, 9)
    tcrb_test = utils.enc_list_bl_max_len(test_data.CDR3b, encoding, 30)
    train_inputs = [tcrb_train, pep_train]
    test_inputs = [tcrb_test, pep_test]
    mdl = nettcr_one_chain()


mdl.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001))
print('Training..')
# Train
history = mdl.fit(train_inputs, y_train, 
                  epochs=EPOCHS, batch_size=128, verbose=1, callbacks=[early_stop])

print('Evaluating..')
# Predict on test data
preds = mdl.predict(test_inputs, verbose=0)
pred_df = pd.concat([test_data, pd.Series(np.ravel(preds), name='prediction')], axis=1)

pred_df.to_csv(args.outfile, index=False)