import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv1D, Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers import concatenate, Dropout, Activation
from keras.initializers import glorot_normal
from keras.activations import sigmoid


def nettcr_ab():
    '''NetTCR ab with the "correct" pooling dimension, that is: the three towers are convoluted, then pooled
       and then concatenated for the FNN'''
    
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
    
    #drop = Dropout(0.3)(dense)
    
    out = Dense(1, activation='sigmoid')(dense)
    
    model = (Model(inputs=[cdra_in, cdrb_in, pep_in],outputs=[out]))
    
    return model

def nettcr_one_chain():
    cdr_in = Input(shape=(30,20))
    pep_in = Input(shape=(9,20))
    
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
    
    cdr_conv1 = Conv1D(16, 1, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdr_in)
    cdr_pool1 = GlobalMaxPooling1D()(cdr_conv1)
    cdr_conv3 = Conv1D(16, 3, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdr_in)
    cdr_pool3 = GlobalMaxPooling1D()(cdr_conv3)
    cdr_conv5 = Conv1D(16, 5, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdr_in)
    cdr_pool5 = GlobalMaxPooling1D()(cdr_conv5)
    cdr_conv7 = Conv1D(16, 7, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdr_in)
    cdr_pool7 = GlobalMaxPooling1D()(cdr_conv7)
    cdr_conv9 = Conv1D(16, 9, padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(cdr_in)
    cdr_pool9 = GlobalMaxPooling1D()(cdr_conv9)
    
    pep_cat = concatenate([pep_pool1, pep_pool3, pep_pool5, pep_pool7, pep_pool9])
    cdr_cat = concatenate([cdr_pool1, cdr_pool3, cdr_pool5, cdr_pool7, cdr_pool9])
    
    cat = concatenate([pep_cat, cdr_cat], axis=1)
    
    dense = Dense(32, activation='sigmoid')(cat)
    
    #drop = Dropout(0.3)(dense)
    
    out = Dense(1, activation='sigmoid')(dense)
    
    model = (Model(inputs=[cdr_in, pep_in],outputs=[out]))
    
    return model