from werkzeug.wrappers import Request, Response
from werkzeug.utils import secure_filename
import os

import tensorflow as tf
from tensorflow import keras
from gtts import gTTS

from flask import Flask, jsonify, request
import numpy as np
import joblib
import pandas as pd
import numpy as np
from sklearn import linear_model

from bs4 import BeautifulSoup
import re
from pickle import load
from flask import Flask, request, jsonify, render_template


from sklearn.feature_extraction.text import CountVectorizer
import os
# os.chdir('Flickr8k')



import string
from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

from tensorflow.compat.v1.keras.backend import set_session

import sys, time, os, warnings 
warnings.filterwarnings("ignore")
import re

import numpy as np
import pandas as pd 
# from PIL import Image
import pickle
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import add
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.utils import shuffle
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import Input, layers

# tf.compat.v1.enable_eager_execution()


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


from flask_caching import Cache
# import predict

app = Flask(__name__) 
i=1
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0.002

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/')
def index():
    return render_template('test.html')



class Attention_model(Model):
    def __init__(self, units):
        super(Attention_model, self).__init__()
        self.W1 = layers.Dense(units)# weights to multiply with feature vector 
        self.W2 = layers.Dense(units)#weights to multiply with hidden state of t-1
        self.V = layers.Dense(1)#to convert score vector to scalar
        self.units=units

    def call(self, features, hidden):
        #features shape: (batch_size, 8*8, embedding_dim) ==>(64,64,256) :output from encoder
        # hidden shape: (batch_size, hidden_size) ==>(64,512)
        hidden_with_time_axis =   tf.expand_dims(hidden, 1)# Expand the hidden shape to shape: (batch_size, 1, hidden_size) to match dimension
        score = tf.keras.activations.tanh(self.W1(features) + self.W2(hidden_with_time_axis))# build your score funciton to shape: (batch_size, 8*8, units)
   # self.W1(features) : (64,64,512)
   # self.W2(hidden_with_time_axis) : (64,1,512)
   # tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis)) : (64,64,512)
   # self.Vattn(tf.nn.tanh(self.Uattn(features) + self.Wattn(hidden_with_time_axis))) : (64,64,1) ==> score

   # you get 1 at the last axis because you are applying score to self.Vattn
   # Then find Probability using Softmax
        
        attention_weights =  tf.keras.activations.softmax(self.V(score), axis=1)# extract your attention weights with shape: (batch_size, 8*8, 1)
        context_vector = attention_weights * features #shape: create the context vector with shape (batch_size, 8*8,embedding_dim)
        context_vector =tf.reduce_sum(context_vector, axis=1) # reduce the shape to (batch_size, embedding_dim)
   # Context Vector(64,256) = AttentionWeights(64,64,1) * features(64,64,256)
   # context_vector shape after sum == (64, 256)  # we will need to expand dims by 1 to be able to concatenate with input token vector

        return context_vector, attention_weights

class Decoder(Model):
    def __init__(self, embed_dim, units, vocab_size):
        super(Decoder, self).__init__()
        self.units=units
        self.attention = Attention_model(self.units)#iniitalise your Attention model with units
        self.embed = tf.keras.layers.Embedding(vocab_size, embed_dim,mask_zero=False)#build your Embedding layer
        self.gru = tf.keras.layers.GRU(self.units,return_sequences=True,return_state=True,recurrent_initializer='glorot_uniform')
        self.d1 = tf.keras.layers.Dense(self.units) #build your Dense layer
        self.d2 = tf.keras.layers.Dense(vocab_size)   #build your Dense layer
        self.dropout = Dropout(0.5)
        

    def call(self,x,features, hidden):
        context_vector, attention_weights =self.attention(features, hidden) #create your context vector & attention weights from attention model
        embed =  self.dropout(self.embed(x))  # embed your input to shape: (batch_size, 1, embedding_dim)
        mask =  self.embed.compute_mask(x)
        embed =  tf.concat([tf.expand_dims(context_vector, 1), embed], axis=-1)# x shape after concatenation == (64, 1,  512)
        output,state = self.gru(embed,mask=mask)# Extract the output & hidden state from GRU layer. Output shape : (batch_size, max_length, hidden_size)
        output = self.d1(output)
        output = tf.reshape(output, (-1, output.shape[2])) # shape : (batch_size * max_length, hidden_size)
        output = self.d2(output) # shape : (batch_size * max_length, vocab_size)
        
        return output,state, attention_weights
    
    
    
    def init_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
    
    def model(self):    # this is used to create the model plot
        x1=Input(shape=(1))
        x2=Input(shape=(64,256))
        x3=Input(shape=(512))
        
        return Model(inputs=[x1,x2,x3],outputs=self.call(x1,x2,x3))       

@app.route('/predict', methods=['POST'])
def predict():
    resp=Response()
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    checkpoint_path = "submit_checkpoint/ckpt-1"
    train_captions = load(open('captions.pkl', 'rb'))

    
    def tokenize_caption(top_k,train_captions):
        # Choosing the top k words from vocabulary
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,oov_token="<unk>",filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
        # oov_token: if given, it will be added to word_index 
        # and used to replace 
        # out-of-vocabulary words during text_to_sequence calls
        
        tokenizer.fit_on_texts(train_captions)
        train_seqs = tokenizer.texts_to_sequences(train_captions)

        # Map '<pad>' to '0'
        tokenizer.word_index['<pad>'] = 0
        tokenizer.index_word[0] = '<pad>'

        # Create the tokenized vectors
        train_seqs = tokenizer.texts_to_sequences(train_captions)
        return train_seqs, tokenizer

    top_k = 5000
    train_seqs , tokenizer = tokenize_caption(top_k ,train_captions)



    def calc_max_length(tensor):
        return max(len(t) for t in tensor)
    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(train_seqs)

    # Find the minimum length of any caption in our dataset
    def calc_min_length(tensor):
        return min(len(t) for t in tensor)
    # Calculates the max_length, which is used to store the attention weights
    min_length = calc_min_length(train_seqs)


    
    embedding_dim = 256
    units = 512
    vocab_size = top_k + 1 
    
    decoder = Decoder(embedding_dim,units,vocab_size)

    class VGG16_Encoder(tf.keras.Model):
        # This encoder passes the features through a Fully connected layer
        def __init__(self, embedding_dim):
            super(VGG16_Encoder, self).__init__()
            # shape after fc == (batch_size, 49, embedding_dim)
            self.fc = tf.keras.layers.Dense(embedding_dim)
            self.dropout = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None)

        def call(self, x):
            #x= self.dropout(x)
            x = self.fc(x)
            x = tf.nn.relu(x)
            return x

    encoder = VGG16_Encoder(embedding_dim)
    optimizer = tf.keras.optimizers.Adam()
    ckpt = tf.train.Checkpoint(encoder=encoder,decoder=decoder,optimizer = optimizer)
    ckpt.restore(checkpoint_path)

    # to_predict_list = request.form.to_dict()
    # Image_path = to_predict_list['pic_url']
    f= request.files['pic']
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    try:
        os.remove(os.path.join(THIS_FOLDER,'static','css','temp5.jpg'))
    except:
        print("No file")
    f.save(os.path.join(THIS_FOLDER,'static','css',secure_filename("temp5.jpg")))
    Image_path =  (os.path.join(THIS_FOLDER,'static','css','temp5.jpg'))
    path2 = "./static/css/{}".format(f.filename) + ".mp3"

    def load_image(filename):
        image_path=filename
        img = tf.io.read_file(image_path)
        print(img)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = preprocess_input(img)
        # print('hello',img.numpy())
        return img, image_path
        
    print("Hi",Image_path)
    attention_features_shape = 64
    image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
    new_input = image_model.input # Any arbitrary shapes with 3 channels
    hidden_layer = image_model.layers[-1].output
    feat_extrac_model = tf.keras.Model(new_input, hidden_layer)
    
    def evaluate(image):
        attn_plt = np.zeros((max_length, attention_features_shape))

        hidden = decoder.init_state(batch_size=1)

        temp_input = tf.expand_dims(load_image(image)[0], 0)
        img_tensor_val = feat_extrac_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

        features = encoder(img_tensor_val)

        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(max_length):
            predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

            attn_plt[i] = tf.reshape(attention_weights, (-1, )).numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(tokenizer.index_word[predicted_id])

            if tokenizer.index_word[predicted_id] == '<end>':
                return result, attn_plt

            dec_input = tf.expand_dims([predicted_id], 0)

        attn_plt = attn_plt[:len(result), :]
        return result, attn_plt


    new_img =  Image_path
    result, attention_plot = evaluate(new_img)
   
    for i in result:
        if i=="<unk>":
            result.remove(i)
        else:
            pass
    
    output = gTTS(text = str(result[:-1]), lang = 'en',slow = False)
    output.save(path2)
    captn =' '
    return render_template('next.html', prediction_text=(captn.join(result).rsplit(' ', 1)[0]),ttt=os.path.join(THIS_FOLDER,'static','css','temp5.jpg'),cap=path2)



if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
