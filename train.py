import numpy as np
import os
from util import load_data_and_labels_from_csv_file, build_vocab, pad_sentences, text_to_sequence, save_vocab_json, generate_word_level_features, customCallBack
import keras
from keras.layers import Embedding, Dropout, LSTM, Dense
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.optimizers import Adam
from keras.models import Sequential
import requests # This library is used to make requests to internet
import zipfile

data_file = "data/SMSSpamCollection"

#Download and unzip the data file in data directory in case it doesn't exists already
if not os.path.exists(data_file):
    data_file_dir = os.path.dirname(data_file)
    if not os.path.exists(data_file_dir): os.makedirs(data_file_dir)

    # We are storing url of dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(url, allow_redirects=True)
    zip_file_download = data_file_dir + '/smsspamcollection.zip'

    # We are writing the content of above request to 'iris.data' file
    open(zip_file_download, 'wb').write(r.content)
    
    #Extract the zip file
    with zipfile.ZipFile(zip_file_download,"r") as zip_ref:
        zip_ref.extractall(data_file_dir)

# Load data
print("Loading data...")
labels, sentences = load_data_and_labels_from_csv_file(data_file)

params = {'max_words_features': 50} 

lines_words_level_features = generate_word_level_features(sentences, params['max_words_features'])
params['max_words_features'] = max([len(lines) for lines in lines_words_level_features])

lines_words_level_features = np.array(lines_words_level_features)

# Build vocabulary
print("Build the vocabulary")
vocabulary = build_vocab(lines_words_level_features, max_vocab_size=10000)
#print(vocabulary)

# Pad sentence
print("Padding sentences...")
x_text = pad_sentences(lines_words_level_features, max_sequence_length=params['max_words_features'])

seq_len = len(x_text[0]) 
print("The sequence length is: ", seq_len)

# Represent sentence with word index, using word index to represent a sentence
x = text_to_sequence(x_text, vocabulary)

# Shuffle data
#np.random.seed(1) #same shuffling each time
shuffle_indices = np.random.permutation(np.arange(len(labels)))
x = x[shuffle_indices]
labels = labels[shuffle_indices]

"""
## Build LSTM model
"""
vocab_size_or_total_features = len(vocabulary) 

# this returns a tensor
print("Creating Model...")

embed_dim = 128
embedding_dropout_factor = 0.4
recurrent_dropout_factor = 0.2
LSTM_dropout_factor = 0.2
layer_dropout_factor = 0.2
LSTM_layer_sizes = [512]

model = Sequential()
model.add(Embedding(vocab_size_or_total_features, embed_dim, input_length=seq_len))
model.add(Dropout(embedding_dropout_factor))
for size in LSTM_layer_sizes[:-1]:
    model.add(LSTM(units=size, return_sequences=True,
                       recurrent_dropout=recurrent_dropout_factor,
                       dropout=LSTM_dropout_factor))
    model.add(Dropout(layer_dropout_factor))
model.add(LSTM(units=LSTM_layer_sizes[-1], recurrent_dropout=recurrent_dropout_factor, dropout=LSTM_dropout_factor))
model.add(Dropout(layer_dropout_factor))
model.add(Dense(1, activation='sigmoid'))

checkpoint_path = "model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
# Save Vocabulary
vocab_file = checkpoint_dir + "/vocab.json"
save_vocab_json(vocab_file, vocabulary, params)

#checkpoint = ModelCheckpoint(filepath=checkpoint_path,  monitor='accuracy', verbose=1, save_best_only=True, mode='auto') # Create callback to save the weights
checkpoint = ModelCheckpoint(filepath=checkpoint_path,  monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto') # Create callback to save the weights
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=checkpoint_dir, histogram_freq=0)
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

epochs = 50
batch_size = 32
verbose = 1
validation_split = 0.2
max_patience = 20

print("Traning Model...")
callbacks=[customCallBack(model, checkpoint_path, max_patience=max_patience), tensorboard_callback]
model.fit(x, labels, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=validation_split, callbacks=callbacks)

