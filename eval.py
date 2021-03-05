import numpy as np
import keras
from util import load_vocab_json, pad_sentences, text_to_sequence, generate_word_level_features

#load model
model = keras.models.load_model("model/cp.ckpt/")
#print(model.summary())

vocab_file = "model/vocab.json"
vocabulary, params = load_vocab_json(vocab_file)

x_text = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&     C's apply 08452810075over18's", "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."]

#labels, sentences = get_data_and_labels(lines)

lines_words_level_features = generate_word_level_features(x_text, params['max_words_features'])
lines_words_level_features = np.array(lines_words_level_features)

seq_len = params['max_words_features'] 

x = pad_sentences(lines_words_level_features, max_sequence_length=seq_len, is_max_sequence_length_modifiable=False)
x = text_to_sequence(x, vocabulary)


print("Generate predictions")
predictions = model.predict(x)
count = 0
for text in x_text:
    print("Text is: \t", text)
    if (predictions[count] > 0.5):
        print("predicted spam with spam prob ", predictions[count])
    else:
        print("predicted ham with spam prob ", predictions[count])
    count += 1 
