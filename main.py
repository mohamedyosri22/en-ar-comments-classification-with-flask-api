from langdetect import detect, DetectorFactory
import pickle
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import re, string
import numpy as np
import sklearn
from werkzeug.serving import WSGIRequestHandler
from flask import Flask, request, render_template, jsonify

DetectorFactory.seed = 0

app = Flask(__name__)

# tokenize function for tfidf

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()


def pred(userinp):
    # pretrained arabic model
    new_model_ar = tf.keras.models.load_model('trained_model/ar-comments')

    with open('trained_model/le.pickle', 'rb') as enc:
        le = pickle.load(enc)

    with open('trained_model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # pretrained english model

    with open('trained_model/NB_model.pickle', 'rb') as model:
        new_model_en = pickle.load(model)

    with open('trained_model/vec.pickle', 'rb') as handle:
        new_vec = pickle.load(handle)

        inp = userinp
        lang = detect(inp)

        if lang == 'ar':

            inp = [inp]

            inp = tokenizer.texts_to_sequences(inp)

            inp = pad_sequences(inp, maxlen=150, dtype='int32', value=0)

            sentiment = new_model_ar.predict(inp, batch_size=1, verbose=2)[0]
            result = np.argmax(sentiment)

            if result == 2:
                ret = 'normal'
            else:
                ret = 'abusive'

        else:
            inp = inp.lower()
            inp = [inp]
            inp = new_vec.transform(inp)

            pred = new_model_en.predict(inp)

            if (pred == 1):
                ret = 'abusive'
            else:
                ret = 'normal'
    return ret


@app.route('/', methods=['POST'])
def predict():
    userinp = request.json
    userinp = userinp['data']
    comment_prediction = pred(userinp)

    return jsonify({"response" : "comment is "+comment_prediction})


@app.route("/")
def ind():
    return "<h1> Comments classification api </h1>"


if __name__ == '__main__':
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(port=3000, debug=True)
