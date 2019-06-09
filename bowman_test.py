from bowman_lstm import TorchRNNSentenceEncoderClassifier
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score, f1_score
import nli
import os
import torch
import json
import re
import utils


DATA_HOME = os.path.join("data", "nlidata")
SNLI_HOME = os.path.join(DATA_HOME, "snli_1.0")

def sentence_encoding_rnn_phi(t1, t2):
    """Map `t1` and `t2` to a pair of lits of leaf nodes."""
    return (t1.leaves(), t2.leaves())

def get_sentence_encoding_vocab(X, n_words=None):    
    wc = Counter([w for pair in X for ex in pair for w in ex])
    wc = wc.most_common(n_words) if n_words else wc.items()
    vocab = {w for w, c in wc}
    vocab.add("$UNK")
    return sorted(vocab)

def fit_sentence_encoding_rnn(X, y):   
    vocab = get_sentence_encoding_vocab(X, n_words=10000)
    mod = TorchRNNSentenceEncoderClassifier(
        vocab, hidden_dim=100, max_iter=100)

    #if torch.cuda.is_available():
    #    mod.cuda()

    mod.fit(X, y)

    return mod


def build_dataset(file):
    
    X = []
    y = []

    with open(file) as f:
        while True:
            line = f.readline()
            if not line:
                break

            json_line = json.loads(line)

            premise = json_line['sentence1']
            hypothesis = json_line['sentence2']
            label = json_line['gold_label']

            if label not in ['contradiction', 'neutral', 'entailment']:
                continue

            premise_parsed = re.findall(r"[\w']+|[.,!?;]", premise)
            hypothesis_parsed = re.findall(r"[\w']+|[.,!?;]", hypothesis)

            X.append((premise_parsed, hypothesis_parsed))
            y.append(label)

        f.close()

    return {
        'X': X,
        'y': y
    }


def train():
    # 1. Build the dataset
    print('Building dataset...')
    file = 'data/nlidata/snli_1.0/snli_1.0_train.jsonl'
    dataset = build_dataset(file)

    X_train = dataset['X']
    y_train = dataset['y']
    
    # 2. Train the model
    print('Initiating model training...')
    mod = fit_sentence_encoding_rnn(X_train, y_train)

    # 3. Save model
    print('Saving model...')
    torch.save(mod, 'LSTM_model_withTahH_100D_100iter_full.pt')


#train()


def test():
    # 1. Build the test dataset
    print('Building dataset...')
    test_file = 'data/nlidata/snli_1.0/breaking_dataset.jsonl'
    dataset = build_dataset(test_file)

    X_test = dataset['X']
    y_test = dataset['y']

    # 2. Make model predictions
    print('Starting predictions...')
    mod = torch.load('LSTM_model_withTahH_100D_100iter_full.pt')
    predictions = mod.predict(X_test)

    # 3. Compute accuracy
    print('Computing accuracy...')
    print(classification_report(y_test, predictions, digits=3))

test()
