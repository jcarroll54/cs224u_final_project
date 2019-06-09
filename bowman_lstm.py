import torch
import torch.nn as nn
import torch.utils.data
from torch_rnn_classifier import TorchRNNClassifier, TorchRNNClassifierModel
from torch_shallow_neural_classifier import TorchShallowNeuralClassifier
from torch_rnn_classifier import TorchRNNClassifier

class TorchRNNSentenceEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, seq_lengths, y):
        self.prem_seqs, self.hyp_seqs = sequences
        self.prem_lengths, self.hyp_lengths = seq_lengths
        self.y = y
        assert len(self.prem_seqs) == len(self.y)

    @staticmethod
    def collate_fn(batch):
        X_prem, X_hyp, prem_lengths, hyp_lengths, y = zip(*batch)
        prem_lengths = torch.LongTensor(prem_lengths)
        hyp_lengths = torch.LongTensor(hyp_lengths)
        y = torch.LongTensor(y)

        return (X_prem, X_hyp), (prem_lengths, hyp_lengths), y

    def __len__(self):
        return len(self.prem_seqs)

    def __getitem__(self, idx):
        return (self.prem_seqs[idx], self.hyp_seqs[idx],
                self.prem_lengths[idx], self.hyp_lengths[idx],
                self.y[idx])

class TorchRNNSentenceEncoderClassifierModel(TorchRNNClassifierModel):
    def __init__(self, vocab_size, embed_dim, embedding, use_embedding,
            hidden_dim, output_dim, bidirectional, device):
        super(TorchRNNSentenceEncoderClassifierModel, self).__init__(
            vocab_size, embed_dim, embedding, use_embedding,
            hidden_dim, output_dim, bidirectional, device)
        self.hypothesis_rnn = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=self.bidirectional)
	self.premise_rnn = nn.LSTM(
	    input_size=self.embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=self.bidirectional)
        if bidirectional:
            classifier_dim = hidden_dim * 2 * 2
        else:
            classifier_dim = hidden_dim * 2
        self.classifier_layer = nn.Sequential(nn.Tanh(), nn.Tanh(), nn.Tanh(), nn.Linear(classifier_dim, output_dim))
    

    def forward(self, X, seq_lengths):
        X_prem, X_hyp = X
        prem_lengths, hyp_lengths = seq_lengths
        prem_state = self.rnn_forward(X_prem, prem_lengths, self.rnn)
        hyp_state = self.rnn_forward(X_hyp, hyp_lengths, self.hypothesis_rnn)
        state = torch.cat((prem_state, hyp_state), dim=1)
        logits = self.classifier_layer(state)
        return logits


class TorchRNNSentenceEncoderClassifier(TorchRNNClassifier):

    def build_dataset(self, X, y):
        X_prem, X_hyp = zip(*X)
        X_prem, prem_lengths = self._prepare_dataset(X_prem)
        X_hyp, hyp_lengths = self._prepare_dataset(X_hyp)
        return TorchRNNSentenceEncoderDataset(
            (X_prem, X_hyp), (prem_lengths, hyp_lengths), y)

    def build_graph(self):
        return TorchRNNSentenceEncoderClassifierModel(
            len(self.vocab),
            embedding=self.embedding,
            embed_dim=self.embed_dim,
            use_embedding=self.use_embedding,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_classes_,
            bidirectional=self.bidirectional,
            device=self.device)

    def predict_proba(self, X):
        with torch.no_grad():
            X_prem, X_hyp = zip(*X)
            X_prem, prem_lengths = self._prepare_dataset(X_prem)
            X_hyp, hyp_lengths = self._prepare_dataset(X_hyp)
            preds = self.model((X_prem, X_hyp), (prem_lengths, hyp_lengths))
            preds = torch.softmax(preds, dim=1).cpu().numpy()
            return preds
