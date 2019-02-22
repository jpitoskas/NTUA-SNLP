import torch
import numpy as np
from torch import nn

from utils.SelfAttention import SelfAttention


class PreLabBaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    @staticmethod
    def mean_pooling(x, lengths):
        sums = torch.sum(x, dim=1).float()
        if (torch.cuda.is_available()):
            sums = sums.cuda()
        _lens = lengths.view(-1, 1).expand(sums.size(0), sums.size(1))
        means = sums / _lens.float()
        return means

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(PreLabBaselineDNN, self).__init__()

        self.emb_dim = embeddings.shape[1]
        # 1 - define the embedding layer
        # EX4
        # embed = nn.Embedding(10, 3)

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        # EX4

        # 3 - define if the embedding layer will be frozen or finetuned
        # EX4
        # ALL 3 TOGETHER
        # self.embed = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=not(trainable_emb), sparse=False)
        self.embed = nn.Embedding(num_embeddings=embeddings.shape[0], embedding_dim=embeddings.shape[1])
        self.embed.weight = nn.Parameter(torch.from_numpy(embeddings), requires_grad=trainable_emb)
        # 4 - define a non-linear transformation of the representations
        # EX5

        self.tanh = nn.Tanh()

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        # EX5

        self.final = nn.Linear(self.emb_dim, output_size)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        self.batch_size = lengths[0]
        self.l = lengths[1]
        self.maxlen = x.shape[1]

        # 1 - embed the words, using the embedding layer
        # EX6

        # batch_size, 21, emb_dim
        # embeddi = torch.zeros([self.batch_size, self.maxlen, self.emb_dim])
        # for i in range(self.batch_size):
        #     e = x[i].long()
        #     embeddi[i] = self.embed(e)  # EX6
        embedding = self.embed(x.long())
        if (torch.cuda.is_available()):
            embedding = embedding.cuda()

        # 2 - construct a sentence representation out of the word embeddings
        # EX6

        # representations = np.zeros((self.batch_size, self.emb_dim))
        # for i in range(self.batch_size):
        #     for j in range(self.emb_dim):
        #         rep_sum = 0
        #         length = self.l[i]
        #         N = min(self.maxlen, length)
        #         for k in range(N):
        #             rep_sum += embeddings[i][k][j]
        #         rep_sum /= length
        #         representations[i][j] = rep_sum

        representations = self.mean_pooling(embedding, self.l)

        # 3 - transform the representations to new ones.
        # EX6
        representations = self.tanh(representations)

        # 4 - project the representations to classes using a linear layer
        # EX6
        logits = self.final(representations)

        return logits

class MeanMaxDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    @staticmethod
    def mean_pooling(x, lengths):
        sums = torch.sum(x, dim=1).float()
        if (torch.cuda.is_available()):
            sums = sums.cuda()
        _lens = lengths.view(-1, 1).expand(sums.size(0), sums.size(1))
        means = sums / _lens.float()
        return means

    @staticmethod
    def max_pooling(x):
        maxed, _ = torch.max(x, dim=1)
        return maxed.float()

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(MeanMaxDNN, self).__init__()

        self.emb_dim = embeddings.shape[1]

        # 1 - define the embedding layer from pretrained weights
        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=not(trainable_emb), sparse=False)

        # 5 - define the final Linear layer which maps the representations to the classes
        self.final = nn.Linear(self.emb_dim, output_size)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        self.batch_size = lengths[0]
        self.l = lengths[1]
        self.maxlen = x.shape[1]

        # 1 - embed the words, using the embedding layer

        # batch_size, 21, emb_dim
        embedding = self.embed(x.long())
        if (torch.cuda.is_available()):
            embedding = embedding.cuda()

        # 2 - construct a sentence representation out of the word embeddings
        representations = torch.cat((self.mean_pooling(embedding, self.l), self.max_pooling(embedding)), 0)

        # 4 - project the representations to classes using a linear layer
        logits = self.final(representations)

        return logits

class LSTMDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    @staticmethod
    def mean_pooling(x, lengths):
        sums = torch.sum(x, dim=1).float()
        if (torch.cuda.is_available()):
            sums = sums.cuda()
        _lens = lengths.view(-1, 1).expand(sums.size(0), sums.size(1))
        means = sums / _lens.float()
        return means

    @staticmethod
    def max_pooling(x):
        maxed, _ = torch.max(x, dim=1)
        return maxed.float()

    def last_timestep(self, unpacked, lengths):
        # Index of the last output for each sequence.
        index = (lengths - 1).view(-1, 1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
        return unpacked.gather(1, index).squeeze()

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(LSTMDNN, self).__init__()

        self.emb_dim = embeddings.shape[1]

        # 1 - define the embedding layer from pretrained weights
        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=not(trainable_emb), sparse=False)

        # 4 - define a non-linear transformation of the representations
        self.tanh = nn.Tanh()

        # LSTM
        self.lstm_embed = nn.LSTM(input_size=self.emb_dim,
                           hidden_size=self.emb_dim, batch_first=True)

        # the dropout "layer" for the output of the RNN
        # self.drop_lstm = nn.Dropout(dropout_rnn)

        # 5 - define the final Linear layer which maps the representations to the classes
        self.final = nn.Linear(self.emb_dim, output_size)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        self.batch_size = lengths[0]
        self.l = lengths[1]
        self.maxlen = x.shape[1]

        # 1 - embed the words, using the embedding layer

        # batch_size, 21, emb_dim
        embedding = self.embed(x.long())
        if (torch.cuda.is_available()):
            embedding = embedding.cuda()

        # 2 - construct a sentence representation out of the word embeddings

        # h_N = torch.zeros([self.batch_size, self.maxlen, self.emb_dim])
        output, (h, c) = self.lstm_embed(embedding.float())
        # h_N[i] = h

        # last(batch_size * hidden_size) is h for last timestep, for each sentence of the batch
        last = self.last_timestep(output, self.l)
        print("kavli")
        # print(h[-1].shape)

        # representations = self.mean_pooling(embedding, self.l)
        representations = torch.cat((last, self.mean_pooling(output, self.l), self.max_pooling(output)), 0)
        raise ValueError
        # 3 - transform the representations to new ones.
        # representations = self.tanh(representations)

        # 4 - project the representations to classes using a linear layer
        logits = self.final(representations)

        return logits

class BidirectionalLSTMDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    @staticmethod
    def mean_pooling(x, lengths):
        sums = torch.sum(x, dim=1).float()
        if (torch.cuda.is_available()):
            sums = sums.cuda()
        _lens = lengths.view(-1, 1).expand(sums.size(0), sums.size(1))
        means = sums / _lens.float()
        return means

    @staticmethod
    def max_pooling(x):
        maxed, _ = torch.max(x, dim=1)
        return maxed.float()

    def last_timestep(self, unpacked, lengths):
        # Index of the last output for each sequence.
        index = (lengths - 1).view(-1, 1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
        return unpacked.gather(1, index).squeeze()

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BidirectionalLSTMDNN, self).__init__()

        self.emb_dim = embeddings.shape[1]

        # 1 - define the embedding layer from pretrained weights
        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=not(trainable_emb), sparse=False)

        # 4 - define a non-linear transformation of the representations
        self.tanh = nn.Tanh()

        # LSTM
        self.lstm_embed = nn.LSTM(input_size=self.emb_dim,
                           hidden_size=self.emb_dim, batch_first=True, bidirectional=True)

        # the dropout "layer" for the output of the RNN
        # self.drop_lstm = nn.Dropout(dropout_rnn)

        # 5 - define the final Linear layer which maps the representations to the classes
        self.final = nn.Linear(self.emb_dim, output_size)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        self.batch_size = lengths[0]
        self.l = lengths[1]
        self.maxlen = x.shape[1]

        # 1 - embed the words, using the embedding layer

        # batch_size, 21, emb_dim
        embedding = self.embed(x.long())
        if (torch.cuda.is_available()):
            embedding = embedding.cuda()

        # 2 - construct a sentence representation out of the word embeddings

        # h_N = torch.zeros([self.batch_size, self.maxlen, self.emb_dim])
        output, (h, c) = self.lstm_embed(embedding.float())
        # h_N[i] = h

        # last(batch_size * hidden_size) is h for last timestep, for each sentence of the batch
        last = self.last_timestep(output, self.l)
        # print(last.size())
        # print("kavli")
        # print(h[-1].shape)

        # representations = self.mean_pooling(embedding, self.l)
        representations = torch.cat((last, self.mean_pooling(output, self.l), self.max_pooling(output)), 0)
        # print(representations.size())
        # print(last)
        # print(self.mean_pooling(output, self.l))
        # print(self.max_pooling(output))
        # print(representations)
        # raise ValueError
        # 3 - transform the representations to new ones.
        # representations = self.tanh(representations)

        # 4 - project the representations to classes using a linear layer
        logits = self.final(representations)

        return logits

class AttentionDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(AttentionDNN, self).__init__()

        self.emb_dim = embeddings.shape[1]

        # 1 - define the embedding layer from pretrained weights
        # self.embed = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=not(trainable_emb), sparse=False)
        # define the embedding layer, with the corresponding dimensions
        self.embed = nn.Embedding(num_embeddings=embeddings.shape[0], embedding_dim=embeddings.shape[1])

        # initialize the weights of the Embedding layer,
        # with the given pre-trained word vectors
        self.embed.weight = nn.Parameter(torch.from_numpy(embeddings), requires_grad=trainable_emb)

        self.attention = SelfAttention(self.emb_dim, batch_first=True)

        # 5 - define the final Linear layer which maps the representations to the classes
        self.final = nn.Linear(self.emb_dim, output_size)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        self.batch_size = lengths[0]
        self.l = lengths[1]
        self.maxlen = x.shape[1]

        # 1 - embed the words, using the embedding layer

        # batch_size, 21, emb_dim
        embedding = self.embed(x.long())
        if (torch.cuda.is_available()):
            embedding = embedding.cuda()

        # 2 - construct a sentence representation out of the word embeddings

        representations, attentions = self.attention(embedding, self.l)

        # 4 - project the representations to classes using a linear layer
        logits = self.final(representations)

        return logits

class AttentionLSTMDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def last_timestep(self, unpacked, lengths):
        # Index of the last output for each sequence.
        index = (lengths - 1).view(-1, 1).expand(unpacked.size(0),
                                               unpacked.size(2)).unsqueeze(1)
        return unpacked.gather(1, index).squeeze()

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(AttentionLSTMDNN, self).__init__()

        self.emb_dim = embeddings.shape[1]

        # 1 - define the embedding layer from pretrained weights
        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=not(trainable_emb), sparse=False)

        # LSTM
        self.lstm_embed = nn.LSTM(input_size=self.emb_dim, hidden_size=self.emb_dim, batch_first=True)

        # the dropout "layer" for the output of the RNN
        # self.drop_lstm = nn.Dropout(dropout_rnn)

        self.attention = SelfAttention(self.emb_dim, batch_first=True)

        # 5 - define the final Linear layer which maps the representations to the classes
        self.final = nn.Linear(self.emb_dim, output_size)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        self.batch_size = lengths[0]
        self.l = lengths[1]
        self.maxlen = x.shape[1]

        # 1 - embed the words, using the embedding layer

        # batch_size, 21, emb_dim
        embedding = self.embed(x.long())
        if (torch.cuda.is_available()):
            embedding = embedding.cuda()

        # 2 - construct a sentence representation out of the word embeddings

        # # h_N = torch.zeros([self.batch_size, self.maxlen, self.emb_dim])
        output, (h, c) = self.lstm_embed(embedding.float())
        # # h_N[i] = h
        last = self.last_timestep(output, self.l)
        # print(h[-1].shape)

        representations, attentions = self.attention(last, self.l)

        # 4 - project the representations to classes using a linear layer
        logits = self.final(representations)

        return logits

class AttentionBidirectionalLSTMDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def last_timestep(self, unpacked, lengths):
        # Index of the last output for each sequence.
        index = (lengths - 1).view(-1, 1).expand(unpacked.size(0), unpacked.size(2)).unsqueeze(1)
        return unpacked.gather(1, index).squeeze()

    def __init__(self, output_size, embeddings, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(AttentionBidirectionalLSTMDNN, self).__init__()

        self.emb_dim = embeddings.shape[1]

        # 1 - define the embedding layer from pretrained weights
        self.embed = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=not(trainable_emb), sparse=False)

        # LSTM
        self.lstm_embed = nn.LSTM(input_size=self.emb_dim, hidden_size=self.emb_dim, batch_first=True, bidirectional=True)

        # the dropout "layer" for the output of the RNN
        # self.drop_lstm = nn.Dropout(dropout_rnn)

        self.attention = SelfAttention(2*self.emb_dim, batch_first=True)

        # 5 - define the final Linear layer which maps the representations to the classes
        self.final = nn.Linear(2*self.emb_dim, output_size)

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        self.batch_size = lengths[0]
        self.l = lengths[1]
        self.maxlen = x.shape[1]

        # 1 - embed the words, using the embedding layer

        # batch_size, 21, emb_dim
        embedding = self.embed(x.long())
        if (torch.cuda.is_available()):
            embedding = embedding.cuda()

        # 2 - construct a sentence representation out of the word embeddings

        # # h_N = torch.zeros([self.batch_size, self.maxlen, self.emb_dim])
        output, (h, c) = self.lstm_embed(embedding.float())
        # # h_N[i] = h
        hidden_size = output.size(2) // 2
        # straight RNN output and reverse one
        lstm_output = output[:, :, :hidden_size]
        lstm_reverse_output = output[:, :, hidden_size:]
        print(lstm_output.size(), lstm_reverse_output.size())
        # raise ValueError
        last = self.last_timestep(output, self.l)
        # print(h[-1].shape)

        representations, attentions = self.attention(last, self.l)

        # 4 - project the representations to classes using a linear layer
        logits = self.final(representations)

        return logits
