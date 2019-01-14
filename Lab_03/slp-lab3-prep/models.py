import torch
import numpy as np
from torch import nn


class BaselineDNN(nn.Module):
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

        super(BaselineDNN, self).__init__()

        self.emb_dim = 50

        # 1 - define the embedding layer
        # EX4
        # embed = nn.Embedding(10, 3)

        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        # EX4

        # 3 - define if the embedding layer will be frozen or finetuned
        # EX4
        # print(embeddings)
        embeddings = torch.from_numpy(embeddings)
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=not(trainable_emb), sparse=False)

        # 4 - define a non-linear transformation of the representations
        # EX5

        self.tanh = nn.Tanh()

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        # EX5

        self.final = nn.Linear(self.emb_dim, output_size)

    def forward(self, x, l, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        self.batch_size = lengths
        # 1 - embed the words, using the embedding layer
        # EX6

        # self.batch_size, 62. self.emb_dim
        embeddings = np.zeros((self.batch_size, 62, self.emb_dim))
        for i in range(self.batch_size):
            e = x[i]
            embeddings[i] = self.embed(e)  # EX6

        # 2 - construct a sentence representation out of the word embeddings
        # EX6
        representations = np.zeros((self.batch_size, self.emb_dim))
        for i in range(self.batch_size):
            for j in range(self.emb_dim):
                rep_sum = 0
                length = l[i]
                for k in range(length):
                    rep_sum += embeddings[i][k][j]
                rep_sum /= length
                representations[i][j] = rep_sum

        # 3 - transform the representations to new ones.
        # EX6
        representations = self.tanh(torch.from_numpy(representations).float())
        print(type(representations))

        # 4 - project the representations to classes using a linear layer
        # EX6
        logits = self.final(representations)

        return logits
