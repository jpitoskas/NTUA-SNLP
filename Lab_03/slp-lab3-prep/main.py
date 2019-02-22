import os
import warnings
from random import randint
import numpy as np
import matplotlib.pyplot as plt

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from models import PreLabBaselineDNN, MeanMaxDNN, LSTMDNN, AttentionDNN, AttentionLSTMDNN, AttentionBidirectionalLSTMDNN, BidirectionalLSTMDNN
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 30
DATASET = "MR"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
print(torch.version.cuda)

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

le = LabelEncoder()
le = le.fit(y_train)
# convert data labels from strings to integers
y_train = le.transform(y_train)  # EX1
y_test = le.transform(y_test)  # EX1
n_classes = le.classes_.size  # EX1 - LabelEncoder.classes_.size


# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

# EX4 - Define our PyTorch-based DataLoader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # EX7
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)  # EX7

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
model = PreLabBaselineDNN(output_size=n_classes,  # EX8
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE)

# move the mode weight to cpu or gpu
model = model.to(DEVICE)
print(model)


# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
criterion = torch.nn.CrossEntropyLoss()  # EX8
parameters = []  # EX8
for p in model.parameters():
    # p.requires_grad = False
    if(p.requires_grad==True):
        parameters.append(p)
optimizer = torch.optim.RMSprop(parameters)  # EX8
# raise NotImplementedError

#############################################################################
# Training Pipeline
#############################################################################

total_train_loss = []
total_test_loss = []

for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader,
                                                            model,
                                                            criterion)
    print()
    print("Train Set: loss={:.4f}, accuracy={:.4f}, f1={:.4f}, recall={:.4f}".format(train_loss, accuracy_score(y_train_gold, y_train_pred), f1_score(y_train_gold, y_train_pred, average='macro'), recall_score(y_train_gold, y_train_pred, average='macro')))
    test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader,
                                                         model,
                                                         criterion)
    print("Test Set: loss={:.4f}, accuracy={:.4f}, f1={:.4f}, recall={:.4f}".format(test_loss, accuracy_score(y_test_gold, y_test_pred), f1_score(y_test_gold, y_test_pred, average='macro'), recall_score(y_train_gold, y_train_pred, average='macro')))

    total_train_loss.append(train_loss)
    total_test_loss.append(test_loss)

plt.figure(1)
title = "Train and Test loss for "+ str(DATASET) + " Dataset with " + str(EMB_DIM) + " Embedding Dimension, " + str(EPOCHS) + " Epochs"
plt.title(title)
plt.plot(np.array(total_train_loss), 'o-', label='Train')
plt.plot(np.array(total_test_loss), 'o-', label='Test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
