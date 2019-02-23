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

from config import EMB_PATH, BASE_PATH
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
MODELS = os.path.join(BASE_PATH, "models")
if not os.path.exists(MODELS):
    os.makedirs(MODELS)
MODEL_PATH = os.path.join(MODELS, "best_model.pth.tar")
PREDICTION = os.path.join(MODELS, "best_model_eval_predictions.txt")
DATA_FILE_PATH = os.path.join(MODELS, "data.json")
LABELS_FILE_PATH = os.path.join(MODELS, "labels.json")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 1
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
# convert data labels from strings to integers
le = le.fit(y_train)
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
optimizer = torch.optim.Adam(parameters)  # EX8
# raise NotImplementedError

#############################################################################
# Training Pipeline
#############################################################################

total_train_loss = []
total_test_loss = []

min_test_loss = 100

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

    if (test_loss < min_test_loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_loss}, MODEL_PATH)
        min_test_loss = test_loss

    total_train_loss.append(train_loss)
    total_test_loss.append(test_loss)

best_model = PreLabBaselineDNN(output_size=n_classes,  # EX8
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE)

# move the mode weight to cpu or gpu
best_model = best_model.to(DEVICE)
optimizer = torch.optim.Adam(parameters)

checkpoint = torch.load(MODEL_PATH)
best_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

_, (best_y_test_gold, best_y_test_pred) = eval_dataset(test_loader,
                                                     best_model,
                                                     criterion)
pred_file = open(PREDICTION, "w")
for word in best_y_test_pred:
    pred_file.write(str(word) + '\n')
pred_file.close()

#NeAt JSONs

data_file = open(DATA_FILE_PATH,'w')

for i in range(len(test_set.data)):
    label = best_y_test_pred[i]
    print(label)
    sentence = test_set.data[i]
    print(sentence)
    # data_file.write('{\n')
    #
    # data_file.write('    "text":  [\n')
    # for word in sentence:
    #     data_file.write('      "' + word + '",\n')
    # data_file.write('    ],\n')
    #
    # data_file.write('    "text":  [\n')
    # data_file.write(str(label) + "\n")
    # data_file.write('    ],\n')
    #
    # data_file.write('    "attention":  [\n')
    # data_file.write('    ],\n')
    #
    # data_file.write('    "id": "sample_"' +str(i) + '"\n')

data_file.close()


filename = open(LABELS_FILE_PATH,'w')

filename.write('{\n')
filename.write('  "2":  {\n')
filename.write('    "name": "positive",\n')
filename.write('    "desc": "really_liked_it"\n')
filename.write('  },\n')

filename.write('  "0":  {\n')
filename.write('    "name": "negative",\n')
filename.write('    "desc": "really_hate_it"\n')
filename.write('  },\n')

filename.write('  "1":  {\n')
filename.write('    "name": "neutral",\n')
filename.write('    "desc": "dont_care"\n')
filename.write('  },\n')
filename.write('}\n')

filename.close()


plt.figure(1)
title = "Train and Test loss for "+ str(DATASET) + " Dataset with " + str(EMB_DIM) + " Embedding Dimension, " + str(EPOCHS) + " Epochs"
plt.title(title)
plt.plot(np.array(total_train_loss), 'o-', label='Train')
plt.plot(np.array(total_test_loss), 'o-', label='Test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
