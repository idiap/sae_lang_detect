# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Shantipriya Parida <shantipriya.parida@idiap.ch>,
#            Esau villatoro tello <esau.villatoro@idiap.ch>,
#            Petr Motlicek <petr.motlicek@idiap.ch>

# This file is part of SAE_LANG_DETECT package.

# SAE_LANG_DETECT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.

# SAE_LANG_DETECT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with SAE_LANG_DETECT. If not, see <http://www.gnu.org/licenses/>.

"""
    Here is a sample implementation of a supervised autoencoder for the language detection task.
    The language detection dataset (Ling10) used for the language detection task. 
    It used Bayesian Optimizer for searching the best hyper-parameters. The result stored in an output folder.
    The character n-gram used for feature extraction and input to the supervised autoencoder for language detection.  
""" 

from skopt.utils import use_named_args
from supervisedAE import SupervisedAE
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import torch
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from zipfile import ZipFile
import requests
import numpy as np
import shutil
import os
import json
from io import open
import pickle

##Load dataset
#...............
LING10_TRAIN_LARGE = "Ling10-trainlarge"
LING10_TRAIN_MEDIUM = "Ling10-trainmedium"
LING10_TRAIN_SMALL = "Ling10-trainsmall"

DATA_DIR = os.path.join(os.getcwd(),"data_dir")
OUTPUT_DIR = os.path.join(os.getcwd(),"output")
MODEL_DIR = os.path.join(os.getcwd(),"model")

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

DATA_FILE = os.path.join(DATA_DIR,LING10_TRAIN_LARGE+".zip")


#DONWLOAD THE DATA IF IT DOESN'T EXIST
if not os.path.exists(DATA_FILE):
    data = requests.get("https://github.com/johnolafenwa/Ling10/releases/download/1.0.0/{}.zip".format(LING10_TRAIN_LARGE),stream=True)
    
    with open(DATA_FILE,"wb") as file:
        shutil.copyfileobj(data.raw,file)

    del data

train_file = os.path.join(DATA_DIR,LING10_TRAIN_LARGE,"train_set.txt")
test_file = os.path.join(DATA_DIR,LING10_TRAIN_LARGE,"test_set.txt")
chars_map_file = os.path.join(DATA_DIR,LING10_TRAIN_LARGE,"chars.json")

metric_file = os.path.join(OUTPUT_DIR,"metrics_ling10_test.txt")
result_file = os.path.join(OUTPUT_DIR,"result_ling10_test.csv")

#If any of the content doesn't exist, extract the zip file
if not os.path.exists(train_file) or not os.path.exists(test_file) or not os.path.exists(chars_map_file):
    extractor = ZipFile(DATA_FILE)
    extractor.extractall(DATA_DIR)
    extractor.close()

train_data = open(train_file,encoding="utf-8").read()
test_data = open(test_file,encoding="utf-8").read()

#Split the train and test into lines
train_data = train_data.splitlines()
test_data = test_data.splitlines()

#Create arrays to store the integer version of the data
train_sentences = []
train_classes = []

test_sentences = []
test_classes = []

for _, line in enumerate(train_data):
    sen, lang_class = line.split("\t")
    train_sentences.append(sen)
    train_classes.append(int(lang_class))

for _, line in enumerate(test_data):
    sen, lang_class = line.split("\t")
    test_sentences.append(sen)
    test_classes.append(int(lang_class))

cf = CountVectorizer(analyzer='char_wb', ngram_range=(1,3), min_df=0.001, max_df=0.3)
train_counts = cf.fit_transform(train_sentences)
test_counts = cf.transform(test_sentences)

tfidf_transformer = TfidfTransformer()
train_tf = tfidf_transformer.fit_transform(train_counts)
test_tf = tfidf_transformer.transform(test_counts)

X_train = train_tf.toarray()
X_test = test_tf.toarray()
y_train = np.asarray(train_classes)
y_test = np.asarray(test_classes)


#path to save model file
path = os.path.join(MODEL_DIR,"model_ling10.pt")
print("model path:",path)

#Input parameter for Autoencoders
input_dim = X_train.shape[1]
emb_dim = 300
num_targets = 10
num_chunks = 30
supervision = 'clf'
convg_thres = 1e-5
max_epochs = 500
is_gpu = True
gpu_ids = 0

model = SupervisedAE(input_dim=input_dim, emb_dim=emb_dim, num_targets=num_targets, \
                    num_chunks=num_chunks, supervision=supervision, convg_thres=convg_thres, \
                    max_epochs=max_epochs, is_gpu=is_gpu, gpu_ids=gpu_ids)

#Hyperparameter search space
search_space = [Integer(1, 5, name='num_layers'),
                Real(10**-5, 10**-2, "log-uniform", name='learning_rate'),
                Real(10**-6, 10**-3, "log-uniform", name='weight_decay'),
                Categorical(['relu', 'sigma'], name='activation')]

@use_named_args(search_space)
def optimize(**params):
    print("Training with hyper-parameters: ", params)
    model.set_params(params)
    model.fit(X_train, y_train)
    return model.loss

# define the space of hyperparameters to search
result = gp_minimize(optimize, search_space, n_calls=10)
# summarizing finding:
print('best score: {}'.format(result.fun))
print('best params:')
print('num_layers: {}'.format(result.x[0]))
print('learning_rate: {}'.format(result.x[1]))
print('weight_decay: {}'.format(result.x[2]))
print('activation: {}'.format(result.x[3]))

num_layers = result.x[0]
learning_rate = result.x[1]
weight_decay = result.x[2]
activation = result.x[3]
model = SupervisedAE(input_dim=input_dim, emb_dim=emb_dim, num_targets=num_targets, \
                    num_layers=num_layers, learning_rate=learning_rate, weight_decay=weight_decay,
                    num_chunks=num_chunks, supervision=supervision, convg_thres=convg_thres, \
                    activation=activation, max_epochs=max_epochs, is_gpu=is_gpu, gpu_ids=gpu_ids)

model.fit(X_train, y_train)
print('saving model here:',path)
model.save(path)

########################################################
# Predict  Test set
########################################################

predicted,_ = model.predict(X_test)
predicted = torch.argmax(predicted, 1).cpu().numpy()

cf = metrics.confusion_matrix(y_test, predicted)
print("confusion matrix for test set:")
print(cf)

##Change according to the appropriate metrics function
print("Writing the metrics into test set file....")
f1 = open(metric_file, 'w+')

#########################################################
# Print testset result
#########################################################
f1.write("Testset Matrics")
f1.write("\n")
f1.write(metrics.classification_report(y_test, predicted))
f1.write(str(cf))
f1.close()

#########################################################
# Print testset output
#########################################################
df = pd.DataFrame(data={'test set':test_sentences, 'pred':predicted})
df.to_csv(result_file, index=False)
print("Done....")
