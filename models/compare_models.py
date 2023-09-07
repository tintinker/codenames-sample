from typing import List, Literal
import joblib
import sys
from torchtext.vocab import GloVe
import numpy as np
import pandas as pd
import torch
from models.inference_model import InferenceModel
from models.neuralnet import MulticlassClassifier
import json

glove = GloVe(name='6B', dim=300)

def list_categories_for_word(df, word):
    if word in df.index:
        categories_for_word = df.columns[df.loc[word].astype(bool)].tolist()
        return ', '.join(categories_for_word)
    else:
        return "Not found in ground truth."

def list_best_models(folder, output="best_worst_models.json", n=10):
    with open(f"{folder}/map.json") as f:
        data = json.load(f)
    data = sorted(data, key=lambda v:v["accuracy"], reverse=True)
    data = data[:n] + data[-n:]
    with open(output, "w+") as f:
        json.dump(data, f)

def compare_category_matches(categories_file):
    """Compare categories generated by the models with the training data

    Args:
        categories_file (str): filename of the training data & labels for model vector output
    """
    
    df = pd.read_csv(categories_file, index_col=0)
    categories = np.array(df.columns)
    
    models: List[InferenceModel] = [
        InferenceModel(categories, "good tree", "treeoutput/tree_10_10.model.joblib", 'tree'),
        
        InferenceModel(categories, "bad tree", "treeoutput/tree_30_10.model.joblib", 'tree'),
        
        InferenceModel(categories, "good nn", "output/0.5_0_False_0.0005_800_0.1399055489964581.pt", 'nn', 
            {
            "dropout_rate": 0.5,
            "weight_decay": 0,
            "batch_norm": False,
            "learning_rate": 0.0005,
            "hidden_dim": 800,
            "accuracy": 0.1399055489964581,
            "name": "0.5_0_False_0.0005_800_0.1399055489964581.pt"
            }),
        
        InferenceModel(categories, "bad nn", "output/0.3918280395373651_0.001_False_0.001_96_0.0.pt", 'nn', 
            {
            "dropout_rate": 0.3918280395373651,
            "weight_decay": 0.001,
            "batch_norm": False,
            "learning_rate": 0.001,
            "hidden_dim": 96,
            "accuracy": 0,
            "name": "0.3918280395373651_0.001_False_0.001_96_0.0.pt"
            }),
    ]
    
    while True:
        word = input("\nPlease enter a word: ")
        embedding = glove[word]

        for model in models:
            print(model.name, ":", model.infer(embedding))

        print("ground truth: ", list_categories_for_word(df, word))

def compare_top_ten(categories_file):
    """Compare the top ten categories for a word based on the model's score

    Args:
        categories_file (str): filename of the training data & labels for model vector output
    """
    
    df = pd.read_csv(categories_file, index_col=0)
    categories = np.array(df.columns)
    
    models: List[InferenceModel] = [
        InferenceModel(categories, "good nn", "output/0.5_0_False_0.0005_800_0.1399055489964581.pt", 'nn', 
            {
            "dropout_rate": 0.5,
            "weight_decay": 0,
            "batch_norm": False,
            "learning_rate": 0.0005,
            "hidden_dim": 800,
            "accuracy": 0.1399055489964581,
            "name": "0.5_0_False_0.0005_800_0.1399055489964581.pt"
            }),
    ]

    
    while True:
        word = input("\nPlease enter a word: ")
        embedding = glove[word]
        for model in models:
            print(model.name, ":", model.rank(embedding, n=10))

        print("ground truth: ", list_categories_for_word(df, word))
    
 
if __name__ == '__main__':
    compare_top_ten(sys.argv[1])
    #compare_category_matches(sys.argv[1])