from typing import List, Literal, Tuple
import joblib
import torch
import numpy as np
from models.neuralnet import MulticlassClassifier
from torchtext.vocab import GloVe

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

glove = GloVe(name='6B', dim=300)

class InferenceModel:
    def __init__(self, categories: np.array, name: str, filename: str, kind: Literal['tree', 'nn'], params: dict = None):
        """

        Args:
            categories (np.array): string labels for the model one-hot vector output
            name (str): human readable description of this model (usually the type and the params)
            filename (str): filename containing the model weights
            kind (Literal['tree','nn']): whether it's a decision tree or neural net
            params (dict, optional): hyperparams used during training. Defaults to None.
        """
        self.name = name
        self.kind = kind
        self.categories = categories

        if self.kind == 'tree':
            self.underlying_model = joblib.load(filename)
        else:
            self.underlying_model = MulticlassClassifier(300, params["hidden_dim"], 600, params["dropout_rate"], params["weight_decay"], params["batch_norm"])
            self.underlying_model.load_state_dict(torch.load(filename))
            self.underlying_model.eval()
            
    def infer(self, word, threshold: float = 0.5) -> List[Tuple[float, str]]:
        embedding = glove[word]

        if self.kind == 'tree':
            prediction = self.underlying_model.predict(embedding.reshape(1, -1))
        else:
            with torch.no_grad():
                predicted = self.underlying_model(embedding)
                predicted = sigmoid(predicted.detach().numpy())
                prediction = (predicted > threshold)
        

        categories = [str(cat) for cat in self.categories[prediction.flatten()]]
        scores = [float(score) for score in predicted[prediction.flatten()]]
        return list(zip(scores, categories))
    
    def get_prob(self, word: str, category: str) -> Tuple[float, str]:
        """Returns the probability (sigmoid) that word is in category

        Args:
            word (str): 
            category (str): 

        Returns:
            score, word
        """
        word = word.strip().lower()
        category = category.strip().lower()

        if category not in self.categories:
            print(f"Category not found for: word: {word} category: {category}. Defaulting to modified cosine similarity")
            if word not in glove.stoi or category not in glove.stoi:
                return 0, word
            similarity = cosine_similarity(glove[word], glove[category])
            
            #if words are more than 70% dissimilar, consider them similar
            if similarity < -0.7:
                return -1 * similarity, word
            return similarity, word
        
        rankings = self.rank(word)
        for score, cat in rankings:
            if cat == category:
                return score, word

    def rank(self, word: str, n=600) -> List[str]:
        """Return the top n categories for the given word

        Args:
            word (str): target word
            n (int, optional): number of categories to return. Defaults to 600.

        Returns:
            List[str]: top n categories
        """
        if self.kind == 'tree':
            raise "Trees can't rank :("
        
        embedding = glove[word]

        with torch.no_grad():
            predicted = self.underlying_model(embedding)
            predicted = sigmoid(predicted.detach().numpy())
            sorted_nzs = np.argsort(predicted)[::-1][:n]
            results = [(predicted[i], str(self.categories[i])) for i in sorted_nzs]
        return results
   