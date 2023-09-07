# Models

Contains two main models (decision tree and neuralnet), as well as utility scripts for qualitative probing of generated models.
The models are run using a hyperparameter sweep, and stored in a generated `outputs` directory. The `map.json` file created inside the `outputs` directory, stores a dictionary `filename` -> `hyperparameters` and `accuracy` information.

## Model Definitions

### Inference Model
Abtract class interface for running inference, can work with a decision tree or a neuralnet under the hood

### Decision Tree
Multilabel decision tree using scikit-learn. Under the hood, one tree per category is generated using a binary (is this word described by this category?). Input is 300d glove word embedding, output is 600 dimension one hot vector.

### Neuralnet
Also multilabel classification. Input is 300d glove word embedding, output is 600 dimension one hot vector representing which categories represent the target (input) word.

## Utility Scripts

### `compare_models.py`

Point the script to a few different model choices, then it will expose a command-line interface. For each word you input, it will run inference using the provided models, comparing the model's predicted categories to the 'ground truth' training data. The models should generalize the training data, such that words not included in the training data are still assigned categories.

### `compare_models.py`

Explore the training data
