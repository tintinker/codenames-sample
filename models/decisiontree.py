import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from torchtext.vocab import GloVe
import joblib 
import sys
from sklearn.multioutput import MultiOutputClassifier
from tqdm import tqdm

def main(df, folder_name):

    # Convert the DataFrame to a numpy array
    word_list = df.index.tolist()
    target = df.to_numpy()
    
    # Convert words to GloVe vectors
    glove = GloVe(name='6B', dim=300)
    word_vectors = [glove[word] for word in word_list]
    print(word_list[:20])

    # Convert word vectors and target to numpy arrays
    X = np.array(word_vectors)
    y = target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a hyperparameter grid for the sweep
    max_depths = [None, 10, 20, 30]
    min_samples_splits = [2, 5, 10]

    for max_depth in tqdm(max_depths, position=0):
        for min_samples_split in tqdm(min_samples_splits, leave=False, position=1):
            tqdm.write(f"\nTraining with max_depth={max_depth}, min_samples_split={min_samples_split}")
            
            # Instantiate the model with verbose setting
            model = MultiOutputClassifier(DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split))
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Predict and evaluate the model
            predicted = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, predicted)
            
            tqdm.write(f"Test accuracy: {test_accuracy:.2f}")
            joblib.dump(model, f"{folder_name}/tree_{max_depth}_{min_samples_split}.model.joblib")

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], index_col=0)
    main(df, sys.argv[2])