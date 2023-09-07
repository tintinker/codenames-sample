# Gameplay

Main gameplay module. Entrypoint is `codenames.py`. Also targeted by `run10x.sh`. Using a pretrained neuralnet or decision tree model (see `models` directory), interfaces with the PlaygroundRL Codenames server to play the game as either a spymaster or a guesser.

As a spymaster, it will run all words on the board through the chosen models, and choose categories (clues) that maximize benefits to the spymasters team, according to the weighting metric chosen by the Q-Table. Weighting metrics are modeled using a Markov Decision Process (see `mdp.py`)

As a guesser, will run the clue through the provided model if the clue matches the model's category list, otherwise will fall back to supervised few-shot learning with chatGPT api.

