import numpy as np
import pandas as pd
from playgroundrl.client import CodenamesState, PlaygroundClient, GameType
from playgroundrl.actions import *
from playgroundrl.client import *
from playgroundrl.args import get_arguments
import time
import logging
from models.inference_model import InferenceModel
from gameplay import mdp
from gameplay.algo import ReinforcementQGiver, Guesser

logging.basicConfig(filename='codenames.log', encoding='utf-8', level=logging.INFO)
BOARD_SIZE = 25

def get_category_models(categories_file) -> InferenceModel:
    """Load category models for use in the gameplay

    Args:
        categories_file (str): a file maps the one hot vector generated by the model to string labels representing categories (clues)

    Returns:
        InferenceModel: _description_
    """
    df = pd.read_csv(categories_file, index_col=0)
    categories = np.array(df.columns)

    give_categories_model = InferenceModel(categories, "great nn", "output/0.5_0_False_0.0005_800_0.1399055489964581.pt", 'nn', 
            {
            "dropout_rate": 0.5,
            "weight_decay": 0,
            "batch_norm": False,
            "learning_rate": 0.0005,
            "hidden_dim": 800,
            "accuracy": 0.1399055489964581,
            "name": "0.5_0_False_0.0005_800_0.1399055489964581.pt"
            })
    
    #guesser categories model can be different from spymaster categories model for more diverse training
    guess_categories_model = give_categories_model

    
    return give_categories_model, guess_categories_model

class TestCodenames(PlaygroundClient):
    """Implements the interface required by the PlaygroundRL Codenames server
    """
    def __init__(self, q_filename:str, categories_file: str):
        super().__init__(
            GameType.CODENAMES,
            model_name="changeme",
            auth_file='auth.json',
            render_gameplay=True
        )
        give_categories_model, guess_categories_model = get_category_models(categories_file)
        self.giver = ReinforcementQGiver(q_filename, give_categories_model)
        self.guesser = Guesser(guess_categories_model)
        self.prev_hint_state = None
        self.prev_action = None

    def callback(self, state: CodenamesState, reward):
        if state.player_moving_id not in self.player_ids:
            self.is_my_turn = False
            logging.info("Is not my turn")
            return None
        
        logging.info("Is my turn!")
        self.is_my_turn = True

        if state.role == "GIVER":
            time.sleep(0.5)
            logging.info("I'm the Giver")

            if self.prev_hint_state is not None:
                current_hint_state = mdp.get_hint_state(state)
                logging.info(f"Current hint state: {current_hint_state}")
                self.giver.updateQ(self.prev_hint_state, current_hint_state, self.prev_action )
            
            hint, hint_state, action = self.giver.hint(state)
            clue, count = hint

            self.prev_hint_state = hint_state
            self.prev_action = action
            
            if len(clue.split()) > 1:
                clue = clue[1]

            return CodenamesSpymasterAction(clue, count)
        
        elif state.role == "GUESSER":
            if self.self_training:
                time.sleep(0.5)
            logging.info("I'm the Guesser")
            
            wordlist = [state.words[i] for i in range(len(state.words)) if state.guessed[i] == "UNKNOWN"]
            logging.info(f"Wordlist {wordlist}")
            
            guesses = self.guesser.guess(state.clue, state.count, wordlist)
            
            logging.info(f"Guesses {guesses}")
            guesses = [state.words.index(g.lower()) for g in guesses]
            return CodenamesGuesserAction(
                guesses = guesses
            )
        

    def gameover_callback(self):
        if self.is_my_turn:
            logging.info(f"Game over i won")
        else:
            logging.info(f"Game over i lost")

        self.giver.gameover(self.is_my_turn, self.prev_hint_state, self.prev_action)


if __name__ == "__main__":
    #save_file = sys.argv[1]
    t = TestCodenames("data/q.json", "data/600_gpt_post_audit.csv",)
    
    t.run(pool=Pool.MODEL_ONLY, num_games=10, self_training=False, game_parameters={"num_players": 4})
