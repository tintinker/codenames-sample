"""
Main logic for the Spymaster and Guesser roles
@author Justin Tinker (jatinker@cs.stanford.edu)
"""

from collections import defaultdict
from typing import List
from playgroundrl.client import CodenamesState
from models.inference_model import InferenceModel
from gameplay.hint import Hinter
import pandas as pd
import numpy as np
from gameplay import mdp
import json
import random
import logging
import dsp

ALPHA = 0.2
DISCOUNT_FACTOR = 0.9

with open("auth.json") as f:
    secrets = json.load(f)

OPENAI_KEY = secrets["OPENAI_KEY"]

class Guesser:

    def __init__(self,  categories_model: InferenceModel) -> None:
        """
        Args:
            categories_model (InferenceModel): underlying model used to map words on the board to provided clue (category)
        """

        self.model: InferenceModel = categories_model
        self.prepare_gpt_fallback()
    
    def prepare_gpt_fallback(self):
        """
        Set up few shot learning to be used if the model does not contain a category for the clue.
        """
        
        Question = dsp.Type(prefix="Criteria:", desc="${the criteria for the word(s), including the requested number and the list of options}")
        Answer = dsp.Type(prefix="Chosen Words:", desc="${a comma-seperated list of english words meeting the criterea}", format=dsp.format_answers)
        hint_template = dsp.Template(instructions=f"Provide the requested number of english words words that best meet the provided criterea. Choose from the list of options.", question=Question(), answer=Answer())
        demos = [
            (f"The requested number of words is 3. Choose the 3 words from the list that are most similar to grow. The list of options is: apple, banana, igloo, berlin, montana, rainforest, choke, roulette, surround.", ['apple, banana, rainforest'])
            ]
        demos = [dsp.Example(question=question, answer=answer) for question, answer in demos]
        
        self.demos = demos
        self.hint_template = hint_template
        self.lm = dsp.GPT3(model='gpt-3.5-turbo', api_key=OPENAI_KEY, model_type='chat')
        
        dsp.settings.configure(lm=self.lm)
 
    def generate_fallback_prompt(self, clue: str, count: int, wordlist: List[int]) -> str:
        """Generate prompt for few shot learning

        Args:
            clue (str): clue provided by spymaster
            count (int): count provided by spymaster
            wordlist (List[int]): words on the board

        Returns:
            the prompt
        """
        return f"The requested number of words is {count}. Choose the {count} words from the list that are most similar to {clue}. The list of options is: {', '.join(wordlist)}."


    def get_gpt_dsp_hint(self, clue: str, count: int, wordlist: List[int]) -> List[str]:
        """Choose words with GPT-3 few shot learning

        Args:
            clue (str): clue provided by spymaster
            count (int): count provided by spymaster
            wordlist (List[int]): words on the board


        Returns:
            List[str]: chosen words
        """
        example = dsp.Example(question=self.generate_fallback_prompt(clue, count, wordlist), demos=self.demos)
        _, completions = dsp.generate(self.hint_template)(example, stage='qa')
        hints = completions.answer.split(",")
        self.lm.inspect_history()
        return [word.strip() for word in hints if word.strip().lower() in wordlist][:count]


    def guess(self, clue: str, count: int, wordlist: List[int]) -> List[str]:
        """Generate guess

        Args:
            clue (str): _description_
            count (int): _description_
            wordlist (List[int]): _description_

        Returns:
            List[str]: chosen words
        """

        if clue not in self.model.categories:
            logging.info("Clue {clue} not in category list, using GPT3")
            return self.get_gpt_dsp_hint(clue, count, wordlist)

        thoughts = [self.model.get_prob(word, clue) for word in wordlist]
        thoughts = sorted(thoughts, key = lambda t:t[0], reverse=True)

        logging.info(f"Clue: {clue}")
        logging.info(f"Thoughts: {thoughts}")

        return [word for score, word in thoughts][:count]
    
class ReinforcementQGiver:
    """Main class for the Spymaster role
    """
    def __init__(self, q_filename: str, categories_model: InferenceModel) -> None:
        """
        Args:
            q_filename (str): filename containing the current Q-table
            categories_model (InferenceModel): underlying model used to map words on the board to provided categories (clues)
        """
        self.q_filename = q_filename

        with open(q_filename) as f:
            q_data = json.load(f)

        self.Q: defaultdict(defaultdict(int)) = q_data["q"]
        self.update_count: defaultdict(defaultdict(int)) = q_data["update_count"]
        
        self.hinter = Hinter(categories_model)
        self.weighting_strategies: List[mdp.WeightingMetric] = mdp.ACTIONS #prioritization strategies

    def choose_optimal_action(self, current_hint_state: mdp.HintState, random_prob: float = 0) -> mdp.WeightingMetric:
        """Based on the current gamestate (how many words successfully guessed by the respective teams), chose how aggressive the clue strategy should be

        Args:
            current_hint_state (mdp.HintState): 
            random_prob (float, optional): alpha value, chance that we explore instead of using the q-tables suggestion. Defaults to 0.

        Returns:
            _type_: _description_
        """
        if current_hint_state not in self.Q:
            self.Q[str(current_hint_state)] = defaultdict(int)
            self.update_count[str(current_hint_state)] = defaultdict(int)

        possible_actions = self.Q[str(current_hint_state)].keys()
        if not possible_actions or random.random() < random_prob:
            return random.choice(self.weighting_strategies)
        else:
            return max(possible_actions, key = lambda a:self.Q[str(current_hint_state)][str(a)])
        

    def hint(self, codenames_state: CodenamesState) -> str:
        """Main interface for generate a clue given the current game state

        Args:
            codenames_state (CodenamesState): current gamestate

        Returns:
            str: one word clue 
        """
        current_hint_state = mdp.get_hint_state(codenames_state)
        strategy_action = self.choose_optimal_action(current_hint_state, random_prob=ALPHA)
        return self.hinter.hint(codenames_state, strategy_action), current_hint_state, strategy_action

    def _calculate_and_update(self, prev_hint_state: mdp.HintState, action: mdp.WeightingMetric, reward: float, futureQ: float):
        """Q learning implementation. Update table based on the success of the previous clue

        Args:
            prev_hint_state (mdp.HintState): 
            action (mdp.WeightingMetric): 
            reward (float): 
            futureQ (float): 
        """
        if prev_hint_state not in self.Q:
            self.Q[str(prev_hint_state)] = defaultdict(int)
            self.update_count[str(prev_hint_state)] = defaultdict(int)

        eta = 1 / (1 + self.update_count[str(prev_hint_state)][str(action)])

        curQ = self.Q[str(prev_hint_state)][str(action)]

        self.update_count[str(prev_hint_state)][str(action)] += 1
        self.Q[str(prev_hint_state)][str(action)] = (1-eta) * curQ + (eta * (reward + DISCOUNT_FACTOR * futureQ))


    def updateQ(self, prev_hint_state: mdp.HintState, new_hint_state: mdp.HintState, action: mdp.WeightingMetric):
        """Utility function for updating the Q-table

        Args:
            prev_hint_state (mdp.HintState): 
            new_hint_state (mdp.HintState): 
            action (mdp.WeightingMetric): 
        """

        if new_hint_state not in self.Q:
            self.Q[str(new_hint_state)] = defaultdict(int)
            self.update_count[str(new_hint_state)] = defaultdict(int)

        futureQ = self.Q[str(new_hint_state)][str(self.choose_optimal_action(new_hint_state))]
        reward = mdp.reward(new_hint_state, prev_hint_state)
        self._calculate_and_update(prev_hint_state, action, reward, futureQ )
        
  

    def gameover(self, won: bool, prev_hint_state: mdp.HintState,  action: mdp.WeightingMetric, q_export_filename=None):
        """Export the q table after the end of the current game for re-use in the future.

        Args:
            won (bool): 
            prev_hint_state (mdp.HintState): 
            action (mdp.WeightingMetric): 
            q_export_filename (_type_, optional): . Defaults to None.
        """
        if won:
            reward = mdp.REWARD_FOR_WINNING
            futureQ = 0
            self._calculate_and_update(prev_hint_state, action, reward, futureQ )
        
        if not q_export_filename:
            q_export_filename = self.q_filename
        
        with open(q_export_filename, "w+") as f:
            json.dump({"q": self.Q, "update_count": self.update_count}, f)

