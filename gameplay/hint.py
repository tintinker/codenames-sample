from collections import defaultdict
from typing import List
from models.inference_model import InferenceModel
from playgroundrl.client import CodenamesState
from spymaster.mdp import WeightingMetric, HintState
import logging
from nltk.corpus import wordnet as word_corpus_loader
import nltk
import dsp
import random
import json 


try:
    word_corpus_loader.ensure_loaded()
except:
    nltk.download("wordnet")

nltk_word_corpus = set(word_corpus_loader.words())

with open("auth.json") as f:
    secrets = json.load(f)

OPENAI_KEY = secrets["OPENAI_KEY"]

default_weights = WeightingMetric(1, 1, -1.5, -1.5, -0.5, -0.5, -10, -10)

MIN_THRESHOLD = 0.4

class Hinter:
    """Main class for generating clues for the Spymaster
    """
    def __init__(self, model: InferenceModel, stretch_threshold = 0.25) -> None:
        """

        Args:
            model (InferenceModel): _description_
            stretch_threshold (float, optional): If the score for the word is above this threshold, we consider the word to be a part of the given category. 
                For a friendly word, we want this to be over 0.5.
                For a non-friendly word, we want a good buffer so our guesser doesn't accidentally guess an enemy word
                Defaults to 0.25.
        """
        self.model = model
        self.stretch_threshold = stretch_threshold
        self.prepare_gpt_fallback()
        self.last_hint = ""

    def prepare_gpt_fallback(self):
        """If there are no good clues available from the pretrained category, fall back to few-shot learning.
        """
        Question = dsp.Type(prefix="Criteria:", desc="${the criteria for the word}")
        Answer = dsp.Type(prefix="Chosen Words:", desc="${a comma-seperated list of three english words meeting the criterea}", format=dsp.format_answers)
        hint_template = dsp.Template(instructions=f"Provide three english words words that meet the provided criterea. Any words returned must be included in the NLTK word corpus.", question=Question(), answer=Answer())
        demos = [
            (f"The word should be similar to {'roulette'}. It should not be related to money, death, or gun. it cannot be a substring or a superstring of apple, banana, or gun.", ['wheel, game, chance'])
            ]
        demos = [dsp.Example(question=question, answer=answer) for question, answer in demos]
        self.demos = demos
        self.hint_template = hint_template
        self.lm = dsp.GPT3(model='gpt-3.5-turbo', api_key=OPENAI_KEY, model_type='chat')
        dsp.settings.configure(lm=self.lm)
    
    def generate_fallback_prompt(self):
        return f"The word should be similar to {random.choice(self.my_words)}. It should not be related to {', '.join(self.opp_words)} or {self.assasin_word}. it cannot be a substring or a superstring of {', '.join(self.my_words + self.neutral_words)}"

    
    def get_my_words(self, state: CodenamesState, remaining: bool = True):
        self.my_words = [state.words[i] for i in range(len(state.words)) if state.actual[i] == state.color and (state.guessed[i] == "UNKNOWN" or not remaining)]
        logging.info(f"My words remaining: {self.my_words}")

    def get_opp_words(self, state: CodenamesState, remaining: bool = True):
        opp_color = "BLUE" if state.color == "RED" else "RED"
        self.opp_words = [state.words[i] for i in range(len(state.words)) if state.actual[i] == opp_color and (state.guessed[i] == "UNKNOWN" or not remaining)]

    def get_neutral_words(self, state: CodenamesState, remaining: bool = True):
        self.neutral_words = [state.words[i] for i in range(len(state.words)) if state.actual[i] == "INNOCENT" and (state.guessed[i] == "UNKNOWN" or not remaining)]

    def get_assasin_word(self, state: CodenamesState):
        for word, color in zip(state.words, state.actual):
            if color == "ASSASSIN":
                self.assasin_word = word
    
    def update_words(self, state: CodenamesState):
        self.get_my_words(state)
        self.get_opp_words(state)
        self.get_neutral_words(state)
        self.get_assasin_word(state)
    
    def get_gpt_dsp_hint(self, codenames_state: CodenamesState):
        """Fall back to GPT-3 few shot learning if no options remain

        Args:
            codenames_state (CodenamesState): 

        Returns:
            hint: str, int
        """
        example = dsp.Example(question=self.generate_fallback_prompt(), demos=self.demos)
        _, completions = dsp.generate(self.hint_template)(example, stage='qa')
        hints = completions.answer.split(",")
        self.lm.inspect_history()
        hints = [{"category": h.strip(), "num_my_words": 1, "score": 1} for h in hints]
        hints = [h for h in hints if self.valid_hint(h, codenames_state)]
        if len(hints) == 0:
            return "test", 1
        self.last_hint = hints[0]["category"]
        return hints[0]["category"], 1

    def get_categories(self, wordlist: List[str], threshold=0.5) -> dict:
        """Using the underlying model, return all the possible categories associated with word on the board

        Args:
            wordlist (List[str]): words on the board
            threshold (float, optional): score must be greater than this value for word to be consider part of the category. Defaults to 0.5.

        Returns:
            dict: possible categories -> associated words
        """
        categories =  defaultdict(list)
        for word in wordlist:
            for score, cat in self.model.infer(word, threshold=threshold):
                categories[cat].append((score, word))
        return categories
    
    def score(self, weights: WeightingMetric, num_my_words, my_word_sum, num_opp_words, opp_word_sum, num_neutral_words, neutral_word_sum, num_ass_words, ass_word_sum):
        """Generate a score for the clue based on the provided weigting metric - balancing proximity to your words vs proximity to unfriendly words

        Args:
            weights (WeightingMetric): 
            num_my_words (int): 
            my_word_sum (float): 
            num_opp_words (int): 
            opp_word_sum (float): 
            num_neutral_words (int): 
            neutral_word_sum (float): 
            num_ass_words (int): 
            ass_word_sum (float): 

        Returns:
            float: score
        """
        return weights.num_my_words * num_my_words \
            + weights.my_word_sum * my_word_sum \
            + weights.num_opp_words * num_opp_words \
            + weights.opp_word_sum * opp_word_sum \
            + weights.num_neutral_words * num_neutral_words \
            + weights.neutral_word_sum * neutral_word_sum \
            + weights.num_ass_words * num_ass_words \
            + weights.ass_word_sum * ass_word_sum
    
    def valid_hint(self, ci: dict, codenames_state: CodenamesState):
        """Test to make sure the clue is valid

        Args:
            ci (dict): a category info dictionary generated from 'generate_category_info' method
            codenames_state (CodenamesState):

        Returns:
            _type_: True if clue is valid
        """
        not_substring = (ci["category"] not in " ".join(codenames_state.words))
        in_corpus =  (ci["category"] in nltk_word_corpus)
        not_superstring = not (sum([1 for w in codenames_state.words if w in ci["category"]]))
        positive_score = ci["score"] > 0
        not_last = not (self.last_hint == ci["category"])
        return not_substring and in_corpus and not_superstring and positive_score and not_last

    def generate_category_info(self, c: str, weights: WeightingMetric, my_categories: List[str], opp_categories_stretch: List[str], neutral_categories_stretch: List[str], assasin_categories_stretch: List[str]):
        """Generate metrics for a given clue

        Args:
            c (str): the clue
            weights (WeightingMetric): 
            my_categories (List[str]): categories that represent our teams words
            opp_categories_stretch (List[str]): categories that represent the opposing teams words (usually using a more generous stretch threshold)
            neutral_categories_stretch (List[str]): see above
            assasin_categories_stretch (List[str]): see above

        Returns:
            dict: category info dictionary
        """
        my_words = [(score, word) for score,word in  my_categories[c]]
        num_my_words = len(my_categories[c])
        my_word_sum = sum([score for score, word in my_words])

        opp_words = []
        num_opp_words = 0
        opp_word_sum = 0
        if c in opp_categories_stretch:
            opp_words = [(score, word) for score,word in  opp_categories_stretch[c]]
            num_opp_words = len(opp_categories_stretch[c])
            opp_word_sum = sum([score for score, word in opp_words])

        neutral_words = []
        num_neutral_words = 0
        neutral_word_sum = 0
        if c in opp_categories_stretch:
            neutral_words = [(score, word) for score,word in  neutral_categories_stretch[c]]
            num_neutral_words = len(neutral_categories_stretch[c])
            neutral_word_sum = sum([score for score, word in neutral_words])

        ass_words = []
        num_ass_words = 0
        ass_word_sum = 0
        if c in opp_categories_stretch:
            ass_words = [(score, word) for score,word in  assasin_categories_stretch[c]]
            num_ass_words = len(assasin_categories_stretch[c])
            ass_word_sum = sum([score for score, word in ass_words])

        total_score = self.score(weights, num_my_words, my_word_sum, num_opp_words, opp_word_sum, num_neutral_words, neutral_word_sum, num_ass_words, ass_word_sum)

        return {
            "category": c,
            "my_words": my_words,
            "num_my_words": num_my_words,
            "my_word_sum": my_word_sum,
            "opp_words": opp_words,
            "num_opp_words": num_opp_words,
            "opp_word_sum": opp_word_sum,
            "neutral_words": neutral_words,
            "num_neutral_words": num_neutral_words,
            "neutral_word_sum": neutral_word_sum,
            "ass_words": ass_words,
            "num_ass_words": num_ass_words,
            "ass_word_sum": ass_word_sum,
            "score": total_score
        }
            

    def hint(self, codenames_state: CodenamesState, weights: WeightingMetric):
        """Main function for generating a clue given a board state

        Args:
            codenames_state (CodenamesState): 
            weights (WeightingMetric): 

        Returns:
            clue: str, int
        """
        logging.info(f"Codenames state: {codenames_state}")
        self.update_words(codenames_state)

        my_threshold = 0.5
        my_categories = self.get_categories(self.my_words, threshold=my_threshold)
        while len(my_categories) < 1 and my_threshold >= MIN_THRESHOLD :
            my_threshold -= 0.01
            print("No hints found. Lowering threshold to: ", my_threshold)
            my_categories = self.get_categories(self.my_words, threshold=my_threshold)

        if len(my_categories) < 1:
            logging.info(f"No categories above threhsold {MIN_THRESHOLD}. Falling to GPT-3")
            return self.get_gpt_dsp_hint(codenames_state)
        
        logging.info(f"Number of potential hints: {len(my_categories)}")

        opp_categories_stretch = self.get_categories(self.opp_words, self.stretch_threshold)
        neutral_categories_stretch = self.get_categories(self.neutral_words, self.stretch_threshold)
        assasin_categories_stretch = self.get_categories([self.assasin_word], self.stretch_threshold)
        
        category_infos = [self.generate_category_info(c, weights, my_categories, opp_categories_stretch, neutral_categories_stretch, assasin_categories_stretch) for c in my_categories]
        category_infos = sorted(category_infos, key=lambda ci:ci["score"], reverse=True)
        
        logging.info("Prefiltered categories:")
        for ci in category_infos:
            logging.info(f"{(ci['category'], ci['num_my_words'], ci['score'])}\n")

        category_infos = [ci for ci in category_infos if self.valid_hint(ci, codenames_state)]
        logging.info("Post-filtered categories:")
        for ci in category_infos:
            logging.info("\n")
            logging.info(f"{ci}")
        
        if len(category_infos) < 1:
            logging.info(f"No categories remaining. Falling to GPT-3")
            return self.get_gpt_dsp_hint(codenames_state)

        best_category = category_infos[0]
        logging.info(f"Best category {best_category}")
        logging.info(f"Returning hint {(best_category['category'], best_category['num_my_words'])}")
        
        self.last_hint = best_category["category"]
        return best_category["category"], best_category["num_my_words"]
        

            





        
        
    

