
from dataclasses import dataclass
from typing import List
from playgroundrl.client import CodenamesState

@dataclass(frozen=True, eq=True)
class WeightingMetric:
    num_my_words: float
    my_word_sum: float
    num_opp_words: float
    opp_word_sum: float 
    num_neutral_words: float
    neutral_word_sum: float
    num_ass_words: float
    ass_word_sum: float

@dataclass(frozen=True, eq=True)
class HintState:
    my_num_guessed: int
    opp_num_guessed: int
    neutral_num_guessed: int
    
num_my_words_ = [1] # constant reference point
num_opp_words_ = [-1, -1.5, -2] # opponent words can be equally, 1.5x, or twice as important as number of your own words
num_ass_words_ = [-10, -20] # assasin can be 10x or 20x as important as your own words 
sum_to_count_ratio_ = [.5, 1, 2] # ratio of importance between the count of words over the threshold for inclusion in a category, and the sum of their probabilities of being in that category
neutral_to_opp_ratio_ = [1, 0.5, 0.25] # ratio of importance between neutral words and opponent words (equal, half as bad, a quarter as bad)

possible_weighting_metrics = [] # actions
for num_my_words in num_my_words_:
    for num_opp_words in num_opp_words_:
        for num_ass_words in num_ass_words_:
            for sum_to_count_ratio in sum_to_count_ratio_:
                for neutral_to_opp_ratio in neutral_to_opp_ratio_:
                    possible_weighting_metrics.append(
                        WeightingMetric(
                            num_my_words = num_my_words, 
                            my_word_sum = num_my_words * sum_to_count_ratio,
                            num_opp_words = num_opp_words,
                            opp_word_sum = num_opp_words * sum_to_count_ratio,
                            num_neutral_words = neutral_to_opp_ratio * num_opp_words,
                            neutral_word_sum = neutral_to_opp_ratio * num_opp_words * sum_to_count_ratio,
                            num_ass_words = num_ass_words,
                            ass_word_sum = sum_to_count_ratio * num_ass_words
                        )
                    )
ACTIONS: List[WeightingMetric] = possible_weighting_metrics

def get_hint_state(codenames_state: CodenamesState) -> HintState:
    my_color = codenames_state.color
    opp_color = "BLUE" if my_color == "RED" else "RED"
    return HintState(codenames_state.guessed.count(my_color), codenames_state.guessed.count(opp_color), codenames_state.guessed.count("INNOCENT"))


REWARD_FOR_WINNING = 50

def reward(cur_hint_state: HintState, prev_hint_state: HintState) -> int:
    my_gain = (cur_hint_state.my_num_guessed - prev_hint_state.my_num_guessed)
    opp_gain = (cur_hint_state.opp_num_guessed - prev_hint_state.opp_num_guessed)
    return my_gain - opp_gain