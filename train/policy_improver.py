import copy
import time

from train.self_challenge import Champion
from train.train import train_model

'''
Given an old policy and some games generated through self-play, the Policy Improver improves the old policy using the generated games, by first training on the games provided by GameGenerator, retraining (or "fine-tuning") the old new policy, then finally evaluating the best player in a tournament-style championship.
'''

class PolicyImprover(object):
    def __init__(self, champion: Champion, championship_rounds):
        self.champion = champion
        self.championship_rounds = championship_rounds

    def train_model(self, new_games):
        model = copy.copy(self.champion.current_policy)
        return train_model(model=model, games=new_games, min_num_games=self.championship_rounds)

    def improve_policy(self, games, pool):
        start = time.time()
        new_policy = self.train_model(games)
        self.champion.test_candidate(new_policy, pool)
        print("Improved policy in: {}".format(time.time() - start))
