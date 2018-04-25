import copy
import time

from train.train import train_model


class PolicyImprover(object):
    def __init__(self, champion, championship_rounds):
        self.champion = champion
        self.championship_rounds = championship_rounds

    def train_model(self, new_games):
        pol_model = copy.copy(self.champion.current_policy)
        val_model = copy.copy(self.champion.current_value)
        return train_model(pol_model = pol_model, val_model = val_model, games=new_games, min_num_games=self.championship_rounds)

    def improve_policy(self, games, pool):
        start = time.time()
        new_policy, new_value = self.train_model(games)
        #self.champion.test_candidate(new_policy, new_value, pool)
        print("Improved policy in: {}".format(time.time() - start))
