import random

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

sns.set(style="whitegrid")


class Drone(object):
    """
    Class which implements the agents in drone communication game.
    """

    def __init__(self, player_id, num_of_channels, strategy=None):
        """
        Parameters
        ----------

        """
        # Set internal parameters
        self.player_id = player_id
        self.num_of_channels = num_of_channels
        if strategy is None:
            self.strategy = random.randint(0, num_of_channels - 1)
        else:
            self.strategy = strategy

    def set_strategy(self, strategy):
        player = np.zeros(self.num_of_channels)
        player[strategy] = 1
        return player

    def update_strategy(self, population, game):
        """
        Under the best experienced payoff protocol, a revising agent tests each of the 'n_of_candidates' of strategies
        against a random agent, with each play of each strategy being against a newly drawn opponent. The revising agent
        then selects the strategy that obtained the greater payoff in the test, with ties resolved at random.
        :param population:
        :param game:
        :return:
        """

        games = []
        n_of_candidates = game.get_test_strategies(self)
        for strategy in n_of_candidates:
            trials = []
            self.strategy = strategy
            for trial in range(game.n_of_trials):
                player_2 = population.get_player(self)
                trials.append(game.play_drone_game(self.set_strategy(strategy),
                                                   player_2.set_strategy(player_2.strategy)))
            games.append(max(trials))
        games = np.array(games)
        self.strategy = n_of_candidates[random.choice(np.where(games == np.max(games))[0])]


class DronePopulation(object):
    """
    Class which implements the populations of players.

    Formulating such a model requires one to specify

        (i) the number of agents N in the population,
        (ii) the n-strategy normal form game the agents are recurrently matched to play,
        (iii) the rule describing how revision opportunities are assigned to the agents, and
        (iv) the protocol according to which agents revise their strategies when opportunities
        to do so arise.

    """

    def __init__(self, n_of_agents, num_of_channels, random_initial_condition='ON', consider_imitating_self=True):
        """
        Parameters
        ----------
        n_of_agents : int
            Size of the populations

        """
        # Set internal parameters
        self.n_of_agents = n_of_agents
        self.num_of_channels = num_of_channels
        self.random_initial_condition = random_initial_condition
        self.initial_condition = self.get_initial_condition(random_initial_condition)
        self.population = self.populate_group()
        self.consider_imitating_self = consider_imitating_self

    def get_initial_condition(self, random_initial_condition):
        if random_initial_condition == 'ON':
            random_initial_distribution = []
            return random_initial_distribution

        elif sum(random_initial_condition) == self.n_of_agents:
            assert len(random_initial_condition) == self.num_of_channels
            random_initial_distribution = random_initial_condition
            return random_initial_distribution

    def populate_group(self):
        if self.random_initial_condition == 'ON':
            population = [Drone(i, self.num_of_channels) for i in range(self.n_of_agents)]
        else:
            ids = random.sample(list(range(self.n_of_agents)), self.n_of_agents)
            population = [Drone(ids.pop(), self.num_of_channels, s)
                          for s in range(self.num_of_channels) for i in range(self.initial_condition[s])]
        return population

    def get_player(self, player_1):
        """
        Returns an opponent avoiding the play of a player with himself.
        :param player_1:
        :return:
        """
        # player_2 = np.random.choice(self.population)
        player_2 = self.population[random.randint(0, len(self.population) - 1)]
        if self.consider_imitating_self:
            return player_2
        else:
            while player_2 == player_1:
                # player_2 = np.random.choice(self.population)
                player_2 = self.population[random.randint(0, len(self.population) - 1)]
            return player_2

    def get_revising_population(self, prob_revision):
        sample_size = int(prob_revision * self.n_of_agents)
        if (sample_size == 0) & (random.random() < prob_revision):
            sample_size = 1
        revising_population = random.sample(list(range(self.n_of_agents)), sample_size)
        return (self.population[p] for p in revising_population)

    def get_strategy_distribution(self):
        strategies = [player.strategy for player in self.population]
        distribution = np.histogram(strategies, bins=list(range(self.num_of_channels + 1)))[0]
        return distribution


class DroneGame(object):
    """
    Class which implements the communication between drones within the frame of evolutionary game theory .

    Formulating such a model requires one to specify

        (i) the number of agents N in the population,
        (ii) the n-strategy normal form game the agents are recurrently matched to play,
        (iii) the rule describing how revision opportunities are assigned to the agents, and
        (iv) the protocol according to which agents revise their strategies when opportunities
        to do so arise.

    """

    def __init__(self, game_rounds, num_of_channels, n_of_agents, n_of_candidates, random_initial_condition,
                 prob_revision=0.001, n_of_revisions_per_tick=10, n_of_trials=10, use_prob_revision='ON',
                 mean_dynamics='OFF', ticks_per_second=5, consider_imitating_self=True, payoff_matrix=None):
        """
        Complete matching is off since BEP does not consider it. Then the agents play his current strategy against a
        random sample of opponents. The size of this sample is specified by the parameter n-of-trials.
        Single sample is off, so the agent tests each of his candidate strategies against distinct, independent samples
        of n-of-trials opponents.

        Parameters
        ----------

        :param game_rounds:
        :param num_of_channels:
        :param n_of_agents:
        :param n_of_candidates: determines the total number of strategies the revising agent considers. The revising
            agent’s current strategy is always part of the set of candidates.
        :param prob_revision: defines the probability that an agent is assigned an opportunity of revision.
        :param n_of_revisions_per_tick: if use_prob_revision is off, this parameter defines the number of revising
            agents.
        :param n_of_trials: specifies the size of the sample of opponents to test the strategies with.
        :param use_prob_revision: defines the assignment of revision opportunities to agents. If it is on, then
            assignments are stochastic and independent.
        :param mean_dynamics:
        :param ticks_per_second: Number of ticks per second.
        :param consider_imitating_self:
        :param payoff_matrix:
        """
        # Set internal parameters
        self.game_rounds = game_rounds
        self.num_of_channels = num_of_channels
        self.n_of_agents = n_of_agents
        self.n_of_candidates = n_of_candidates
        self.random_initial_condition = random_initial_condition
        self.prob_revision = prob_revision
        self.n_of_revisions_per_tick = n_of_revisions_per_tick
        self.n_of_trials = n_of_trials
        self.use_prob_revision = use_prob_revision
        self.consider_imitating_self = consider_imitating_self
        self.payoff_matrix = self.get_payoff_matrix(payoff_matrix)
        self.mean_dynamics = mean_dynamics
        self.drones = DronePopulation(self.n_of_agents,
                                      self.num_of_channels,
                                      self.random_initial_condition,
                                      self.consider_imitating_self)
        self.ticks_per_second = ticks_per_second

    def get_payoff_matrix(self, payoff_matrix):
        n = self.num_of_channels
        if payoff_matrix is None:
            payoff_matrix = np.zeros((n, n))
            for i in range(n):
                payoff_matrix[i, i] = i + 1
            return payoff_matrix
        else:
            payoff_matrix = np.array(payoff_matrix)
            assert payoff_matrix.shape == (n, n)
            return np.array(payoff_matrix)

    def play_drone_game(self, player_1, player_2):
        return player_1 @ self.payoff_matrix @ player_2

    def get_test_strategies(self, player_instance):
        strategies = list(range(self.num_of_channels))
        strategies.remove(player_instance.strategy)
        strategies.insert(0, player_instance.strategy)
        return strategies

    def update_strategies(self):
        """
        Under the best experienced payoff protocol, a revising agent tests each of the 'n_of_candidates' of strategies
        against a random agent, with each play of each strategy being against a newly drawn opponent. The revising agent
        then selects the strategy that obtained the greater payoff in the test, with ties resolved at random.

        :param game:
        :return:
        """
        if self.use_prob_revision == 'ON':
            for player_1 in self.drones.population:
                if random.random() < self.prob_revision:
                    player_1.update_strategy(self.drones, self)
        else:
            revising_population = random.sample(self.drones.population, self.n_of_revisions_per_tick)
            for player_1 in revising_population:
                player_1.update_strategy(self.drones, self)

    def plot_distributions(self, g, plot_dist, ax):
        distribution = self.drones.get_strategy_distribution()
        plot_dist.append(distribution[::-1] / sum(distribution))
        df_plot_dist = pd.DataFrame(plot_dist)
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'][:len(distribution)]
        df_plot_dist.columns = ['c{}'.format(i) for i in range(len(df_plot_dist.columns))]
        plt.stackplot(df_plot_dist.index,
                      [df_plot_dist['{}'.format(c)].values for c in df_plot_dist.columns],
                      colors=colors)
        plt.title("Second {}".format(g / self.ticks_per_second))
        plt.draw()
        plt.pause(0.0001)
        print("Second {}: {}".format(g / self.ticks_per_second, distribution))

    def get_expectation_value(self):
        distribution = self.drones.get_strategy_distribution()
        # expectation = integral [x f_bar dx], f_bar = f / integral [f_dx]
        x = range(self.num_of_channels)
        integral_f_dx = sum(distribution * np.diff(range(self.num_of_channels + 1)))
        f_bar = distribution / integral_f_dx
        dx = np.diff(range(self.num_of_channels + 1))
        expectation = sum(x * f_bar * dx)
        return expectation

    def simulate_drone_game(self):
        """
        Under the best experienced payoff protocol, a revising agent tests each of the 'n_of_candidates' of strategies
        against a random agent, with each play of each strategy being against a newly drawn opponent. The revising
        agent then selects the strategy that obtained the greater payoff in the test, with ties resolved at random.
        :return:
        """
        plt.figure()
        length_x = self.game_rounds / self.ticks_per_second
        if self.mean_dynamics == 'OFF':
            ax = plt.axes(xlim=(0, length_x), ylim=(0, 1))
        else:
            ax = plt.axes()
        plt.xlabel("Seconds")
        plt.ylabel("Distribution")
        plot_dist = []
        mean_dynamic = []
        plt.ion()
        for g in range(self.game_rounds):
            self.update_strategies()
            if (g % self.ticks_per_second == 0) & (self.mean_dynamics == 'OFF'):
                self.plot_distributions(g, plot_dist, ax)
            else:
                expectation = self.get_expectation_value()
                mean_dynamic.append(expectation)

        if self.mean_dynamics == 'OFF':
            plt.show()
        else:
            plt.plot(mean_dynamic)
            plt.show(block=True)
        return self.drones.get_strategy_distribution(), plot_dist


def main():
    game_rounds = 200
    ticks_per_second = 5
    num_of_channels = 4
    n_of_agents = 200
    n_of_candidates = num_of_channels
    random_initial_condition = [200, 0, 0, 0]
    prob_revision = 0.2
    n_of_revisions_per_tick = 10
    n_of_trials = 1
    use_prob_revision = 'ON'
    consider_imitating_self = True
    mean_dynamics = 'ON'
    prisioner_matrix = [[-5, -1], [-10, -2]]
    penalti_matrix = [[0, 1], [1, 0]]
    flg = [[1, 2, 3], [4, 3, 4], [3, 2, 5]]
    coordination = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    g = DroneGame(game_rounds,
                  num_of_channels,
                  n_of_agents,
                  n_of_candidates,
                  random_initial_condition,
                  prob_revision,
                  n_of_revisions_per_tick,
                  n_of_trials,
                  use_prob_revision,
                  mean_dynamics,
                  ticks_per_second,
                  consider_imitating_self)

    print(g.drones.get_strategy_distribution())
    g.simulate_drone_game()
    print(g.drones.get_strategy_distribution())


if __name__ == '__main__':
    main()
