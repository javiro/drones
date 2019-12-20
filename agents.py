import numpy as np
import random
import matplotlib.pyplot as plt


def plot_states(sli, sl):
    axes_initial = plt.subplot(211)
    plt.imshow(sli)
    axes_end = plt.subplot(212)
    plt.imshow(sl)
    axes_initial.set_ylabel('Initial')
    axes_end.set_ylabel('End')
    plt.show()


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
            self.strategy = np.random.randint(num_of_channels)
        else:
            self.strategy = strategy

    def set_strategy(self):
        player = np.zeros(self.num_of_channels)
        player[self.strategy] = 1
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
            self.strategy = strategy
            trials = []
            for trial in range(game.n_of_trials):
                trials.append(game.play_drone_game(self, population.get_player()))
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

    def __init__(self, n_of_agents, num_of_channels, random_initial_condition='ON'):
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

    def get_initial_condition(self, random_initial_condition):
        if random_initial_condition == 'ON':
            random_initial_distribution = []
            return random_initial_distribution

        elif sum(random_initial_condition) == self.n_of_agents:
            random_initial_distribution = random_initial_condition
            return random_initial_distribution

    def populate_group(self):
        population = []
        if self.random_initial_condition == 'ON':
            for i in range(self.n_of_agents):
                player = Drone(i, self.num_of_channels)
                population.append(player)
        else:
            ids = random.sample(list(range(self.n_of_agents)), self.n_of_agents)
            for s in range(self.num_of_channels):
                for i in range(self.initial_condition[s]):
                    player = Drone(ids.pop(), self.num_of_channels, s)
                    population.append(player)
        return population

    def get_player(self):
        return np.random.choice(self.population)

    def get_revising_population(self, prob_revision):
        sample_size = int(prob_revision * self.n_of_agents)
        if (sample_size == 0) & (random.random() < prob_revision):
            sample_size = 1
        revising_population = random.sample(list(range(self.n_of_agents)), sample_size)
        return [self.population[p] for p in revising_population]

    def get_strategy_distribution(self):
        strategies = [player.strategy for player in self.population]
        distribution = np.histogram(strategies, bins=list(range(self.num_of_channels + 1)))[0]
        plt.show()
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
                 prob_revision=0.001, n_of_revisions_per_tick=10, n_of_trials=10, use_prob_revision='OFF'):
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
        self.payoff_matrix = self.get_payoff_matrix()
        self.drones = DronePopulation(self.n_of_agents, self.num_of_channels, self.random_initial_condition)

    def get_payoff_matrix(self):
        n = self.num_of_channels
        payoff_matrix = np.zeros((n, n))
        for i in range(n):
            payoff_matrix[i, i] = i + 1
        return payoff_matrix

    def play_drone_game(self, player_1_instance, player_2_instance):
        player_1 = player_1_instance.set_strategy()
        player_2 = player_2_instance.set_strategy()
        return player_1 @ self.payoff_matrix @ player_2

    def get_test_strategies(self, player_instance):
        return [*[player_instance.strategy], *random.sample(list(range(self.num_of_channels)), self.n_of_candidates - 1)]

    def update_strategies(self):
        """
        Under the best experienced payoff protocol, a revising agent tests each of the 'n_of_candidates' of strategies
        against a random agent, with each play of each strategy being against a newly drawn opponent. The revising agent
        then selects the strategy that obtained the greater payoff in the test, with ties resolved at random.

        :param game:
        :return:
        """

        sample_size = int(self.prob_revision * self.n_of_agents)
        if (sample_size == 0) & (random.random() < self.prob_revision):
            sample_size = 1
        revising_population = random.sample(list(range(self.n_of_agents)), sample_size)

        for player_1 in revising_population:
            self.drones.population[player_1].update_strategy(self.drones, self)

    def simulate_drone_game(self):
        """
        Under the best experienced payoff protocol, a revising agent tests each of the 'n_of_candidates' of strategies
        against a random agent, with each play of each strategy being against a newly drawn opponent. The revising
        agent then selects the strategy that obtained the greater payoff in the test, with ties resolved at random.
        :return:
        """
        for g in range(1, self.game_rounds):
            self.update_strategies()
            if g % 10 == 0:
                print("Round {}: {}".format(g, self.drones.get_strategy_distribution()))
        return self.drones.get_strategy_distribution()


def main():
    game_rounds = 500
    num_of_channels = 5
    n_of_agents = 200
    n_of_candidates = 5
    random_initial_condition = [200, 0, 0, 0, 0]
    prob_revision = 0.2
    n_of_revisions_per_tick = 10
    n_of_trials = 1
    use_prob_revision = 'OFF'

    g = DroneGame(game_rounds,
                  num_of_channels,
                  n_of_agents,
                  n_of_candidates,
                  random_initial_condition,
                  prob_revision,
                  n_of_revisions_per_tick,
                  n_of_trials,
                  use_prob_revision)

    print(g.drones.get_strategy_distribution())
    g.simulate_drone_game()
    print(g.drones.get_strategy_distribution())


if __name__ == '__main__':
    main()
