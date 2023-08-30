import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from modules.graph import measure_strength

class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3


class Food:
    def __init__(self):
        self.position = None
        self.level = None



class Player:
    def __init__(self, i):
        self.id = i
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None

    def setup(self, position, level, field_size):
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class ForagingEnv(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "position", "game_over", "sight", "current_step",'reward'],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        type
    ):


        if type == "easy":
            players = [2,3]
            player_level = [1,2]
            foods = [1,2]
            food_level = [1,2]
            sight = 1
            field_size = (10,10)
            max_episode_steps = 50
            force_coop = False
            normalize_reward = False

        elif type == 'medium':
            players = [2, 2, 3]
            player_level = [0, 1, 2,]
            foods = [1, 1]
            food_level = [1, 2]
            field_size = (12,12)
            sight = 1
            max_episode_steps = 80
            force_coop = False
            normalize_reward = False

        elif type == 'hard':
            players = [2, 2, 3]
            player_level = [0, 1, 2,]
            foods = [1, 1]
            food_level = [1, 2]
            field_size = (15,15)
            sight = 1
            max_episode_steps = 100
            force_coop = False
            normalize_reward = False
        else:
            raise ValueError("Invalid type")



        grid_observation=True

        self.logger = logging.getLogger(__name__)
        self.seed()
        self.players = [Player(i) for i in range(sum(players))]
        self.player_list = [a * [player_level[i]] for i, a in enumerate(players)]
        self.player_list = [item for sublist in self.player_list for item in sublist]
        self.players_id_one_hot = np.eye(sum(players))

        self.foods = [Food() for _ in range(sum(foods))]
        self.food_list = [a * [food_level[i]] for i, a in enumerate(foods)]
        self.food_list = [item for sublist in self.food_list for item in sublist]

        self.field = np.zeros(field_size, np.int32)

        self.max_player_level = max(player_level)+1


        self._food_spawned = 0.0
        self.sight = sight
        self.force_coop = force_coop

        self._game_over = None
        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward
        self._grid_observation = grid_observation

        self.n_agents = len(self.players)

        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(6)] * len(self.players)))
        self.observation_space = gym.spaces.Tuple(tuple([self._get_observation_space()] * len(self.players)))

        self.viewer = None
        self.current_step = 0



        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        if not self._grid_observation:
            field_x = self.field.shape[1]
            field_y = self.field.shape[0]
            # field_size = field_x * field_y


            max_food = len(self.food_list)
            max_food_level = max(self.food_list) * len(self.players)

            min_obs = [-1, -1, 0] * max_food + [0, 0, 1] * len(self.players)
            max_obs = [field_x, field_y, max_food_level] * max_food + [
                field_x,
                field_y,
                max(self.player_list),
            ] * len(self.players)
        else:
            # grid observation space
            grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)


            loc_min = np.zeros((self.field.shape[1] + self.n_agents, 2), dtype=np.float32).reshape(-1,1)
            loc_max = np.ones((self.field.shape[1] + self.n_agents, 2), dtype=np.float32).reshape(-1,1)

            # loc_min = np.zeros((self.field.shape[1] + self.max_player_level, 2), dtype=np.float32).reshape(-1,1)
            # loc_max = np.ones((self.field.shape[1] + self.max_player_level, 2), dtype=np.float32).reshape(-1,1)

            # agents_min = np.zeros(grid_shape, dtype=np.float32).reshape(-1, 1)
            # agents_max = np.ones(grid_shape, dtype=np.float32).reshape(-1, 1) * max(self.player_list)
            # foods layer: foods level
            max_food_level = max(self.food_list)
            foods_min = np.zeros(grid_shape, dtype=np.float32).reshape(-1,1)
            foods_max = np.ones(grid_shape, dtype=np.float32).reshape(-1,1) * max_food_level

            # access layer: i the cell available
            access_min = np.zeros(grid_shape, dtype=np.float32).reshape(-1,1)
            access_max = np.ones(grid_shape, dtype=np.float32).reshape(-1,1)

            # total layer
            min_obs = np.concatenate([loc_min, foods_min, access_min])
            max_obs = np.concatenate([loc_max, foods_max, access_max])

        return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

    @classmethod
    def from_obs(cls, obs):
        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.level, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(players, None, None, None, None)
        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action) or player.level == 0
            ]
            for player in self.players
        }


    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col):
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def spawn_food(self):

        for i, food in enumerate(self.foods):
            attempts = 0
            while attempts < self.field_size[0]**2:
                row = self.np_random.randint(1, self.rows - 1)
                col = self.np_random.randint(1, self.cols - 1)

                # check if it has neighbors:
                if (
                    self.neighborhood(row, col).sum() > 0
                    or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                    or not self._is_empty_location(row, col)
                ):
                    attempts += 1
                else:
                    self.field[row, col] = self.food_list[i]
                    food.level = self.food_list[i]
                    food.position = (row, col)
                    break

        self._food_spawned = self.field.sum()



    def _is_empty_location(self, row, col):
        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False
        return True

    def spawn_players(self):

        spec_player_index = []
        for i, player in enumerate(self.players):
            attempts = 0
            player.reward = 0

            while attempts < self.field_size[0]**2:
                row = self.np_random.randint(0, self.rows)
                col = self.np_random.randint(0, self.cols)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        self.player_list[i],# self.np_random.randint(1, player_level),
                        self.field_size,
                    )
                    if self.player_list[i]== 0:
                        spec_player_index.append(i)
                    break
                attempts += 1

        for i, player_index in enumerate(spec_player_index):
            self.players[player_index].position = self.foods[i].position
            #self.completed_agent += 1



    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                and self.field[player.position[0], player.position[1] + 1] == 0
            )
        elif action == Action.LOAD:
            return self.adjacent_food(*player.position) > 0

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))


    def _make_obs(self, player):
        return self.Observation(
            actions = self._valid_actions[player],
            position = player.position,
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
            reward= player.reward,

        )


    def _make_gym_obs(self):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            # obs[: observation.field.size] = observation.field.flatten()

            # self player is always first
            self_players = [observation.position]

            for i in range(len(self.food_list)):
                obs[3 * i] = -1
                obs[3 * i + 1] = -1
                obs[3 * i + 2] = 0

            for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
                obs[3 * i] = y
                obs[3 * i + 1] = x
                obs[3 * i + 2] = observation.field[y, x]

            for i in range(len(self.players)):
                obs[self.food_list * 3 + 3 * i] = -1
                obs[self.food_list * 3 + 3 * i + 1] = -1
                obs[self.food_list * 3 + 3 * i + 2] = 0

            for i, p in enumerate(self_players):
                obs[self.food_list * 3 + 3 * i] = p.position[0]
                obs[self.food_list * 3 + 3 * i + 1] = p.position[1]
                # obs[self.food_list * 3 + 3 * i + 2] = p.level

            return obs

        def make_global_grid_arrays():
            """
            Create global arrays for grid observation space
            """
            grid_shape_x, grid_shape_y = self.field_size
            grid_shape_x += 2 * self.sight
            grid_shape_y += 2 * self.sight
            grid_shape = (grid_shape_x, grid_shape_y)

            self.agents_layer = np.zeros(grid_shape, dtype=np.float32)
            for player in self.players:
                player_x, player_y = player.position
                self.agents_layer[player_x + self.sight, player_y + self.sight] = player.level
            
            foods_layer = np.zeros(grid_shape, dtype=np.float32)
            foods_layer[self.sight:-self.sight, self.sight:-self.sight] = self.field.copy()
            for food in self.foods:
                food_x, food_y = food.position
                foods_layer[food_x + self.sight, food_y + self.sight] = food.level


            access_layer = np.ones(grid_shape, dtype=np.float32)
            # out of bounds not accessible
            access_layer[:self.sight, :] = 0.0
            access_layer[-self.sight:, :] = 0.0
            access_layer[:, :self.sight] = 0.0
            access_layer[:, -self.sight:] = 0.0
            #agent locations are not accessible
            for player in self.players:
                player_x, player_y = player.position
                access_layer[player_x + self.sight, player_y + self.sight] = 0.0

            #food locations are not accessible
            foods_x, foods_y = self.field.nonzero()
            for x, y in zip(foods_x, foods_y):
                access_layer[x + self.sight, y + self.sight] = 0.0
            
            return np.stack([self.agents_layer, foods_layer, access_layer])

        def get_agent_grid_bounds(agent_x, agent_y):
            return agent_x, agent_x + 2 * self.sight + 1, agent_y, agent_y + 2 * self.sight + 1



        def get_agent_position(position, level):

            # start_x, end_x, start_y, end_y = bounds
            # agent_layer = self.agents_layer[start_x:end_x, start_y:end_y].copy()
            # np.where(agent_layer!=level, 0, agent_layer)
            # grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)
            # agent_layer = np.zeros(grid_shape, dtype=np.float32)
            # agent_layer = agent_layer.flatten()


            # (agent_x, agent_y) = position
            one_hot_id = np.zeros(self.max_player_level, dtype=np.float32)
            one_hot_id[level] = 1

            # one_hot_x = np.zeros((self.field_size[0]), dtype=np.float32)
            # one_hot_x[agent_x] = 1
            # one_hot_y = np.zeros((self.field_size[1]), dtype=np.float32)
            # one_hot_y[agent_y] = 1

            #return np.concatenate([one_hot_x, one_hot_y])

            return one_hot_id
        
        def get_player_reward(obs):
                return obs.reward


        observations = [self._make_obs(player) for player in self.players]
        # if self._grid_observation:
        layers = make_global_grid_arrays()
        agents_bounds = [get_agent_grid_bounds(*player.position) for player in self.players]
        nobs = tuple([layers[:, start_x:end_x, start_y:end_y].reshape(-1, 1) for start_x, end_x, start_y, end_y in agents_bounds])
        # agents_obs = [get_agent_position(player.position, player.level).reshape(-1,1) for index, player in enumerate(self.players)]
        # nobs = [np.concatenate((raw_nobs[i], agents_obs[i])) for i in range(len(raw_nobs))]

        # else:
        #     nobs = tuple([make_obs_array(obs) for obs in observations])
        nreward = [get_player_reward(obs) for obs in observations]
        ndone = [obs.game_over for obs in observations]
        # ninfo = [{'observation': obs} for obs in observations]
        ninfo = {'num_collisions': self.num_collisions,
                 'completed_agent': self.completed_agent_list}

        return nobs, nreward, ndone, ninfo

    def reset(self):
        self.completed_agent = 0
        self.field = np.zeros(self.field_size, np.int32)
        #print(self.current_step)

        self.num_collisions = 0
        self.completed_agent_list = [0 for _ in range(self.n_agents)]


        self.spawn_food()
        self.spawn_players()


        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        nobs, _, _, _ = self._make_gym_obs()
        return nobs

    def step(self, actions):


        self.current_step += 1

        for p in self.players:
            p.reward = 0

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        loading_players = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)


        # and do movements for non colliding players
        for k, v in collisions.items():
            if len(v) > 1: # make sure no more than an player will arrive at location
                self.num_collisions += 1
                for paly in v:
                    paly.reward = -0.5
            else:
                v[0].position = k
                #v[0].reward = -0.1

        # finally process the loadings:
        while loading_players:
            # find adjacent food
            player = loading_players.pop()
            frow, fcol = self.adjacent_food_location(*player.position)
            food = self.field[frow, fcol]
            # if food != 0:
            #     for f in self.foods:
            #         if f.position == (frow, fcol):
            #             food = f.level
            #             break


            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [
                p for p in adj_players if p in loading_players or p is player
            ]

            adj_player_level = sum([a.level for a in adj_players])

            loading_players = loading_players - set(adj_players)

            if adj_player_level < food:
                # failed to load
                continue

            # else the food was loaded and each player scores points
            for a in adj_players:
                a.reward = float(a.level * food)
                # if self._normalize_reward:
                #     a.reward = a.reward / float(adj_player_level * self._food_spawned)
                a.level = 0
                a.position = (frow, fcol)



        # self._game_over = (self.completed_agent == self.n_agents or self._max_episode_steps <= self.current_step)
        self._game_over = (self._max_episode_steps <= self.current_step)

        self._gen_valid_moves()


        for index, p in enumerate(self.players):
            if p.level == 0:
                self.completed_agent_list[index] = 1
            p.score += p.reward

        return self._make_gym_obs()

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()



    def get_graph(self):

        G = nx.Graph()
        G.add_nodes_from([i for i in range(self.n_agents)])

        # for i in range(self.ncar):
        #     G.add_node(i, feature = np.array(self.obs[i]))

        for i in range (self.n_agents):
            for j in range (self.n_agents):
                if i != j:
                    if np.linalg.norm(np.array(self.players[i].position) - np.array(self.players[j].position)) <= 3.0:
                        G.add_edge(i, j)
                    # if self.players[i].level == self.players[j].level or \
                    #     np.linalg.norm(np.array(self.players[i].position) - np.array(self.players[j].position))<=2.0:
                    #         G.add_edge(i,j)
        # nx.draw(G, with_labels=True, node_color='#A0CBE2', edge_color='#A0CBE2', node_size=100, width=1)
        # plt.show()



        # g = nx.Graph()
        # g.add_nodes_from(G.nodes(data=False))
        #
        # for e in G.edges():
        #     strength = measure_strength(G, e[0], e[1])
        #     print(strength)
        #     if strength > 0.5:
        #         g.add_edge(e[0], e[1])
        #
        # #set = [list(c) for c in nx.connected_components(g)]
        #
        # subax1 = plt.subplot(121)
        # nx.draw(G, with_labels=True, node_color='#A0CBE2', edge_color='#A0CBE2', node_size=100, width=1)
        # subax2 = plt.subplot(122)
        # nx.draw(g, pos=nx.spring_layout(g), with_labels=True, node_color='#A0CBE2', edge_color='#A0CBE2',
        #         node_size=100, edge_cmap=plt.cm.Blues, width=1)
        # plt.show()
        return G
