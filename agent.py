import itertools
import random
from collections import deque

import torch
import torch.nn.functional as F
from torch import nn, optim

import dnbpy


class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, h2_nodes, out_actions) -> None:
        """
        (Feed-forward) Neural Network with 2 hidden layers
        Input Layer: Edge States (boolean) + Box States (-1, 0, 1)
        Hidden Layers:
           first hidden layer: Edge States + Box States + Current Score
           second hidden layer: Edge States + Box States + Current Score
        Output Layer: No. of edges (which edge to select next)
        """
        super().__init__()

        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.h1 = nn.Linear(h1_nodes, h2_nodes)
        self.h2_out = nn.Linear(h2_nodes, out_actions)

    # feed an input into the network
    def forward(self, x):
        # print(f"Input tensor type: {x.type()}, shape: {x.shape}")

        x = F.relu(self.fc1(x))  # Rectified Linear Unit Activation
        # print(f"After fc1 layer: {x.type()}, shape: {x.shape}")

        x = F.relu(self.h1(x))
        # print(f"After h1 layer: {x.type()}, shape: {x.shape}")

        x = self.h2_out(x)
        # print(f"After h2_out layer: {x.type()}, shape: {x.shape}")

        return x


class ReplayMemory:
    def __init__(self, replay_memory_size) -> None:
        self.memory = deque([], maxlen=replay_memory_size)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def first_n(self, n):
        return list(itertools.islice(self.memory, 0, n))

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    # (Static) Hyperparameters
    learning_rate = 0.008
    discount_factor = 0.7
    network_sync_rate = 4
    replay_memory_size = 64
    mini_batch_size = 4

    # NN
    loss_fn = nn.MSELoss()
    # optimizer = None

    def __init__(self, board_size, h1_nodes=128, h2_nodes=128) -> None:
        self.board_size = board_size
        m, n = board_size
        n_actions = n * (m - 1) * 2 + m * (n - 1) * 2

        # No. of edges + 2 (1 for each player's score)
        self.in_states = n_actions + 2
        self.out_actions = n_actions

        # Policy and Target Network Initialisations
        self.policy_net = DQN(self.in_states, h1_nodes, h2_nodes, self.out_actions)
        self.target_net = DQN(self.in_states, h1_nodes, h2_nodes, self.out_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(self.replay_memory_size)
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9999

        self.bot_player_name = "dqn"
        self.op_player_name = "$L1"

    # Function Equivalences:
    # initialisation:       reset() = dbnpy.Game(board_size, players)
    # actions:              act(state) = select_edge(self, board_state, score, opp_score)
    # state transitions:    step(state, action) = game.select_edge(move, current_player)
    # store transitions:    self.memory.append()
    # replay:               self.replay()
    # sync network:         -

    # Vectorize game state to be fed into DQN
    def state_to_dqn_input(self, game):
        """
        Converts a game state into a tensor. This tensor is fed into the Deep Q Network.
        Example:
            Given n_edges, player_score, opponent_score
        Return: tensor([*n_edges, player_score, opponent_score])
        """
        state = torch.zeros(self.in_states, dtype=torch.float)
        state[: self.out_actions] = torch.FloatTensor(game.get_board_state())
        # The the second last scalar should be the bot's score and the last should be the opponent's
        state[-2] = game.get_score(self.bot_player_name)
        state[-1] = game.get_score(self.op_player_name)
        return state

    # Done AI agent's moves
    # act(self, state)
    def select_edge(self, game):
        # Epsilon-greedy: randomly select a state if below epsilon threshold
        if random.random() < self.epsilon:
            move = random.sample(game.get_legal_moves(), 1)[0]
        # Exploitation: select best action
        else:
            with torch.no_grad():
                q_vals = self.policy_net(self.state_to_dqn_input(game))
                move = torch.argmax(q_vals).item()

        return move

    # Reward System:
    # Winning has the highest reward
    # Chaining (i.e next_player = current_player) has decent reward
    # Making a box is equivalent to chaining;
    #   however a single move can make multiple boxes
    #   must reward more boxes accordingly
    #
    # Note: Agent may make illegal moves during exploitation
    def is_winner(self, game):
        dqnbot_score = game.get_score(self.bot_player_name)
        # note: game_get_next_player returns the next opponent
        op_score = game.get_score(self.op_player_name)
        if dqnbot_score > op_score:
            return 1
        elif op_score > dqnbot_score:
            return -1
        else:
            return 0

    def step(self, game, move, current_player):
        try:
            next_player, boxes_made = game.select_edge(move, current_player)
            done = game.is_finished()

            reward = 2 + boxes_made * 10

            w = self.is_winner(game)

            if done:
                if w == 1:
                    reward += 100  # W
                elif w == -1:
                    reward -= 100  # L
                else:
                    reward += 50  # Tie
            else:
                if next_player == current_player:
                    reward += 20  # for creating a chain

        # if the bot violates a rule
        except Exception as e:
            print(e)
            # game.terminate()
            # done = True
            done = False
            reward = -10000000000

        next_state = self.state_to_dqn_input(game)
        return next_state, reward, done

    def update(self, action_tuple):
        state, action, reward, next_state, done = action_tuple

        q_value = self.policy_net(state)
        next_q_value = self.target_net(next_state)

        target_q_value = reward + self.discount_factor * next_q_value * (~done)

        loss = self.loss_fn(q_value, target_q_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def replay(self):
        # 5. TODO Sample a mini batch from the replay buffer
        # batch = self.memory.sample(self.mini_batch_size)
        batch = self.memory.first_n(self.mini_batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        # print(f"states shape: {states.shape}")

        actions = torch.tensor(actions, dtype=torch.float).unsqueeze(1)
        # print(f"actions shape: {actions.shape}")

        rewards = torch.tensor(rewards, dtype=torch.int).unsqueeze(1)
        # print(f"rewards shape: {rewards.shape}")

        next_states = torch.stack(next_states)
        # print(f"next_states shape: {next_states.shape}")

        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)
        # print(f"dones shape: {dones.shape}")

        # q_values = self.policy_net(states).gather(1, actions)
        q_values = self.policy_net(states).max(1)[0].unsqueeze(1)
        # print(f"q_values shape: {q_values.shape}")

        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        # print(f"next_q_values shape: {next_q_values.shape}")

        target_q_values = rewards + self.discount_factor * next_q_values * (~dones)
        # print(f"target_q_values shape: {target_q_values.shape}")

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes, render=False):
        # create environment instance
        board_size = self.board_size
        games_finished = 0
        games_won = 0

        players = [self.op_player_name, self.bot_player_name]
        # n_edges = cols * (rows - 1) * 2 + rows * (cols - 1) * 2

        # training loop
        for episode in range(episodes):
            # TODO deep learning loop

            # 1. initialise a new game at each step
            game = dnbpy.Game(board_size, players)
            computer_players = {
                "$random": dnbpy.RandomPolicy(),
                "$L1": dnbpy.Level1HeuristicPolicy(board_size=board_size),
                "$L2": dnbpy.Level2HeuristicPolicy(board_size=board_size),
                "$L3": dnbpy.Level3MinimaxPolicy(
                    board_size=board_size, depth=None, update_alpha=True
                ),
            }
            total_reward = 0
            steps = 0

            while not game.is_finished() and not game.is_terminated():
                current_player = game.get_current_player()
                opp_player = game.get_next_player()

                # -----------------
                if current_player in computer_players:
                    # Computer's turn
                    move = computer_players[current_player].select_edge(
                        game.get_board_state(),
                        game.get_score(current_player),
                        game.get_score(opp_player),
                    )
                    # print("player %s selects edge %s" % (current_player, move))
                    game.select_edge(move, current_player)
                else:
                    # Agent's turn
                    # try:
                    # 2. Select an action
                    state = self.state_to_dqn_input(game)
                    move = self.select_edge(game)
                    # print("player %s selects edge %s" % (current_player, move))

                    # 3. perform the action and receive next_state, reward, terminated=True/False
                    next_state, reward, done = self.step(game, move, current_player)
                    total_reward += reward

                    # Without Replay Learning
                    self.update(action_tuple=(state, move, reward, next_state, done))
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    self.target_net.eval()

                    """ Replay Learning
                    # 4. Store the transition in the replay buffer
                    self.memory.append((state, move, reward, next_state, done))

                    # 5. Replay when appropriate
                    if len(self.memory) >= self.mini_batch_size:
                        self.replay()

                    # 6. Sync the network at intervals
                    steps += 1
                    if steps % self.network_sync_rate == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                        self.target_net.eval()
                    """
                # except Exception as e:
                #     print(e)

            # print(game)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if game.is_finished():
                games_finished += 1
                winner = self.is_winner(game)
                if winner == 1:
                    games_won += 1
            print(
                f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon}"
            )

        return (games_finished, games_won)

    def save_model(self, path="dqn_model.pth"):
        saved = {
            "epsilon": self.epsilon,
            "optimizer": self.optimizer.state_dict(),
            "target_net": self.target_net.state_dict(),
            "policy_net": self.policy_net.state_dict(),
            # "memory": "",
        }
        torch.save(saved, path)

    def load_model(self, path="dqn_model.pth"):
        loaded = torch.load(path)
        self.epsilon = loaded["epsilon"]
        self.optimizer.load_state_dict(loaded["optimizer"])
        self.target_net.load_state_dict(loaded["target_net"])
        self.policy_net.load_state_dict(loaded["policy_net"])


if __name__ == "__main__":
    board_size = (3, 3)  # Example board size
    agent = DQNAgent(board_size)
    completed, won = agent.train(episodes=10000)
    print("Completed Games: ", completed)
    print("Won Games: ", won)
    agent.save_model()
