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
        x = F.relu(self.fc1(x))  # Rectified Linear Unit Activation
        x = F.relu(self.h1(x))
        x = self.h2_out(x)
        return x


class ReplayMemory:
    def __init__(self, replay_memory_size) -> None:
        self.memory = deque([], maxlen=replay_memory_size)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    # (Static) Hyperparameters
    learning_rate = 0.001
    discount_factor = 0.7
    network_sync_rate = 10
    replay_memory_size = 512
    mini_batch_size = 32

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
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

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
        state = torch.zeros(self.in_states, dtype=torch.uint8)
        state[: self.out_actions] = torch.ByteTensor(game.get_board_state())
        state[-2] = game.get_score(game.get_current_player())
        state[-1] = game.get_score(game.get_next_player())
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
    # Note: Agent does not make illegal moves anyway

    def step(self, game, move, current_player):
        next_player, boxes_made = game.select_edge(move, current_player)
        done = game.is_finished()

        reward = boxes_made * 10

        if done:
            dqnbot_score = game.get_score(current_player)
            # note: game_get_next_player returns the next opponent
            op_score = game.get_score(game.get_next_player())

            if dqnbot_score > op_score:
                reward += 100  # W
            elif op_score > dqnbot_score:
                reward -= 100  # L
            else:
                reward += 50  # Tie
        else:
            if next_player == current_player:
                reward += 20  # for creating a chain

        next_state = self.state_to_dqn_input(game)
        return next_state, reward, done

    def replay(self):
        # 5. TODO Sample a mini batch from the replay buffer
        batch = self.memory.sample(self.mini_batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states)
        # actions = torch.ByteTensor(actions).unsqueeze(1)
        actions = torch.ByteTensor(actions)
        rewards = torch.IntTensor(rewards)
        next_states = torch.cat(next_states)
        dones = torch.BoolTensor(dones)

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + self.discount_factor * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes, render=False):
        # create environment instance
        board_size = self.board_size
        players = ["$L3", "dqn"]
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

            while not game.is_finished():
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
                    try:
                        # 2. Select an action
                        state = self.state_to_dqn_input(game)
                        move = self.select_edge(game)
                        # print("player %s selects edge %s" % (current_player, move))

                        # 3. perform the action and receive next_state, reward, terminated=True/False
                        next_state, reward, done = self.step(game, move, current_player)
                        total_reward += reward

                        # 4. Store the transition in the replay buffer
                        self.memory.append((state, move, reward, next_state, done))

                        # 5. Replay when appropriate
                        if len(self.memory) >= self.mini_batch_size:
                            self.replay()

                        # 6. Sync the network at intervals
                        steps += 1
                        if steps % self.network_sync_rate == 0:
                            self.target_net.load_state_dict(
                                self.policy_net.state_dict()
                            )
                            self.target_net.eval()
                    except Exception as e:
                        print(e)

            print(game)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

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
    agent.train(episodes=5)
    agent.save_model()