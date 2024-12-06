Training Environment Credits: lantunes/dnbpy

---

Dependencies:

- torch
- numpy


to train agent:
```sh
python agent.py
```

save output to a file:
```sh
python agent.py > file2.txt
```

---

# Training & Observations:

#### **Attempt 1:**

Terminates on Wrong Move: True

Legal Edge Selection (w Epsilon Greedy): False

Wrong Move Reward: -100000

Losing Reward: -100

Winning Reward: 100

Observations:
- Game always terminates with wrong move. Games never complete.
- Agent likely has no idea what valid moves even are.

#### **Attempt 2:**

Terminates on Wrong Move: True

Legal Edge Selection (w Epsilon Greedy): True

Wrong Move Reward: -50

Losing Reward: -100

Winning Reward: 100

+ Some games complete.
+ When epsilon reduces and exploitation starts to take place, the rewards begin to drop.
+ Again, Agent likely has no idea what valid moves even are, even though it learns something from random.

#### **Attempt 3:**

Terminates on Wrong Move: False

Legal Edge Selection (w Epsilon Greedy): True

Wrong Move Reward: -50

Losing Reward: -100

Winning Reward: 100

Replay Size: 512

Batch Size: 32

Sync Rate: 10

+ All Games run until completion.
+ Agent is trying to learn the correct moves but the batch and replay sizes are probably too big

#### **Attempt 4:**

Terminates on Wrong Move: False

Legal Edge Selection (w Epsilon Greedy): True

Wrong Move Reward: -50

Losing Reward: -100

Winning Reward: 100

Replay Size: 64

Batch Size: 4

Sync Rate: 4

#### **Attempt 5:**

Increase learning rate from 0.001 to 0.008,

- Similar convergence

#### **Attempt 6:**

Get First N memory samples instead of random N

+ Slightly Faster Convergence, overall, penalties seem to reduce during exploitation

#### **Attempt 7:**

Don't use replay learning

#### **Attempt 8:**

Reduce Penalty for wrong step

#### **Attempt 9:**

Near Infinite Penalty for wrong step

#### **Attempt 10:**

Reduce Epsilon Decay so that agent learns valid moves first

+ More Games Won

#### **Attempt 11:**

Let's see if this policy works. Using 10k episodes.

- 51/10k games won
- Agent does not learn the game because the rules of the game are not known.
