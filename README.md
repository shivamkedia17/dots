Credits: lantunes/dnbpy

Refer to above repo's readme for more usage.

Dependencies:
torch
np
idr what else

to train agent:
```sh
python agent.py
```

save output to a file:
```sh
python agent.py > file2.txt
```

TODO:
load trained agent as an option to play against in play.py

# Training:

Attempt 1:
Terminates on Wrong Move: True
Legal Edge Selection (w Epsilon Greedy): False
Wrong Move Reward: -100000
Losing Reward: -100
Winning Reward: 100

Observations:
- Game always terminates with wrong move. Games never complete.
- Agent likely has no idea what valid moves even are.

Attempt 2:
Terminates on Wrong Move: True
Legal Edge Selection (w Epsilon Greedy): True
Wrong Move Reward: -250
Losing Reward: -100
Winning Reward: 100

+ Some games complete.
+ When epsilon reduces and exploitation starts to take place, the rewards begin to drop.
+ Again, Agent likely has no idea what valid moves even are, even though it learns something from random.

Attempt 3:
Terminates on Wrong Move: False
