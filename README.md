# Game AI Project

Deep Q Learning with modificatons to progress in SNES games.
Currently targetting Classic Kong on SNES.

Created basic building blocks before we set up environment fully.
Now utilizing RLE.

# Running and Requirements

To install requirements, run `sh setup-reqs.sh`.

This will run a simple `pip install -e .` over the `gym-rle` directory to install the library properly, as well as install some extra packages.

Currently to test environment interaction, you can run the `tester.py` in the root of the repo.
`runner.py` should be the final product once we get our DQN implemention functioning.

To shut down the program, `ctrl + c` in the running terminal will end the runner.

![Example](https://raw.githubusercontent.com/westinn/game-ai-project/master/examples/setup-example.png)
