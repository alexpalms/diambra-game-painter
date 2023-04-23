# DIAMBRA Game Painter

<img src="https://raw.githubusercontent.com/alexpalms/diambra-game-painter/main/graphics/diambra-game-painter.jpg" alt="diambra" width="100%"/>

This project is an experiment that applies in real-time the style of famous paintings to popular fighting retro games, which are provided as Reinforcement Learning environments by DIAMBRA (<a href="https://github.com/diambra" target_="blank">GitHub</a> - <a href="https://diambra.ai" target_="blank">Website</a>).

It is based on <a href="https://github.com/1627180283/real-time-Style-Transfer" target="_blank">this implementation</a> of the paper <a href="https://arxiv.org/abs/1603.08155" target="_blank">Perceptual Losses for Real-Time Style Transfer and Super-Resolution</a>.

## How to run it

- Create a virtual environment using your preferred tool (Conda, VirtualEnv, etc)
- Install required python packages: `pip install -r requirements.txt`
- Run the script on a random game: `diambra run python gameEmulationStyleTransf.py`
- Run the script on a specific game: `diambra run python gameEmulationStyleTransf.py --gameId game-ID` where available games and their game-IDs can be found in <a href="https://docs.diambra.ai" target_="blank">DIAMBRA documentation</a>
