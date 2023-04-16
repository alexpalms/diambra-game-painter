import diambra.arena
from diambra.arena.utils.gym_utils import env_spaces_summary, available_games
import random
import argparse
import sys, os

basePath = os.path.join(os.path.abspath(''))
sys.path.append(os.path.join(basePath, "realTimeStyleTransferSlim/"))
from styleTransferViz import *
from stylesList import stylesList

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gameId', type=str, default="random", help='Game ID')
    opt = parser.parse_args()
    print(opt)

    game_dict = available_games(False)
    if opt.gameId == "random":
        game_id = random.sample(game_dict.keys(),1)[0]
    else:
        game_id = opt.gameId if opt.gameId in game_dict.keys() else random.sample(game_dict.keys(),1)[0]

    # Settings
    settings = {
        "step_ratio": 6,
        "hardcore": False,
        "difficulty": 4,
        "characters": ("Random"),
        "char_outfits": 1,
        "action_space": "multi_discrete",
        "attack_but_combination": False
    }

    stViz = styleTransferViz(stylesList)

    env = diambra.arena.make(game_id, settings)
    observation = env.reset()

    counter = 0
    styleCounter = 0
    counterLimit = 100

    while styleCounter < 23:

        actions = env.action_space.sample()
        observation, reward, done, info = env.step(actions)
        gameFrame = observation["frame"]

        stViz.styleGame(gameFrame, waitPress=1)

        if done:
            observation = env.reset()

        if counter == counterLimit:
            counter = 0
            styleCounter += 1
            stViz.nextStyle()

        counter += 1

    env.close()







