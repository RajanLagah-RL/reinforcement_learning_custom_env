import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt 
import pickle 
from matplotlib import style 
import time 

# ggplot is just grammer for making graphs
style.use("ggplot")

# Matrix size where player,food,enemy can b
SIZE = 10

# how many episodes
HM_EPISODES = 25000

# Reducing reward on moving on empty
MOVE_PENALTY = 1

# Reducing reward on moving on Enemy
ENEMY_PENALTY = 300

# Reducing reward on moving on Food
FOOD_REWARD = 25

# for randomness / exploitation of data 
epsilon = 0.5
EPS_DECAY = 0.9998

# To show render
SHOW_EVERY = 100

start_q_table = None 

# use in formula
LEARNING_RATE = 0.1
DISCOUNT = 0.95

# Just to map colors
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

# defining colors
d = {
    1:(255,175,0),
    2:(0,255,0),
    3:(0,0,255)
}


# class for players
class Blob:
    def __init__(self):
        # randomly on matrix
        self.x = np.random.randint(0,SIZE)
        self.y = np.random.randint(0,SIZE)

    def __str__(self):
        return f"{self.x},{self.y}"

    def __sub__(self,other):
        return (self.x - other.x,self.y - other.y)

    # actions that player can perform
    def action(self,choice):
        if choice == 0:
            self.move(x=1,y=1)
        if choice == 1:
            self.move(x=-1,y=-1)
        if choice == 2:
            self.move(x=1,y=-1)
        if choice == 3:
            self.move(x=-1,y=1)  

    # to move player
    def move(self,x=False,y=False):
        if not x:
            self.x += np.random.randint(-1,2)
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1,2)
        else:
            self.y += y
        # if they are off grid
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE -1:
            self.x = SIZE -1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1
# initializing q-table 
# for v will try to have every ((player,food),(player,enemy)) combination and then will set weight to random actions ( total actions are 4) for bigning 
if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE+1,SIZE):
        for y1 in range(-SIZE+1,SIZE):
            for x2 in range(-SIZE+1,SIZE):
                for y2 in range(-SIZE+1,SIZE):
                    q_table[(x1,y1),(x2,y2)] = [np.random.uniform(-5,0) for i in range(4)]
else:
    # We can also load q-table from database/file
    with open(start_q_table,"rb") as f:
        q_table = pickle.load(f)

# for graphing
episode_rewards = []

for episode in range(HM_EPISODES):
    # intializing players
    player = Blob()
    food = Blob()
    enemy = Blob()

    # check if we can now render the output
    if episode % SHOW_EVERY == 0:
        print(f"on # {episode},epsilon:{epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    # initialzing episode_reward value can b any
    episode_reward = 0

    for i in range(200):
        # we will give max 200 steps to player to get to food
        obs = (player-food,player-enemy)

        # for exploration we will decay it and then we will start exploitation
        if np.random.random()>epsilon:
            # We will decide action at that place (x1,y1),(x2,y2) from [reward(action1),reward(action2),reward(action3),reward(action4)]
            # remember q_table is map of all position (player,food),(player,enemy) to reward of all 4 actions.
            # q_table[obs] is array reward of all 4 actions
            # np.argmax(q_table[obs]) is the action that have max reward 
            action = np.argmax(q_table[obs])
        else:
            # for exploitation
            action = np.random.randint(0,4)

        # Moving player
        player.action(action)

        # enemy.move()
        # food.move()

        # rewarding/pnalities
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        # after moving player
        new_obs = (player-food, player-enemy)
        # finding q/best reward for specific position
        max_future_q = np.max(q_table[new_obs])
        # current q / best current best reward 
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            # formula
            new_q = (1-LEARNING_RATE)*current_q+LEARNING_RATE*(reward + DISCOUNT*max_future_q)

        # setting reward for specific action
        q_table[obs][action] = new_q

        # to show game
        if show:
            # createing rgb plane
            env = np.zeros((SIZE,SIZE,3), dtype=np.uint8)
            # setting colors at that position
            env[food.y][food.x] = d[FOOD_N]
            env[player.y][player.x] = d[PLAYER_N]
            env[enemy.y][enemy.x] = d[ENEMY_N]

            img = Image.fromarray(env,"RGB")
            img = img.resize((300,300))
            cv2.imshow("",np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                # wait to show win
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                # wait to show steps
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break 
            # for graph
            episode_reward = reward
            # when game over
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                break
        # for graph
        episode_rewards.append(episode_reward)
        # reducing epsilon
        epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards,np.ones((SHOW_EVERY,))/ SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}")
plt.xlabel("episode #")
plt.show()


with open(f"qtable-{int(time.time())},pickle","wb") as f:
    pickle.dump(q_table,f)







