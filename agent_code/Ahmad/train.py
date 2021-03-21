import pickle
import random
from collections import namedtuple, deque
from typing import List
import pandas as pd
import sklearn as sk
import events as e
from .callbacks import state_to_features
from sklearn import tree
# This is only an example!
Transition = namedtuple('Transition',
                        ( 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 400  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...


#def (data):
#    for i in data:
        
# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def encod_action(action):

    if action==e.BOMB_DROPPED:
        return 0
    elif action ==e.BOMB_EXPLODED:
        return 1    
    elif action== e.COIN_COLLECTED:
        return 2
    elif action==e.COIN_FOUND:
        return 3
    elif action==e.CRATE_DESTROYED:
        return 4
    elif action==e.GOT_KILLED:
        return 5
    elif action==e.INVALID_ACTION:
        return 6
    elif action==e.KILLED_OPPONENT:
        return 7
    elif action==e.KILLED_SELF:
        return 8
    elif action==e.MOVED_DOWN:
        return 9
    elif action==e.MOVED_LEFT:
        return 10
    elif action==e.MOVED_RIGHT:
        return 11
    elif action==e.MOVED_UP:
        return 12
    elif action==e.OPPONENT_ELIMINATED:
        return 13
    elif action==e.SURVIVED_ROUND:
        return 14
    elif action==e.WAITED:
        return 15


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    #self_action=encod_action(self_action)    
    #self.dataframe= pd.DataFrame(columns = ['action'])
    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition( self_action, state_to_features(old_game_state), reward_from_events(self, events)))
    
def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    #self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition( last_action, state_to_features(last_game_state), reward_from_events(self, events)))
    self.dataFrame = pd.DataFrame(self.transitions,columns= ["lastaction", 'state','reward'])
    
    #self.dataframe= pd.DataFrame(self.transitions,columns = ['state', 'action', 'next_state','reward'])
    #model_input= self.dataframe.drop("action",axis="columns")
    #model_target= self.dataframe["action"]
    self.dataFrame.to_csv("data.csv",index=False,sep="F")
    #model_input.to_csv("model_input.csv")
    #model_target.to_csv("model_target.csv")
    f = open("Transitions.txt", "a")
    #f.write(str(self.transitions[0][3])+" \n that is what im talkin about \n")
    #for i in self.transitions:
        
    #f.write(str(type(self.dataFrame["state"]))+"\n")
    #f.write(str(type(self.dataFrame["state"][0]))+"\n")
    #f.write(str((self.dataFrame["state"]))+"\n")
    for i in range(10):
        if self.dataFrame.at[i,"state"]==None:
            continue
        f.write("\n Fuck"+str(type(self.dataFrame.at[i,"state"]))+"Fuck \n Fuck \n")
        f.write("\n Fuck"+str(self.dataFrame.at[i,"state"][0])+"yeah digga we know how to get you \n")
        f.write("\n Fuck"+str(self.dataFrame.at[i,"state"])+"Fuck \n Fuck \n")
        f.write("\n Lenght"+str(len(self.dataFrame.at[i,"state"]))+"Fuck \n Fuck \n")

    
    f.close()
     #   f.write(str(type(i[2]))+"\n")
      #  f.write(str(type(i[3]))+"\n")
       # f.write("Fuck\n")
    #f.write("type of transitions"+str(type(self.transitions))+"\n")
    #f.write("type of events"+str(type(self.transitions[3]))+"\n")
    #f.write("Transitions \n")
    #f.write(str(self.transitions)+"\n")
    #f.write("should be the rewards \n")
    #f.write(str(self.transitions[3])+"\n")
    # Store the model
    #self.model = tree.DecisionTreeClassifier()   
    #self.model.fit(flatten(self.Transitions) ,self.last_action)

    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
       


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -10,
        e.SURVIVED_ROUND:2,
        e.BOMB_DROPPED:1,
        e.BOMB_EXPLODED:1,
        e.INVALID_ACTION:-2,
        e.GOT_KILLED: -6  
        # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
