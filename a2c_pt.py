from math import gamma
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import copy
import random
import math
from collections import deque
from gym.wrappers import RecordEpisodeStatistics
from typing import NamedTuple

#https://colab.research.google.com/github/yfletberliac/rlss-2019/blob/master/labs/DRL.01.REINFORCE%2BA2C.ipynb#scrollTo=xDifFS9I4X7A