# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:36:44 2021

@author: trang.le
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.kaggle_meta_utils import *

KAGGLE_META = "C:/Users/trang.le/Downloads/archive/"
competitionID = int("23823")
submissions = pd.read_csv(os.path.join(KAGGLE_META,'Submissions.csv'))
teams = pd.read_csv(os.path.join(KAGGLE_META,'Teams.csv'))
teams = teams[teams.CompetitionId == competitionID]
