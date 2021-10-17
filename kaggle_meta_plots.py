# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:36:44 2021

@author: trang.le
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kaggle_meta_utils import *
from datetime import datetime

KAGGLE_META = "/data/kaggle-dataset/kaggle_meta_archive/"
COMPETITION_ID = 23823

def days_from_start(s2):
    s1 = "01/26/2021" # competition start date
    FMT = "%m/%d/%Y"
    tdelta = datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)
    return tdelta.days

submissions = pd.read_csv(os.path.join(KAGGLE_META,'Submissions.csv'))
teams = pd.read_csv(os.path.join(KAGGLE_META,'Teams.csv'))
teams = teams[teams.CompetitionId == COMPETITION_ID]
submissions_hpa = submissions[submissions.TeamId.isin(teams.Id)]
#submissions_hpa = submissions_hpa[~submissions_hpa.PublicScoreLeaderboardDisplay.isna()]
submissions_hpa["DiffDate"] = [days_from_start(d) for d in submissions_hpa.SubmissionDate]
submissions_hpa = submissions_hpa[submissions_hpa.DiffDate>=0]
print(submissions_hpa.columns)
print(f"{sum(submissions_hpa.SubmissionDate != submissions_hpa.ScoreDate)} different between submission date and score date")
print(f"{len(submissions_hpa)} total submissions, including {sum(submissions_hpa.IsAfterDeadline==False)} within and {sum(submissions_hpa.IsAfterDeadline)} after deadline")

submissions_hpa = submissions_hpa[~submissions_hpa.IsAfterDeadline]
print(submissions_hpa.DiffDate.min(),submissions_hpa.DiffDate.max())

plt.figure()
