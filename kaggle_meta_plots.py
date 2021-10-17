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
import seaborn as sns

KAGGLE_META = "/data/kaggle-dataset/kaggle_meta_archive/"
COMPETITION_ID = 23823

def days_from_start(s2):
    s1 = "01/26/2021" # competition start date
    FMT = "%m/%d/%Y"
    tdelta = datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)
    return tdelta.days

def filter_submissions_hpa():
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
    submissions_hpa.to_csv("./tmp/submissions_hpa.csv",index=False)
    return submissions_hpa

submissions_hpa = pd.read_csv("./tmp/submissions_hpa.csv")
n_teams = len(submissions_hpa.TeamId.unique())
aggregated_performance = submissions_hpa.groupby(['TeamId', 'DiffDate']).agg({
    'PublicScoreLeaderboardDisplay': 'mean',
    'PrivateScoreLeaderboardDisplay': 'mean'
    }).reset_index()

aggregated_performance.to_csv("./tmp/aggregated_performance.csv")
print(aggregated_performance.head())
print(aggregated_performance.shape, f"Number of teams {n_teams}")

###
splot = sns.lineplot(data=aggregated_performance, 
    x="DiffDate", 
    y="PublicScoreLeaderboardDisplay", 
    hue="TeamId")
splot.set_xlabel("Days during competition", fontsize = 20)
splot.set_ylabel("Public mAP", fontsize = 20)
sfig = splot.get_figure()
sfig.savefig("./tmp/PublicScoreLeaderboardDisplay_lineplot.png", orientation="landscape")
plt.close()

###
palette = dict(zip(aggregated_performance.TeamId.unique(),
                   sns.color_palette("rocket_r", n_teams)))
p = sns.relplot(data=aggregated_performance, 
    x="DiffDate", 
    y="PublicScoreLeaderboardDisplay", 
    hue="TeamId",
    palette=palette)
p.savefig("./tmp/PublicScoreLeaderboardDisplay_relplot.png")
plt.close()

###
aggregated_performance = submissions_hpa.groupby('DiffDate').agg({
    'PublicScoreLeaderboardDisplay': 'mean',
    'PrivateScoreLeaderboardDisplay': 'mean'
    }).reset_index()
aggregated_performance = aggregated_performance.melt('DiffDate')

splot = sns.lineplot(data=aggregated_performance, 
    x="DiffDate", 
    y="value", 
    hue="variable")
splot.set_xlabel("Days during competition", fontsize = 20)
splot.set_ylabel("Average mAP", fontsize = 20)
sfig = splot.get_figure()
sfig.savefig("./tmp/LeaderboardDisplay_lineplot.png", orientation="landscape")

### Divide teams into quantiles
quantiles = np.range(0.1, 0.9, 0.1)
Quantiles_teams = np.quantile(submissions_hpa.PublicScoreLeaderboardDisplay, quantiles)