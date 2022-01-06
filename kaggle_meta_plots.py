# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 11:36:44 2021

@author: trang.le
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from kaggle_meta_utils import *
from datetime import datetime
import seaborn as sns
from itertools import accumulate

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
    submissions_hpa = submissions_hpa=submissions.merge(teams, how='inner', left_on='TeamId', right_on='Id')
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

#%%
#submissions_hpa = filter_submissions_hpa()
#submissions_hpa = pd.read_csv("Y:/HPA_SingleCellClassification/tmp/submissions_hpa.csv")
submissions_hpa = pd.read_csv("./tmp/submissions_hpa.csv")
n_teams = len(submissions_hpa.TeamId.unique())
aggregated_performance = submissions_hpa.dropna(subset=['ScoreDate']).groupby(['TeamId', 'DiffDate']).agg({
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
aggregated_performance = submissions_hpa.dropna(subset=['ScoreDate']).groupby('DiffDate').agg({
    'PublicScoreLeaderboardDisplay': 'mean',
    'PrivateScoreLeaderboardDisplay': 'mean'
    }).reset_index()
aggregated_performance = aggregated_performance.melt('DiffDate')
splot = sns.relplot(data=aggregated_performance, 
    x="DiffDate", 
    y="value", 
    hue="variable")
splot.set_xlabel("Days during competition", fontsize = 20)
splot.set_ylabel("Average mAP", fontsize = 20)
sfig = splot.get_figure()
sfig.savefig("./tmp/LeaderboardDisplay_lineplot.png", orientation="landscape")

### Divide teams into quantiles
quantiles = np.arange(0, 1, 0.1)
Quantiles_teams = np.quantile(submissions_hpa.dropna(subset=['ScoreDate']).PublicScoreLeaderboardDisplay, quantiles)

# Divide teams into top 10, top 100
# 3D plot of time (Weekly/Monley), scores and freq (n_teams)
# frequency of n_submissions through times


### Plotting max public score so far
aggregated_performance = submissions_hpa.dropna(subset=['ScoreDate']).groupby(['TeamId', 'DiffDate'], sort=True).agg({
    'PublicScoreLeaderboardDisplay': 'max',
    'PrivateScoreLeaderboardDisplay': 'max'
    }).reset_index()

tmp = aggregated_performance.groupby("TeamId")['PublicScoreLeaderboardDisplay'].apply(lambda d: list(accumulate(d, max))).to_list()
aggregated_performance['Max_so_far'] = [val for sublist in tmp for val in sublist]
# Metric change date!!!

splot = sns.lineplot(data=aggregated_performance, 
    x="DiffDate", 
    y="Max_so_far", 
    hue="TeamId")
splot.set_xlabel("Days during competition", fontsize = 20)
splot.set_ylabel("Public mAP", fontsize = 20)
sfig = splot.get_figure()


### Grouping teams into quantiles/medal ranks
palette = dict(zip(aggregated_performance.Medal.unique(),
                   sns.color_palette("rocket_r", 3)))
aggregated_performance = submissions_hpa.dropna(subset=['ScoreDate']).groupby(['Medal', 'DiffDate'], sort=True).agg({
    'PublicScoreLeaderboardDisplay': 'mean',
    'PrivateScoreLeaderboardDisplay': 'mean'
    }).reset_index()

tmp = aggregated_performance.groupby("Medal")['PublicScoreLeaderboardDisplay'].apply(lambda d: list(accumulate(d, max))).to_list()
aggregated_performance['Max_so_far'] = [val for sublist in tmp for val in sublist]
aggregated_performance['Medal'] = aggregated_performance['Medal'].astype('uint8')
splot = sns.lineplot(data=aggregated_performance, 
    x="DiffDate", 
    y="Max_so_far", 
    hue="Medal",
    palette=palette)
splot.set_xlabel("Days during competition", fontsize = 20)
splot.set_ylabel("Public mAP", fontsize = 20)
sfig = splot.get_figure()
sfig.savefig("./tmp/PublicScoreLeaderboardDisplay_mean_MaxScoreProgression_medalgroups.png", orientation="landscape")

plt.scatter(submissions_hpa.PrivateLeaderboardRank, submissions_hpa.PublicLeaderboardRank)
