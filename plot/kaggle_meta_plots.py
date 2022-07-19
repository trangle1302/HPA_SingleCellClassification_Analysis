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
submissions_hpa['Medal'][submissions_hpa.Medal.isna()] = 'None'
n_teams = len(submissions_hpa.TeamId.unique())

aggregated_performance = submissions_hpa.dropna(subset=['ScoreDate']).groupby(['TeamId','PrivateLeaderboardRank']).agg({
    'PublicScoreLeaderboardDisplay': 'count',
    }).reset_index()
aggregated_performance['AvgSubmissionNumber'] = aggregated_performance.PublicScoreLeaderboardDisplay/105

print(f'Average submission number of all teams {aggregated_performance.AvgSubmissionNumber.mean()}')
print(f'Average submission number of top 10 teams {aggregated_performance[aggregated_performance.PrivateLeaderboardRank<11].AvgSubmissionNumber.mean()}') 

submissions_hpa.dropna(subset=['ScoreDate']).groupby(['TeamId','Medal']).agg({
    'PublicScoreLeaderboardDisplay': 'count',
    }).groupby('Medal').agg({
    'PublicScoreLeaderboardDisplay': 'mean',
    })
        
  
#%% Team membership
teams = pd.read_csv(os.path.join(KAGGLE_META,'Teams.csv'))
teams = teams[teams.Id.isin(submissions_hpa.dropna(subset=['ScoreDate']).TeamId)]
members = pd.read_csv(os.path.join(KAGGLE_META,'TeamMemberships.csv'))
members = members[members.TeamId.isin(submissions_hpa.dropna(subset=['ScoreDate']).TeamId)]
members.groupby('TeamId').agg({'UserId':'count'}).mean()

aggregated_performance = submissions_hpa.dropna(subset=['ScoreDate']).groupby(['TeamId', 'DiffDate']).agg({
    'PublicScoreLeaderboardDisplay': 'max',
    'PrivateScoreLeaderboardDisplay': 'max'
    }).reset_index()

aggregated_performance.to_csv("./tmp/aggregated_performance.csv")
print(aggregated_performance.head())
print(aggregated_performance.shape, f"Number of teams {n_teams}")

'''
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
'''
###
bins = range(0,106,15)
names = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90','91-105']

#submissions_hpa['DateRange'] =  pd.cut(submissions_hpa['DiffDate'], bins, labels=names)
#submissions_hpa['DateRange'] =  pd.cut(submissions_hpa['DiffDate'], bins, labels=bins[1:])

aggregated_performance = submissions_hpa.dropna(subset=['ScoreDate'])
aggregated_performance.loc[:,'DateRange'] =  pd.cut(aggregated_performance['DiffDate'], bins, labels=bins[1:])

aggregated_performance = aggregated_performance.groupby(['TeamId', 'DateRange'], sort=True).agg({
    'PublicScoreLeaderboardDisplay': 'max',
    'PrivateScoreLeaderboardDisplay': 'max'
    }).reset_index()


aggregated_performance = aggregated_performance[['PublicScoreLeaderboardDisplay','PrivateScoreLeaderboardDisplay','DateRange']].melt('DateRange')
splot = sns.violinplot(data=aggregated_performance, 
    x="DateRange", 
    y="value", 
    hue="variable")
plt.ylim(0,0.7)

splot.set_xlabel("Days during competition", fontsize = 20)
splot.set_ylabel("Average mAP", fontsize = 20)
sfig = splot.get_figure()



aggregated_performance = submissions_hpa.dropna(subset=['ScoreDate']).groupby('DiffDate').agg({
    'PublicScoreLeaderboardDisplay': 'max',
    'PrivateScoreLeaderboardDisplay': 'max'
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
aggregated_performance = submissions_hpa.dropna(subset=['ScoreDate']).groupby(['Medal', 'DiffDate'], sort=True).agg({
    'PublicScoreLeaderboardDisplay': 'mean',
    'PrivateScoreLeaderboardDisplay': 'mean'
    }).reset_index()

palette = dict(zip(aggregated_performance.Medal.unique(),
                   sns.color_palette("rocket_r", 4)))

tmp = aggregated_performance.groupby("Medal")['PublicScoreLeaderboardDisplay'].apply(lambda d: list(accumulate(d, max))).to_list()
aggregated_performance['Max_so_far'] = [val for sublist in tmp for val in sublist]
#aggregated_performance['Medal'] = aggregated_performance['Medal'].astype('uint8')
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
