#
# Utility code to read from Meta Kaggle dataset:
# https://www.kaggle.com/kaggle/meta-kaggle
# Adjusted from codes of JAMES TROTMAN (https://www.kaggle.com/jtrotman)
#

import pandas as pd
import pathlib

MKDIR = pathlib.Path("C:/Users/trang.le/Downloads/archive")
#MKDIR = pathlib.Path("../input/meta-kaggle")
if not MKDIR.is_dir():
    MKDIR = pathlib.Path("../input")


def read_csv_filtered(csv, col, values, **kwargs):
    dfs = [df.loc[df[col].isin(values)] for df in pd.read_csv(csv, chunksize=50000, **kwargs)]
    return pd.concat(dfs, axis=0)


def parse_datetime(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format="%m/%d/%Y %H:%M:%S", cache=False)


def parse_date(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format="%m/%d/%Y", cache=True)


################################################################################
## Competitions.csv                                                           ##
################################################################################
def read_competitions(**kwargs):
    dtype = {
        "Id": "int32",
        "Slug": "O",
        "Title": "O",
        "Subtitle": "O",
        "HostSegmentTitle": "O",
        "ForumId": "float32",
        "OrganizationId": "float32",
        "CompetitionTypeId": "int32",
        "HostName": "O",
        "EnabledDate": "O",
        "DeadlineDate": "O",
        "ProhibitNewEntrantsDeadlineDate": "O",
        "TeamMergerDeadlineDate": "O",
        "TeamModelDeadlineDate": "O",
        "ModelSubmissionDeadlineDate": "O",
        "FinalLeaderboardHasBeenVerified": "bool",
        "HasKernels": "bool",
        "OnlyAllowKernelSubmissions": "bool",
        "HasLeaderboard": "bool",
        "LeaderboardPercentage": "int32",
        "LeaderboardDisplayFormat": "float32",
        "EvaluationAlgorithmAbbreviation": "O",
        "EvaluationAlgorithmName": "O",
        "EvaluationAlgorithmDescription": "O",
        "EvaluationAlgorithmIsMax": "O",
        "ValidationSetName": "O",
        "ValidationSetValue": "O",
        "MaxDailySubmissions": "int32",
        "NumScoredSubmissions": "int32",
        "MaxTeamSize": "float32",
        "BanTeamMergers": "bool",
        "EnableTeamModels": "bool",
        "EnableSubmissionModelHashes": "bool",
        "EnableSubmissionModelAttachments": "bool",
        "RewardType": "O",
        "RewardQuantity": "float32",
        "NumPrizes": "int32",
        "UserRankMultiplier": "float32",
        "CanQualifyTiers": "bool",
        "TotalTeams": "int32",
        "TotalCompetitors": "int32",
        "TotalSubmissions": "int32",
    }
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "Competitions.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "Competitions.csv", dtype=dtype, **kwargs)
    parse_datetime(df, "EnabledDate")
    parse_datetime(df, "DeadlineDate")
    parse_datetime(df, "ProhibitNewEntrantsDeadlineDate")
    parse_datetime(df, "TeamMergerDeadlineDate")
    parse_datetime(df, "TeamModelDeadlineDate")
    parse_datetime(df, "ModelSubmissionDeadlineDate")
    return df


################################################################################
## CompetitionTags.csv                                                        ##
################################################################################
def read_competition_tags(**kwargs):
    dtype = {"Id": "int32", "CompetitionId": "int32", "TagId": "int32"}
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "CompetitionTags.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "CompetitionTags.csv", dtype=dtype, **kwargs)

    return df


################################################################################
## Datasets.csv                                                               ##
################################################################################
def read_datasets(**kwargs):
    dtype = {
        "Id": "int32",
        "CreatorUserId": "int32",
        "OwnerUserId": "float32",
        "OwnerOrganizationId": "float32",
        "CurrentDatasetVersionId": "float64",
        "CurrentDatasourceVersionId": "float64",
        "ForumId": "int32",
        "Type": "int32",
        "CreationDate": "O",
        "ReviewDate": "O",
        "FeatureDate": "O",
        "LastActivityDate": "O",
        "TotalViews": "int32",
        "TotalDownloads": "int32",
        "TotalVotes": "int32",
        "TotalKernels": "int32",
    }
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "Datasets.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "Datasets.csv", dtype=dtype, **kwargs)
    parse_datetime(df, "CreationDate")
    parse_date(df, "ReviewDate")
    parse_date(df, "FeatureDate")
    parse_date(df, "LastActivityDate")
    return df


################################################################################
## DatasetTags.csv                                                            ##
################################################################################
def read_dataset_tags(**kwargs):
    dtype = {"Id": "int32", "DatasetId": "int32", "TagId": "int32"}
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "DatasetTags.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "DatasetTags.csv", dtype=dtype, **kwargs)

    return df


################################################################################
## DatasetVersions.csv                                                        ##
################################################################################
def read_dataset_versions(**kwargs):
    dtype = {
        "Id": "int32",
        "DatasetId": "int32",
        "DatasourceVersionId": "int32",
        "CreatorUserId": "int32",
        "LicenseName": "O",
        "CreationDate": "O",
        "VersionNumber": "float32",
        "Title": "O",
        "Slug": "O",
        "Subtitle": "O",
        "Description": "O",
        "VersionNotes": "O",
        "TotalCompressedBytes": "float32",
        "TotalUncompressedBytes": "float32",
    }
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "DatasetVersions.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "DatasetVersions.csv", dtype=dtype, **kwargs)
    parse_datetime(df, "CreationDate")
    return df


################################################################################
## DatasetVotes.csv                                                           ##
################################################################################
def read_dataset_votes(**kwargs):
    dtype = {"Id": "int32", "UserId": "int32", "DatasetVersionId": "int32", "VoteDate": "O"}
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "DatasetVotes.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "DatasetVotes.csv", dtype=dtype, **kwargs)
    parse_date(df, "VoteDate")
    return df


################################################################################
## Datasources.csv                                                            ##
################################################################################
def read_datasources(**kwargs):
    dtype = {
        "Id": "int32",
        "CreatorUserId": "int32",
        "CreationDate": "O",
        "Type": "int32",
        "CurrentDatasourceVersionId": "int32",
    }
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "Datasources.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "Datasources.csv", dtype=dtype, **kwargs)
    parse_datetime(df, "CreationDate")
    return df


################################################################################
## EpisodeAgents.csv                                                          ##
################################################################################
def read_episode_agents(**kwargs):
    dtype = {
        "Id": "int32",
        "EpisodeId": "int32",
        "Index": "int32",
        "Reward": "float32",
        "State": "int32",
        "SubmissionId": "int32",
        "InitialConfidence": "float32",
        "InitialScore": "float32",
        "UpdatedConfidence": "float32",
        "UpdatedScore": "float32",
    }
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "EpisodeAgents.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "EpisodeAgents.csv", dtype=dtype, **kwargs)

    return df


################################################################################
## Episodes.csv                                                               ##
################################################################################
def read_episodes(**kwargs):
    dtype = {"Id": "int32", "Type": "int32", "CompetitionId": "int32", "CreateTime": "O", "EndTime": "O"}
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "Episodes.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "Episodes.csv", dtype=dtype, **kwargs)
    parse_datetime(df, "CreateTime")
    parse_datetime(df, "EndTime")
    return df


################################################################################
## ForumMessages.csv                                                          ##
################################################################################
def read_forum_messages(**kwargs):
    dtype = {
        "Id": "int32",
        "ForumTopicId": "int32",
        "PostUserId": "int32",
        "PostDate": "O",
        "ReplyToForumMessageId": "float32",
        "Message": "O",
        "Medal": "float32",
        "MedalAwardDate": "O",
    }
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "ForumMessages.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "ForumMessages.csv", dtype=dtype, **kwargs)
    parse_datetime(df, "PostDate")
    parse_date(df, "MedalAwardDate")
    return df


################################################################################
## ForumMessageVotes.csv                                                      ##
################################################################################
def read_forum_message_votes(**kwargs):
    dtype = {"Id": "int32", "ForumMessageId": "int32", "FromUserId": "int32", "ToUserId": "int32", "VoteDate": "O"}
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "ForumMessageVotes.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "ForumMessageVotes.csv", dtype=dtype, **kwargs)
    parse_date(df, "VoteDate")
    return df


################################################################################
## Forums.csv                                                                 ##
################################################################################
def read_forums(**kwargs):
    dtype = {"Id": "int32", "ParentForumId": "float32", "Title": "O"}
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "Forums.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "Forums.csv", dtype=dtype, **kwargs)

    return df


################################################################################
## ForumTopics.csv                                                            ##
################################################################################
def read_forum_topics(**kwargs):
    dtype = {
        "Id": "int32",
        "ForumId": "int32",
        "KernelId": "float32",
        "LastForumMessageId": "float32",
        "FirstForumMessageId": "float32",
        "CreationDate": "O",
        "LastCommentDate": "O",
        "Title": "O",
        "IsSticky": "bool",
        "TotalViews": "int32",
        "Score": "int32",
        "TotalMessages": "int32",
        "TotalReplies": "int32",
    }
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "ForumTopics.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "ForumTopics.csv", dtype=dtype, **kwargs)
    parse_datetime(df, "CreationDate")
    parse_datetime(df, "LastCommentDate")
    return df


################################################################################
## KernelLanguages.csv                                                        ##
################################################################################
def read_kernel_languages(**kwargs):
    dtype = {"Id": "int32", "Name": "O", "DisplayName": "O", "IsNotebook": "bool"}
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "KernelLanguages.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "KernelLanguages.csv", dtype=dtype, **kwargs)

    return df


################################################################################
## Kernels.csv                                                                ##
################################################################################
def read_kernels(**kwargs):
    dtype = {
        "Id": "int32",
        "AuthorUserId": "int32",
        "CurrentKernelVersionId": "float64",
        "ForkParentKernelVersionId": "float64",
        "ForumTopicId": "float32",
        "FirstKernelVersionId": "float64",
        "CreationDate": "O",
        "EvaluationDate": "O",
        "MadePublicDate": "O",
        "IsProjectLanguageTemplate": "bool",
        "CurrentUrlSlug": "O",
        "Medal": "float32",
        "MedalAwardDate": "O",
        "TotalViews": "int32",
        "TotalComments": "int32",
        "TotalVotes": "int32",
    }
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "Kernels.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "Kernels.csv", dtype=dtype, **kwargs)
    parse_datetime(df, "CreationDate")
    parse_date(df, "EvaluationDate")
    parse_date(df, "MadePublicDate")
    parse_date(df, "MedalAwardDate")
    return df


################################################################################
## KernelTags.csv                                                             ##
################################################################################
def read_kernel_tags(**kwargs):
    dtype = {"Id": "int32", "KernelId": "int32", "TagId": "int32"}
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "KernelTags.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "KernelTags.csv", dtype=dtype, **kwargs)

    return df


################################################################################
## KernelVersionCompetitionSources.csv                                        ##
################################################################################
def read_kernel_version_competition_sources(**kwargs):
    dtype = {"Id": "int32", "KernelVersionId": "int32", "SourceCompetitionId": "int32"}
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "KernelVersionCompetitionSources.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "KernelVersionCompetitionSources.csv", dtype=dtype, **kwargs)

    return df


################################################################################
## KernelVersionDatasetSources.csv                                            ##
################################################################################
def read_kernel_version_dataset_sources(**kwargs):
    dtype = {"Id": "int32", "KernelVersionId": "int32", "SourceDatasetVersionId": "int32"}
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "KernelVersionDatasetSources.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "KernelVersionDatasetSources.csv", dtype=dtype, **kwargs)

    return df


################################################################################
## KernelVersionKernelSources.csv                                             ##
################################################################################
def read_kernel_version_kernel_sources(**kwargs):
    dtype = {"Id": "int32", "KernelVersionId": "int32", "SourceKernelVersionId": "int32"}
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "KernelVersionKernelSources.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "KernelVersionKernelSources.csv", dtype=dtype, **kwargs)

    return df


################################################################################
## KernelVersionOutputFiles.csv                                               ##
################################################################################
def read_kernel_version_output_files(**kwargs):
    dtype = {
        "Id": "int32",
        "KernelVersionId": "int32",
        "FileName": "str",
        "ContentLength": "int64",
        "ContentTypeExtension": "str",
        "CompressionTypeExtension": "str",
    }
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "KernelVersionOutputFiles.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "KernelVersionOutputFiles.csv", dtype=dtype, **kwargs)

    return df


################################################################################
## KernelVersions.csv                                                         ##
################################################################################
def read_kernel_versions(**kwargs):
    dtype = {
        "Id": "int32",
        "ScriptId": "int32",
        "ParentScriptVersionId": "float64",
        "ScriptLanguageId": "int32",
        "AuthorUserId": "int32",
        "CreationDate": "O",
        "VersionNumber": "float32",
        "Title": "O",
        "EvaluationDate": "O",
        "IsChange": "bool",
        "TotalLines": "float32",
        "LinesInsertedFromPrevious": "float32",
        "LinesChangedFromPrevious": "float32",
        "LinesUnchangedFromPrevious": "float32",
        "LinesInsertedFromFork": "float32",
        "LinesDeletedFromFork": "float32",
        "LinesChangedFromFork": "float32",
        "LinesUnchangedFromFork": "float32",
        "TotalVotes": "int32",
    }
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "KernelVersions.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "KernelVersions.csv", dtype=dtype, **kwargs)
    parse_datetime(df, "CreationDate")
    parse_date(df, "EvaluationDate")
    return df


################################################################################
## KernelVotes.csv                                                            ##
################################################################################
def read_kernel_votes(**kwargs):
    dtype = {"Id": "int32", "UserId": "int32", "KernelVersionId": "int32", "VoteDate": "O"}
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "KernelVotes.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "KernelVotes.csv", dtype=dtype, **kwargs)
    parse_date(df, "VoteDate")
    return df


################################################################################
## Organizations.csv                                                          ##
################################################################################
def read_organizations(**kwargs):
    dtype = {"Id": "int32", "Name": "O", "Slug": "O", "CreationDate": "O", "Description": "O"}
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "Organizations.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "Organizations.csv", dtype=dtype, **kwargs)
    parse_date(df, "CreationDate")
    return df


################################################################################
## Submissions.csv                                                            ##
################################################################################
def read_submissions(**kwargs):
    dtype = {
        "Id": "int32",
        "SubmittedUserId": "float32",
        "TeamId": "int32",
        "SourceKernelVersionId": "float64",
        "SubmissionDate": "O",
        "ScoreDate": "O",
        "IsAfterDeadline": "bool",
        "PublicScoreLeaderboardDisplay": "float32",
        "PublicScoreFullPrecision": "str",
        "PrivateScoreLeaderboardDisplay": "float32",
        "PrivateScoreFullPrecision": "str",
    }
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "Submissions.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "Submissions.csv", dtype=dtype, **kwargs)
    parse_date(df, "SubmissionDate")
    parse_date(df, "ScoreDate")
    return df


################################################################################
## Tags.csv                                                                   ##
################################################################################
def read_tags(**kwargs):
    dtype = {
        "Id": "int32",
        "ParentTagId": "float32",
        "Name": "O",
        "Slug": "O",
        "FullPath": "O",
        "Description": "O",
        "DatasetCount": "int32",
        "CompetitionCount": "int32",
        "KernelCount": "int32",
    }
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "Tags.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "Tags.csv", dtype=dtype, **kwargs)

    return df


################################################################################
## TeamMemberships.csv                                                        ##
################################################################################
def read_team_memberships(**kwargs):
    dtype = {"Id": "int32", "TeamId": "int32", "UserId": "int32", "RequestDate": "O"}
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "TeamMemberships.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "TeamMemberships.csv", dtype=dtype, **kwargs)
    parse_date(df, "RequestDate")
    return df


################################################################################
## Teams.csv                                                                  ##
################################################################################
def read_teams(**kwargs):
    dtype = {
        "Id": "int32",
        "CompetitionId": "int32",
        "TeamLeaderId": "float32",
        "TeamName": "O",
        "ScoreFirstSubmittedDate": "O",
        "LastSubmissionDate": "O",
        "PublicLeaderboardSubmissionId": "float64",
        "PrivateLeaderboardSubmissionId": "float64",
        "IsBenchmark": "bool",
        "Medal": "float32",
        "MedalAwardDate": "O",
        "PublicLeaderboardRank": "float32",
        "PrivateLeaderboardRank": "float32",
    }
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "Teams.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "Teams.csv", dtype=dtype, **kwargs)
    parse_date(df, "ScoreFirstSubmittedDate")
    parse_date(df, "LastSubmissionDate")
    parse_date(df, "MedalAwardDate")
    return df


################################################################################
## UserAchievements.csv                                                       ##
################################################################################
def read_user_achievements(**kwargs):
    dtype = {
        "Id": "int32",
        "UserId": "int32",
        "AchievementType": "O",
        "Tier": "int32",
        "TierAchievementDate": "O",
        "Points": "int32",
        "CurrentRanking": "float32",
        "HighestRanking": "float32",
        "TotalGold": "int32",
        "TotalSilver": "int32",
        "TotalBronze": "int32",
    }
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "UserAchievements.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "UserAchievements.csv", dtype=dtype, **kwargs)
    parse_date(df, "TierAchievementDate")
    return df


################################################################################
## UserFollowers.csv                                                          ##
################################################################################
def read_user_followers(**kwargs):
    dtype = {"Id": "int32", "UserId": "int32", "FollowingUserId": "int32", "CreationDate": "O"}
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "UserFollowers.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "UserFollowers.csv", dtype=dtype, **kwargs)
    parse_date(df, "CreationDate")
    return df


################################################################################
## UserOrganizations.csv                                                      ##
################################################################################
def read_user_organizations(**kwargs):
    dtype = {"Id": "int32", "UserId": "int32", "OrganizationId": "int32", "JoinDate": "O"}
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "UserOrganizations.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "UserOrganizations.csv", dtype=dtype, **kwargs)
    parse_date(df, "JoinDate")
    return df


################################################################################
## Users.csv                                                                  ##
################################################################################
def read_users(**kwargs):
    dtype = {"Id": "int32", "UserName": "O", "DisplayName": "O", "RegisterDate": "O", "PerformanceTier": "int8"}
    if "filter" in kwargs:
        col, values = kwargs.pop("filter")
        df = read_csv_filtered(MKDIR / "Users.csv", col, values, dtype=dtype, **kwargs)
    else:
        df = pd.read_csv(MKDIR / "Users.csv", dtype=dtype, **kwargs)
    parse_date(df, "RegisterDate")
    return df