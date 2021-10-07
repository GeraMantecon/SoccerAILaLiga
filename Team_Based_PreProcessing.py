import sys
import warnings
import numpy as np
import pandas as pd
from collections import Counter

from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

def WinnerFunction(row):
    if row['a_wins'] == 1 and row['a_loses'] == 0 and row['a_draws'] ==0 and row['h_wins'] == 0 and row['h_loses'] == 1 and row['h_draws'] == 0:
        result = 'away_win'
    elif row['a_wins'] == 0 and row['a_loses'] == 1 and row['a_draws'] ==0 and row['h_wins'] == 1 and row['h_loses'] == 0 and row['h_draws'] == 0:
        result = 'home_win'
    elif row['a_wins'] == 0 and row['a_loses'] == 0 and row['a_draws'] ==1 and row['h_wins'] == 0 and row['h_loses'] == 0 and row['h_draws'] == 1:
        result = 'draw'
    else:
        result ='error'
    return result

def GetTeams(df_teams):
    teams = []
    for team in list(set(df_teams['a_team_title'].tolist() + df_teams['h_team_title'].tolist())):
        df_local_h = df_teams[df_teams['h_team_title'] == team]
        df_local_a = df_teams[df_teams['a_team_title'] == team]

        season_h = list(set(df_local_h['season'].tolist()))
        season_a = list(set(df_local_a['season'].tolist()))
        season_h.sort()
        season_a.sort()
        print(team)
        print(season_h)
        print(season_a)

        seasons = [2020,2019,2018,2017,2016,2015]
        seasons.sort()
        print(seasons)
        if (seasons == season_h) and (seasons == season_a):
            teams.append(team)
    return teams

def CreateMatchesDF(df_teams):
    home_dataset = df_teams[df_teams['h_a'] == 'h'].add_prefix('h_')
    away_dataset = df_teams[df_teams['h_a'] == 'a'].add_prefix('a_')

    home_dataset.rename(columns = {'h_season':'season','h_date':'date'}, inplace = True)
    away_dataset.rename(columns = {'a_season':'season','a_date':'date'}, inplace = True)

    matches_dataset = pd.merge(home_dataset,away_dataset,how='inner',on=['season','date'],validate = 'many_to_many')
    matches_dataset['match_result'] = matches_dataset.apply(WinnerFunction,axis=1)
    matches_dataset = matches_dataset[matches_dataset['match_result'] != 'error']
    return matches_dataset

def SplitTrainTest(df_matches):
    to_drop = ['a_pts','h_pts','a_scored','h_scored','a_missed','h_missed','a_deep','h_deep','a_deep_allowed','h_deep_allowed',
    'a_ppda_att','h_ppda_att','a_ppda_allowed_att','h_ppda_allowed_att','a_ppda_def','h_ppda_def','a_ppda_allowed_def',
    'h_ppda_allowed_def','a_wins','a_loses','a_draws','h_wins','h_loses',
    'h_draws','a_result','h_result','h_h_a','a_h_a','dummy_points']
    x_columns = [column for column in df_matches.columns if 'x' in column]
    df_matches = df_matches.drop(x_columns,axis=1)
    train_set = df_matches[df_matches['season'] != 2020]
    test_set = df_matches[df_matches['season'] == 2020]
    train_set = train_set.drop(to_drop, axis = 1)
    test_set = test_set.drop(to_drop, axis = 1)
    return train_set,test_set

def PreProcessSums(df_matches,save=False):
    df_temp = pd.DataFrame()
    df_matches['dummy_points'] = np.nan
    df_matches['a_pts_season'] = np.nan
    df_matches['h_pts_season'] = np.nan
    df_matches['a_scores_avg'] = np.nan
    df_matches['h_scores_avg'] = np.nan
    df_matches['a_missed_avg'] = np.nan
    df_matches['h_missed_avg'] = np.nan
    df_matches['a_deep_avg'] = np.nan
    df_matches['h_deep_avg'] = np.nan
    df_matches['a_deep_allowed_avg'] = np.nan
    df_matches['h_deep_allowed_avg'] = np.nan
    df_matches['a_ppda_att_avg'] = np.nan
    df_matches['h_ppda_att_avg'] = np.nan
    df_matches['a_ppda_allowed_att_avg'] = np.nan
    df_matches['h_ppda_allowed_att_avg'] = np.nan
    df_matches['a_ppda_def_avg'] = np.nan
    df_matches['h_ppda_def_avg'] = np.nan
    df_matches['a_ppda_allowed_def_avg'] = np.nan
    df_matches['h_ppda_allowed_def_avg'] = np.nan
    df_matches['a_win_rate_season'] = np.nan
    df_matches['h_win_rate_season'] = np.nan
    df_matches['a_win_rate_hist'] = np.nan
    df_matches['h_win_rate_hist'] = np.nan
    df_matches['date'] =  pd.to_datetime(df_matches['date'], format='%Y-%m-%d %H:%M:%S')
    teams = list(set(df_matches['a_team_title'].tolist()  + df_matches['h_team_title'].tolist()))
    seasons = df_matches['season'].unique()
    for season in seasons:
        for team in teams:
            df_temp_local = df_matches[(df_matches['season'] == season) & ((df_matches['a_team_title'] == team) | (df_matches['h_team_title'] == team))]
            df_temp_local = df_temp_local.sort_values(by=['date'], ascending=True)
            if not df_temp_local.empty:
                #Points aggregate
                start_date = df_temp_local['date'].min()
                df_temp_local['dummy_points'] = df_temp_local.apply(lambda x: df_temp_local[(x.date >= start_date) & (df_temp_local.date < x.date ) & (df_temp_local.a_team_title == team) & (df_temp_local.match_result == 'away_win')]['a_pts'].sum() +
                                                                               df_temp_local[(x.date >= start_date) & (df_temp_local.date < x.date ) & (df_temp_local.h_team_title == team) & (df_temp_local.match_result == 'home_win')]['h_pts'].sum() +
                                                                               df_temp_local[(x.date >= start_date) & (df_temp_local.date < x.date ) & (df_temp_local.match_result == 'draw')].count(), axis=1)
                a_dummy_dict = df_temp_local[df_temp_local['a_team_title'] == team]['dummy_points'].to_dict()
                h_dummy_dict = df_temp_local[df_temp_local['h_team_title'] == team]['dummy_points'].to_dict()
                df_matches['a_pts_season'].update(pd.Series(a_dummy_dict))
                df_matches['h_pts_season'].update(pd.Series(h_dummy_dict))
                #Win rate.
                #Win rate per season.
                df_temp_local['a_win_rate_season'] = df_temp_local.apply(lambda x: (df_temp_local[(x.date >= start_date) & (df_temp_local.date < x.date ) & (df_temp_local.a_team_title == team) & (df_temp_local.match_result == 'away_win')].count() /
                                                                              df_temp_local[(x.date >= start_date) & (df_temp_local.date < x.date )].count())*100,axis=1)
                df_temp_local['h_win_rate_season'] = df_temp_local.apply(lambda x: (df_temp_local[(x.date >= start_date) & (df_temp_local.date < x.date ) & (df_temp_local.h_team_title == team) & (df_temp_local.match_result == 'home_win')].count() /
                                                                              df_temp_local[(x.date >= start_date) & (df_temp_local.date < x.date )].count())*100,axis=1)
                a_dummy_dict = df_temp_local[df_temp_local['a_team_title'] == team]['a_win_rate_season'].to_dict()
                h_dummy_dict = df_temp_local[df_temp_local['h_team_title'] == team]['h_win_rate_season'].to_dict()
                df_matches['a_win_rate_season'].update(pd.Series(a_dummy_dict))
                df_matches['h_win_rate_season'].update(pd.Series(h_dummy_dict))
                df_matches['a_win_rate_season'] = df_matches['a_win_rate_season'].fillna(0.0)
                df_matches['h_win_rate_season'] = df_matches['h_win_rate_season'].fillna(0.0)
                #Win rate history.
                df_temp_hist = df_matches[(df_matches['a_team_title'] == team) | (df_matches['h_team_title'] == team)]
                df_temp_hist['a_win_rate_hist'] = df_temp_hist.apply(lambda x: (df_temp_hist[(df_temp_hist.match_result == 'away_win') & (df_temp_hist.a_team_title == team)].count() /
                                                                              df_temp_hist[df_temp_hist.a_team_title == team].count())*100,axis=1)
                df_temp_hist['h_win_rate_hist'] = df_temp_hist.apply(lambda x: (df_temp_hist[(df_temp_hist.match_result == 'home_win') & (df_temp_hist.h_team_title == team)].count() /
                                                                              df_temp_hist[df_temp_hist.h_team_title == team].count())*100,axis=1)
                a_dummy_dict = df_temp_hist['a_win_rate_hist'].to_dict()
                h_dummy_dict = df_temp_hist['h_win_rate_hist'].to_dict()
                df_matches['a_win_rate_hist'].update(pd.Series(a_dummy_dict))
                df_matches['h_win_rate_hist'].update(pd.Series(h_dummy_dict))
                df_matches['a_win_rate_hist'] = df_matches['a_win_rate_hist'].fillna(0.0)
                df_matches['h_win_rate_hist'] = df_matches['h_win_rate_hist'].fillna(0.0)
                #Avg scores
                a_dummy_scores = df_temp_local[df_temp_local['a_team_title'] == team].sort_values(by=['date'], ascending=True)
                h_dummy_scores = df_temp_local[df_temp_local['h_team_title'] == team].sort_values(by=['date'], ascending=True)
                a_dummy_scores['a_scores_avg'] = a_dummy_scores['a_scored'].rolling(5,min_periods=1).mean().fillna(0.0)
                h_dummy_scores['h_scores_avg'] = h_dummy_scores['h_scored'].rolling(5,min_periods=1).mean().fillna(0.0)
                a_dummy_dict = a_dummy_scores['a_scores_avg'].to_dict()
                h_dummy_dict = h_dummy_scores['h_scores_avg'].to_dict()
                df_matches['a_scores_avg'].update(pd.Series(a_dummy_dict))
                df_matches['h_scores_avg'].update(pd.Series(h_dummy_dict))
                #Avg missed
                a_dummy_missed = df_temp_local[df_temp_local['a_team_title'] == team].sort_values(by=['date'], ascending=True)
                h_dummy_missed = df_temp_local[df_temp_local['h_team_title'] == team].sort_values(by=['date'], ascending=True)
                a_dummy_missed['a_missed_avg'] = a_dummy_missed['a_missed'].rolling(5,min_periods=1).mean().fillna(0.0)
                h_dummy_missed['h_missed_avg'] = h_dummy_missed['h_missed'].rolling(5,min_periods=1).mean().fillna(0.0)
                a_dummy_dict = a_dummy_missed['a_missed_avg'].to_dict()
                h_dummy_dict = h_dummy_missed['h_missed_avg'].to_dict()
                df_matches['a_missed_avg'].update(pd.Series(a_dummy_dict))
                df_matches['h_missed_avg'].update(pd.Series(h_dummy_dict))
                #Avg deep passes
                a_dummy_deep = df_temp_local[df_temp_local['a_team_title'] == team].sort_values(by=['date'], ascending=True)
                h_dummy_deep = df_temp_local[df_temp_local['h_team_title'] == team].sort_values(by=['date'], ascending=True)
                a_dummy_deep['a_deep_avg'] = a_dummy_deep['a_deep'].rolling(5,min_periods=1).mean().fillna(0.0)
                h_dummy_deep['h_deep_avg'] = h_dummy_deep['h_deep'].rolling(5,min_periods=1).mean().fillna(0.0)
                a_dummy_dict = a_dummy_deep['a_deep_avg'].to_dict()
                h_dummy_dict = h_dummy_deep['h_deep_avg'].to_dict()
                df_matches['a_deep_avg'].update(pd.Series(a_dummy_dict))
                df_matches['h_deep_avg'].update(pd.Series(h_dummy_dict))
                #Avg deep allowed
                a_dummy_deep_allowed = df_temp_local[df_temp_local['a_team_title'] == team].sort_values(by=['date'], ascending=True)
                h_dummy_deep_allowed = df_temp_local[df_temp_local['h_team_title'] == team].sort_values(by=['date'], ascending=True)
                a_dummy_deep_allowed['a_deep_allowed_avg'] = a_dummy_deep_allowed['a_deep_allowed'].rolling(5,min_periods=1).mean().fillna(0.0)
                h_dummy_deep_allowed['h_deep_allowed_avg'] = h_dummy_deep_allowed['h_deep_allowed'].rolling(5,min_periods=1).mean().fillna(0.0)
                a_dummy_dict = a_dummy_deep_allowed['a_deep_allowed_avg'].to_dict()
                h_dummy_dict = h_dummy_deep_allowed['h_deep_allowed_avg'].to_dict()
                df_matches['a_deep_allowed_avg'].update(pd.Series(a_dummy_dict))
                df_matches['h_deep_allowed_avg'].update(pd.Series(h_dummy_dict))
                #Avg ppda_att
                a_dummy_ppda_att = df_temp_local[df_temp_local['a_team_title'] == team].sort_values(by=['date'], ascending=True)
                h_dummy_ppda_att = df_temp_local[df_temp_local['h_team_title'] == team].sort_values(by=['date'], ascending=True)
                a_dummy_ppda_att['a_ppda_att_avg'] = a_dummy_ppda_att['a_ppda_att'].rolling(5,min_periods=1).mean().fillna(0.0)
                h_dummy_ppda_att['h_ppda_att_avg'] = h_dummy_ppda_att['h_ppda_att'].rolling(5,min_periods=1).mean().fillna(0.0)
                a_dummy_dict = a_dummy_ppda_att['a_ppda_att_avg'].to_dict()
                h_dummy_dict = h_dummy_ppda_att['h_ppda_att_avg'].to_dict()
                df_matches['a_ppda_att_avg'].update(pd.Series(a_dummy_dict))
                df_matches['h_ppda_att_avg'].update(pd.Series(h_dummy_dict))
                #Avg ppda_allowed_att
                a_dummy_ppda_allowed_att = df_temp_local[df_temp_local['a_team_title'] == team].sort_values(by=['date'], ascending=True)
                h_dummy_ppda_allowed_att = df_temp_local[df_temp_local['h_team_title'] == team].sort_values(by=['date'], ascending=True)
                a_dummy_ppda_allowed_att['a_ppda_allowed_att_avg'] = a_dummy_ppda_allowed_att['a_ppda_allowed_att'].rolling(5,min_periods=1).mean().fillna(0.0)
                h_dummy_ppda_allowed_att['h_ppda_allowed_att_avg'] = h_dummy_ppda_allowed_att['h_ppda_allowed_att'].rolling(5,min_periods=1).mean().fillna(0.0)
                a_dummy_dict = a_dummy_ppda_allowed_att['a_ppda_allowed_att_avg'].to_dict()
                h_dummy_dict = h_dummy_ppda_allowed_att['h_ppda_allowed_att_avg'].to_dict()
                df_matches['a_ppda_allowed_att_avg'].update(pd.Series(a_dummy_dict))
                df_matches['h_ppda_allowed_att_avg'].update(pd.Series(h_dummy_dict))
                #Avg ppda_def
                a_dummy_ppda_def = df_temp_local[df_temp_local['a_team_title'] == team].sort_values(by=['date'], ascending=True)
                h_dummy_ppda_def = df_temp_local[df_temp_local['h_team_title'] == team].sort_values(by=['date'], ascending=True)
                a_dummy_ppda_def['a_ppda_def_avg'] = a_dummy_ppda_def['a_ppda_def'].rolling(5,min_periods=1).mean().fillna(0.0)
                h_dummy_ppda_def['h_ppda_def_avg'] = h_dummy_ppda_def['h_ppda_def'].rolling(5,min_periods=1).mean().fillna(0.0)
                a_dummy_dict = a_dummy_ppda_def['a_ppda_def_avg'].to_dict()
                h_dummy_dict = h_dummy_ppda_def['h_ppda_def_avg'].to_dict()
                df_matches['a_ppda_def_avg'].update(pd.Series(a_dummy_dict))
                df_matches['h_ppda_def_avg'].update(pd.Series(h_dummy_dict))
                #Avg ppda_allowed_def
                a_dummy_ppda_allowed_def = df_temp_local[df_temp_local['a_team_title'] == team].sort_values(by=['date'], ascending=True)
                h_dummy_ppda_allowed_def = df_temp_local[df_temp_local['h_team_title'] == team].sort_values(by=['date'], ascending=True)
                a_dummy_ppda_allowed_def['a_ppda_allowed_def_avg'] = a_dummy_ppda_allowed_def['a_ppda_allowed_def'].rolling(5,min_periods=1).mean().fillna(0.0)
                h_dummy_ppda_allowed_def['h_ppda_allowed_def_avg'] = h_dummy_ppda_allowed_def['h_ppda_allowed_def'].rolling(5,min_periods=1).mean().fillna(0.0)
                a_dummy_dict = a_dummy_ppda_allowed_def['a_ppda_allowed_def_avg'].to_dict()
                h_dummy_dict = h_dummy_ppda_allowed_def['h_ppda_allowed_def_avg'].to_dict()
                df_matches['a_ppda_allowed_def_avg'].update(pd.Series(a_dummy_dict))
                df_matches['h_ppda_allowed_def_avg'].update(pd.Series(h_dummy_dict))
            if save == True:
                df_matches.to_csv('StatsScrapper/processed_team_stats_dataset.csv')
    return df_matches

def main():

    #Load team dataset.
    team_dataset = pd.read_csv('StatsScrapper/teams_stats_dataset.csv',index_col=0)

    #Match matches.
    matches_dataset = CreateMatchesDF(team_dataset)

    #Pre-Process matches.
    matches_dataset = PreProcessSums(matches_dataset,save=False)

    #Split train and test.
    train_set_data, test_set_data = SplitTrainTest(matches_dataset)
    train_set_data.to_csv('StatsScrapper/team_training_set.csv')
    test_set_data.to_csv('StatsScrapper/team_test_set.csv')

    #Subsample balanced train.
    train_set_data = train_set_data.sort_values(by=['date'], ascending=True)
    train_set_data_draw = train_set_data[train_set_data['match_result'] == 'draw'].tail(500)
    train_set_data_aw = train_set_data[train_set_data['match_result'] == 'away_win'].tail(500)
    train_set_data_hw = train_set_data[train_set_data['match_result'] == 'home_win'].tail(500)
    train_set_data = pd.concat([train_set_data_draw,train_set_data_aw,train_set_data_hw])
    train_set_data.to_csv('StatsScrapper/team_training_set_balanced.csv')

if __name__ == '__main__':
    main()
