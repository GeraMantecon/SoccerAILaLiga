import sys
import pandas as pd

def GetStats(away_team,home_team):
    dataset = pd.read_csv('team_training_set_balanced.csv',index_col=0)
    #Get last match of each team
    last_away_match = dataset[dataset['a_team_title'] == away_team].sort_values(by=['date'], ascending=True).tail(1)
    last_home_match = dataset[dataset['h_team_title'] == home_team].sort_values(by=['date'], ascending=True).tail(1)
    #Remove some columns
    last_away_match = last_away_match[last_away_match.columns.difference(['match_result','season','date','a_team_title','h_team_title'])]
    last_home_match = last_home_match[last_home_match.columns.difference(['match_result','season','date','h_team_title','a_team_title'])]
    #Clean to only away or home stats
    last_away_match = last_away_match[[column for column in last_away_match.columns if column.startswith('a_')]]
    last_home_match = last_home_match[[column for column in last_home_match.columns if column.startswith('h_')]]
    last_away_match = last_away_match.reset_index(drop=True)
    last_home_match = last_home_match.reset_index(drop=True)
    last_match = pd.concat([last_away_match, last_home_match], axis=1)
    return last_match

def FormatPrediction(prediction,away_team,home_team):
    if prediction == 'away_win':
        result = {'match_result':str(away_team)}
    elif prediction == 'home_win':
        result = {'match_result':str(home_team)}
    else:
        result == {'match_result': 'draw'}
    return result

if __name__ == '__main__':
    main()
