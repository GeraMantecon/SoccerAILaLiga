import sys
import json
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

######################## SETTINGS ########################
stats_url = 'https://understat.com/league/La_liga'

seasons = ['2020','2019','2018','2017','2016','2015']

teams_stats_var = 'teamsData'

players_stats_var = 'playersData'

player_history_var = 'matchesData'
##########################################################

def main():
    df_teams_stats = pd.DataFrame()
    df_players_stats = pd.DataFrame()

    for season in seasons:

        #Build URL and call request method.
        url = 'https://understat.com/league/La_liga/'+season
        res = requests.get(url)

        #Make soup and obtain variables where info is stored.
        soup = BeautifulSoup(res.content,'lxml')
        scripts = soup.find_all('script')

        teams_stats = ''
        players_stats = ''

        for element in scripts:
            if teams_stats_var in element.text:
                teams_stats = element.text.strip()
                teams_stats = teams_stats.split('\'')[1]
            if players_stats_var in element.text:
                players_stats = element.text.strip()
                players_stats = players_stats.split('\'')[1]

        #Cast to dict.
        teams_stats = json.loads(teams_stats.encode('utf8').decode('unicode_escape'))
        players_stats = json.loads(players_stats.encode('utf8').decode('unicode_escape'))

        #Create teams DF (nested dictionary).
        teams_history = []
        for team in teams_stats.values():
            for history in team['history']:
                history['team_title'] = team['title']
                unzipped_keys = []
                for key in list(history.keys()):
                    if type(history[key]) is dict:
                        for skey,value in history[key].items():
                            history[key+'_'+skey] = value
                        del history[key]
                teams_history.append(history)
        df_teams_stats_local = pd.DataFrame(teams_history)
        df_teams_stats_local['season'] = season
        df_teams_stats = pd.concat([df_teams_stats,df_teams_stats_local])

        #Crate players DF.
        df_players_stats_local = pd.DataFrame(players_stats)
        df_players_stats_local['season'] = season
        df_players_stats = pd.concat([df_players_stats,df_players_stats_local])

    #Save datasets.
    df_teams_stats.to_csv('teams_stats_dataset.csv')
    df_players_stats.to_csv('players_stats_dataset.csv')

    #Get players history.
    players_id = df_players_stats['id'].tolist()

    #Players history dataframe.
    df_players_history = pd.DataFrame()


    counter = 0
    for player in players_id:
        counter += 1
        #Build URL and do get request
        url = 'https://understat.com/player/'+player
        res = requests.get(url)

        #Make soup and obtain data.
        soup = BeautifulSoup(res.content,'lxml')
        scripts = soup.find_all('script')

        player_history = ''

        for element in scripts:
            if player_history_var in element.text:
                player_history = element.text.strip()
                player_history = player_history.split('\'')[1]

        #To DataFrame
        df_player_history = pd.DataFrame(json.loads(player_history.encode('utf8').decode('unicode_escape')))
        df_players_history = pd.concat([df_players_history,df_player_history])
        print(counter)

    #Save player history dataset.
    df_players_history.to_csv('players_stats_history_dataset.csv')
    print(df_players_history.columns)

if __name__ == '__main__':
    main()
