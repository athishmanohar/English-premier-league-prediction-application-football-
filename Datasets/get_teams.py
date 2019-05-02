import pandas as pd
teams=pd.read_csv('2016.csv')
teams=teams['HomeTeam'].unique()
teams.sort()
print(teams)