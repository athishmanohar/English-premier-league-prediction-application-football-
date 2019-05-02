import pandas as pd
raw_data = []
raw_data.append(pd.read_csv('E0.csv'))
print(raw_data[0].head())

selected_columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR']
seasons = []
for data in raw_data:
    seasons.append(data[selected_columns])

# HTGS - Home Team Goal Scored
# ATGS = Away Team Goal Scored

# Create DataFrame for aggregated goals scored by each teams till each matchweeks
def get_agg_goals_scored(season):
    # Create  a dictonary with team names as keys
    teams = {}
    for i in season.groupby('HomeTeam').mean().T.columns:  # get team name as index
        teams[i] = []
    
    # Goals scored each matchweek by each team (as either Home Team or Away Team)
    for i in range(len(season)):
        HTGS = season.iloc[i]['FTHG']
        ATGS = season.iloc[i]['FTAG']
        teams[season.iloc[i].HomeTeam].append(HTGS)
        teams[season.iloc[i].AwayTeam].append(ATGS)
        
    # Create a dataframe for goals scored where rows are teams and cols are matchweek.
    goals_scored = pd.DataFrame(data=teams, index=[i for i in range(1,(len(season) // 10) + 1)]).T
    goals_scored[0] = 0
    
    # Aggregate goals scored till each matchweek.
    for i in range(2,(len(season) // 10) + 1):
        goals_scored[i] = goals_scored[i] + goals_scored[i-1]
        
    return goals_scored


# HTGC - Home Team Goal Conceded
# ATGC = Away Team Goal Conceded

# Create DataFrame for aggregated goals conceded by each teams till each matchweeks
def get_agg_goals_conceded(season):
    # Create  a dictonary with team names as key
    teams = {}
    for i in season.groupby('HomeTeam').mean().T.columns:  # get team name as index
        teams[i] = []
    
    # Goals conceded each matchweek by each team (as either Home Team or Away Team)
    for i in range(len(season)):
        ATGC = season.iloc[i]['FTHG']
        HTGC = season.iloc[i]['FTAG']
        teams[season.iloc[i].HomeTeam].append(HTGC)
        teams[season.iloc[i].AwayTeam].append(ATGC)
        
    # Create a dataframe for goals conceded where rows are teams and cols are matchweek.
    goals_conceded = pd.DataFrame(data=teams, index=[i for i in range(1,(len(season) // 10) + 1)]).T
    goals_conceded[0] = 0
    
    # Aggregate goals conceded till each matchweek.
    for i in range(2,(len(season) // 10) + 1):
        goals_conceded[i] = goals_conceded[i] + goals_conceded[i-1]
        
    return goals_conceded

# Add aggregate goals scored and conceded of Home Team and Away Team before matchweek to gameplay_stat
# AHTGS - Aggreated Home Team Goal Scored
# AATGS - Aggreated Away Team Goal Scored
# AHTGC - Aggreated Home Team Goal Conceded
# AATGC - Aggreated Away Team Goal Conceded

def get_gss(season):
    AGS = get_agg_goals_scored(season)
    AGC = get_agg_goals_conceded(season)
    
    j = 0
    AHTGS = []
    AATGS = []
    AHTGC = []
    AATGC = []
    
    for i in range(len(season)):
        ht = season.iloc[i].HomeTeam
        at = season.iloc[i].AwayTeam
        AHTGS.append(AGS.loc[ht][j])
        AATGS.append(AGS.loc[at][j])
        AHTGC.append(AGC.loc[ht][j])
        AATGC.append(AGC.loc[at][j])
        
        if ((i + 1) % 10) == 0:
            j = j + 1
            
    season['AHTGS'] = AHTGS
    season['AATGS'] = AATGS
    season['AHTGC'] = AHTGC
    season['AATGC'] = AATGC
    
    return season


# Apply to each season
for i, _ in enumerate(seasons):
    seasons[i] = get_gss(seasons[i])

def get_points(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0

def get_cuml_points(match_results, season):
    matchres_points = match_results.applymap(get_points)
    for i in range(2, (len(season) // 10) + 1):
        matchres_points[i] = matchres_points[i] + matchres_points[i-1]
        
    matchres_points.insert(column=0, loc=0, value=[0 * i for i in range(20)])
    return matchres_points

def get_match_results(season):
    # Create dictionary with team names as keys
    teams = {}
    for i in season.groupby('HomeTeam').mean().T.columns:
        teams[i] = []
        
    # the value corresponding to keys is a list containing the match result
    for i in range(len(season)):
        if season.iloc[i].FTR == 'H':
            teams[season.iloc[i].HomeTeam].append('W')
            teams[season.iloc[i].AwayTeam].append('L')
        elif season.iloc[i].FTR == 'A':
            teams[season.iloc[i].HomeTeam].append('L')
            teams[season.iloc[i].AwayTeam].append('W')
        else:
            teams[season.iloc[i].HomeTeam].append('D')
            teams[season.iloc[i].AwayTeam].append('D')
            
    return pd.DataFrame(data=teams, index=[i for i in range(1, (len(season) // 10) + 1)]).T

# HTP - Home Team Points
# ATP - Away Team Points

def get_agg_points(season):
    match_results = get_match_results(season)
    cum_pts = get_cuml_points(match_results, season)
    HTP = []
    ATP = []
    j = 0
    for i in range(len(season)):
        ht = season.iloc[i].HomeTeam
        at = season.iloc[i].AwayTeam
        HTP.append(cum_pts.loc[ht][j])
        ATP.append(cum_pts.loc[at][j])
        
        if ((i + 1) % 10) == 0:
            j = j + 1
            
    season.loc[:,'HTP'] = HTP
    season.loc[:,'ATP'] = ATP
    
    return season

# Apply to each season
for i, _ in enumerate(seasons):
    seasons[i] = get_agg_points(seasons[i])

def get_form(season, num):          # the num th before
    form = get_match_results(season)
    form_final = form.copy()
    for i in range(num, (len(season) // 10) + 1):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i-j]
            j += 1
    return form_final

def add_form(season, num):
    form = get_form(season, num)
    h = ['M' for i in range(num * 10)]    # since form is not available for n MW (n*10)
    a = ['M' for i in range(num * 10)]
    
    j = num
    for i in range((num * 10), len(season)):
        ht = season.iloc[i].HomeTeam
        at = season.iloc[i].AwayTeam
        
        past = form.loc[ht][j]    # get past n results
        h.append(past[num - 1])   # 0 index is most recent
        
        past = form.loc[at][j]    # get past n results
        a.append(past[num - 1])   # 0 in dex is most recent
        
        if ((i + 1) % 10) == 0:
            j = j + 1
            
    season['HM' + str(num)] = h
    season['AM' + str(num)] = a
    
    return season

def add_form_df(season):
    season = add_form(season, 1)
    season = add_form(season, 2)
    season = add_form(season, 3)
    return season

# Apply to each season
for i, _ in enumerate(seasons):
    seasons[i] = add_form_df(seasons[i])

# Rearranging columns
cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'AHTGS', 'AATGS', 'AHTGC', 'AATGC', 'HTP', 'ATP', 'HM1', 'HM2', 'HM3',
        'AM1', 'AM2', 'AM3']

# Apply to each season
for i, _ in enumerate(seasons):
    seasons[i] = seasons[i][cols]

standings = pd.read_csv('EPLStandings.csv')
standings.set_index(['Team'], inplace=True)
standings = standings.fillna(18)

def get_last(season, standings, year):
    home_team_lp = []
    away_team_lp = []
    for i in range(len(season)):
        ht = season.iloc[i].HomeTeam
        at = season.iloc[i].AwayTeam
        home_team_lp.append(standings.loc[ht][year])
        away_team_lp.append(standings.loc[at][year])
    season['HomeTeamLP'] = home_team_lp
    season['AwayTeamLP'] = away_team_lp
    return season

# Apply to each season
seasons[0] = get_last(seasons[0], standings, 15)

def get_mw(season):
    j = 1
    match_week = []
    for i in range(len(season)):
        match_week.append(j)
        if ((i + 1) % 10) == 0:
            j = j + 1
    season['MW'] = match_week
    return season

# Apply to each season
for i, _ in enumerate(seasons):
    seasons[i] = get_mw(seasons[i])

gameplays = pd.concat(seasons, ignore_index=True)

# Diff in last year positions
gameplays['DiffLP'] = gameplays['HomeTeamLP'] - gameplays['AwayTeamLP']

# Get Goal Difference
gameplays['HTGD'] = gameplays['AHTGS'] - gameplays['AHTGC']
gameplays['ATGD'] = gameplays['AATGS'] - gameplays['AATGC']

# Gets the form points.
def get_form_points(string):
    sum = 0
    for letter in string:
        sum += get_points(letter)
    return sum

gameplays['HTFormPtsStr'] = gameplays['HM1'] + gameplays['HM2'] + gameplays['HM3']
gameplays['ATFormPtsStr'] = gameplays['AM1'] + gameplays['AM2'] + gameplays['AM3']

gameplays['HTFormPts'] = gameplays['HTFormPtsStr'].apply(get_form_points)
gameplays['ATFormPts'] = gameplays['ATFormPtsStr'].apply(get_form_points)

# Gets difference form point
gameplays['DiffFormPts'] = gameplays['HTFormPts'] - gameplays['ATFormPts']

# Scale HTP, ATP, HTGD, ATGD, DiffFormPts.
cols = ['HTP','ATP','HTGD','ATGD','DiffFormPts']
gameplays.MW = gameplays.MW.astype(float)

for col in cols:
    gameplays[col] = gameplays[col] / gameplays.MW

gameplays.to_csv('test1.csv')