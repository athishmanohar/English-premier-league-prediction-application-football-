import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
import itertools

raw_data_1 = pd.read_csv('2000.csv')
raw_data_2 = pd.read_csv('2001.csv')
raw_data_3 = pd.read_csv('2002.csv')
raw_data_4 = pd.read_csv('2003.csv')
raw_data_5 = pd.read_csv('2004.csv')
raw_data_6 = pd.read_csv('2005.csv')
raw_data_7 = pd.read_csv('2007.csv')
raw_data_8 = pd.read_csv('2008.csv')
raw_data_9 = pd.read_csv('2009.csv')
raw_data_10 = pd.read_csv('2010.csv')
raw_data_11 = pd.read_csv('2011.csv')
raw_data_12 = pd.read_csv('2012.csv')
raw_data_13 = pd.read_csv('2013.csv')
raw_data_14 = pd.read_csv('2014.csv')
raw_data_15 = pd.read_csv('2015.csv')
raw_data_16 = pd.read_csv('2016.csv')

def parse_date(date):
    if date == '':
        return None
    else:
        return dt.strptime(date, '%d/%m/%y').date()

def parse_date_other(date):
    if date == '':
        return None
    else:
        return dt.strptime(date, '%d/%m/%Y').date()

raw_data_1.Date = raw_data_1.Date.apply(parse_date)    
raw_data_2.Date = raw_data_2.Date.apply(parse_date)    
raw_data_3.Date = raw_data_3.Date.apply(parse_date_other)         # The date format for this dataset is different  
raw_data_4.Date = raw_data_4.Date.apply(parse_date)    
raw_data_5.Date = raw_data_5.Date.apply(parse_date)    
raw_data_6.Date = raw_data_6.Date.apply(parse_date)    
raw_data_7.Date = raw_data_7.Date.apply(parse_date)    
raw_data_8.Date = raw_data_8.Date.apply(parse_date)    
raw_data_9.Date = raw_data_9.Date.apply(parse_date)    
raw_data_10.Date = raw_data_10.Date.apply(parse_date)
raw_data_11.Date = raw_data_11.Date.apply(parse_date)
raw_data_12.Date = raw_data_12.Date.apply(parse_date)
raw_data_13.Date = raw_data_13.Date.apply(parse_date)
raw_data_14.Date = raw_data_14.Date.apply(parse_date)
raw_data_15.Date = raw_data_15.Date.apply(parse_date)
raw_data_16.Date = raw_data_16.Date.apply(parse_date)

#Gets all the statistics related to gameplay
                      
columns_req = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HS','AS',
               'HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']

playing_statistics_1 = raw_data_1[columns_req]                      
playing_statistics_2 = raw_data_2[columns_req]
playing_statistics_3 = raw_data_3[columns_req]
playing_statistics_4 = raw_data_4[columns_req]
playing_statistics_5 = raw_data_5[columns_req]
playing_statistics_6 = raw_data_6[columns_req]
playing_statistics_7 = raw_data_7[columns_req]
playing_statistics_8 = raw_data_8[columns_req]
playing_statistics_9 = raw_data_9[columns_req]
playing_statistics_10 = raw_data_10[columns_req]
playing_statistics_11 = raw_data_11[columns_req]   
playing_statistics_12 = raw_data_12[columns_req]
playing_statistics_13 = raw_data_13[columns_req]
playing_statistics_14 = raw_data_14[columns_req]
playing_statistics_15 = raw_data_15[columns_req]
playing_statistics_16 = raw_data_16[columns_req]

playing_statistics = pd.concat([playing_statistics_1, playing_statistics_2, playing_statistics_3, playing_statistics_4,
                                playing_statistics_5, playing_statistics_6, playing_statistics_7, playing_statistics_8,
                                playing_statistics_9, playing_statistics_10, playing_statistics_11,playing_statistics_12, 
                                playing_statistics_13, playing_statistics_14, playing_statistics_15, playing_statistics_16])

def get_result_stats(playing_stats, year):
    return pd.DataFrame(data = [ len(playing_stats[playing_stats.FTR == 'H']),
                                 len(playing_stats[playing_stats.FTR == 'A']),
                                 len(playing_stats[playing_stats.FTR == 'D'])],
                        index = ['Home Wins', 'Away Wins', 'Draws'],
                        columns =[year]
                       ).T

result_stats_agg = get_result_stats(playing_statistics, 'Overall')
result_stats_1 = get_result_stats(playing_statistics_1, '2000-01')
result_stats_2 = get_result_stats(playing_statistics_2, '2001-02')
result_stats_3 = get_result_stats(playing_statistics_3, '2002-03')
result_stats_4 = get_result_stats(playing_statistics_4, '2003-04')
result_stats_5 = get_result_stats(playing_statistics_5, '2004-05')
result_stats_6 = get_result_stats(playing_statistics_6, '2005-06')
result_stats_7 = get_result_stats(playing_statistics_7, '2006-07')
result_stats_8 = get_result_stats(playing_statistics_8, '2007-08')
result_stats_9 = get_result_stats(playing_statistics_9, '2008-09')
result_stats_10 = get_result_stats(playing_statistics_10, '2009-10')
result_stats_11 = get_result_stats(playing_statistics_11, '2010-11')
result_stats_12 = get_result_stats(playing_statistics_12, '2011-12')
result_stats_13 = get_result_stats(playing_statistics_13, '2012-13')
result_stats_14 = get_result_stats(playing_statistics_14, '2013-14')
result_stats_15 = get_result_stats(playing_statistics_15, '2014-15')
result_stats_16 = get_result_stats(playing_statistics_16, '2015-16')

result_stats = pd.concat([result_stats_1, result_stats_2, result_stats_3, result_stats_4, result_stats_5,
                          result_stats_6, result_stats_7, result_stats_8, result_stats_9, result_stats_10,
                          result_stats_11, result_stats_12, result_stats_13, result_stats_14, result_stats_15,result_stats_16])
#result_stats.to_csv('analyse.csv')

#Plotting the result dataframe
ax = result_stats.plot(kind='bar', color = ['steelblue','sandybrown', 'turquoise'], figsize = [16,7.5], 
                       title='Result Statistics for Different Years')
plt.xticks(rotation=0)
ax.set_ylabel('Frequency', size=12)
ax.set_xlabel('Season', size=12)
plt.show()

#Plotting agg result dataframe
ax1 = result_stats_agg.T.plot(kind='bar', color = ['steelblue','sandybrown', 'turquoise'], figsize = [16,7.5], 
                       title='Aggregate Result Statistics', legend = False)
plt.xticks(rotation=0)
ax1.set_ylabel('Frequency', size=12)
ax1.set_xlabel('Season', size=12)
plt.show()

result_prop = result_stats.T

for column in result_prop.columns:
    result_prop[column] = (result_prop[column] * 100) / 380  #No. of total matches in a year

# Renames columns
result_prop.rename(index={'Home Wins':'Home', 'Away Wins':'Away', 'Draws':'Draw'}, inplace=True)
print(result_prop)

# Plots a line plot of the win percentages for different parameters in different seasons.

ax = result_prop.T.plot(figsize = [16,8], title = 'Win Percentage for Home team and Away team')
ax.margins(y=.75)
ax.set_xlabel('Season', size =12)
ax.set_ylabel('Win Perentage', size =12)
plt.xticks( np.arange(16), ('2000-01', '2001-02', '2002-03', '2003-04', '2004-05', '2005-06', '2006-07', '2007-08',
                            '2008-09', '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16'))
plt.show()


result_avg_prop = pd.DataFrame((result_prop['2000-01'] + result_prop['2001-02'] + result_prop['2002-03'] +
                                result_prop['2003-04'] + result_prop['2004-05'] + result_prop['2005-06'] +
                                result_prop['2006-07'] + result_prop['2007-08'] + result_prop['2008-09'] + 
                                result_prop['2009-10'] + result_prop['2010-11'] + result_prop['2011-12'] + 
                                result_prop['2012-13'] + result_prop['2013-14'] + result_prop['2014-15'] +result_prop['2015-16']) / 16, 
                                columns = ['Win Percentage'])

# Plots average win percentage as a pie chart.
ax = result_avg_prop.plot(kind='pie', figsize =[6,6],autopct='%.2f', y='Win Percentage', fontsize =20, labels = None,
                          legend = True, colors = ['steelblue','sandybrown', 'turquoise'])
ax.set_title('Aggregate Win Percentage', size=25)
plt.show()
