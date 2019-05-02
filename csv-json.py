import csv
import json
import pandas as pd

def fun():
	df=pd.read_csv("Datasets/2016.csv")
	key_list=df['HomeTeam'].unique().tolist()
	key_list.sort()
	print(len(key_list))
	flare = dict()
	flare = {"name":"Premier League", "children": []}
	# if 'the_parent' is NOT a key in the flare.json yet, append it
	for parent in key_list:
	    #child_lis=df[parent]
	    #child_list = [x for x in child_lis if str(x) != 'nan']
	    flare['children'].append({"name":parent})
	    #for child in child_list:
	   	#	flare['children'][key_list.index(parent)]['children'].append({"name":child})
	with open("data1.json",'w') as f:
		f.write(json.dumps(flare))
	

fun()