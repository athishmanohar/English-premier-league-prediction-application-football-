import pandas as pd
# Visualising distribution of data
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
#produces a prediction model in the form of an ensemble of weak prediction models, typically decision tree
import xgboost as xgb
#the outcome (dependent variable) has only a limited number of possible values. 
#Logistic Regression is used when response variable is categorical in nature.
from sklearn.linear_model import LogisticRegression
#A random forest is a meta estimator that fits a number of decision tree classifiers 
#on various sub-samples of the dataset and use averaging to improve the predictive 
#accuracy and control over-fitting.
from sklearn.svm import SVC

# Read data
data = pd.read_csv('final_dataset.csv')
# Remove first 3 matchweeks
data = data[data.MW > 3]

# Store matchweek # 4 fixtures for prediction
teams = data[['HomeTeam', 'AwayTeam']]

# Select labels & features 
# FTR - Full time result
# HTGD - Home team goal difference
# ATGD - away team goal difference
# HTP - Home team points
# ATP - Away team points
# HM1, HM2, HM3 - Home team past-match form
# AM1, MM2, AM3 - Away team past-match form
# DiffFormPts Diff in points
# DiffLP - Differnece in last years prediction

features_labels = data[['FTR', 'HTGD','ATGD','HTP','ATP','HM1', 'HM2', 'HM3', 'AM1', 'AM2', 'AM3','DiffFormPts','DiffLP']]

#scatter_matrix(features_labels[['HTGD','ATGD','HTP','ATP','DiffFormPts','DiffLP']], figsize=(10,10), grid=True)
#plt.show()

# Separate into feature set and target variable
# FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
X_all = features_labels.drop(['FTR'],1)
y_all = features_labels['FTR']

Home=[]
Away=[]
#last 3 wins for both sides
X_all.HM1 = X_all.HM1.astype('str')
X_all.HM2 = X_all.HM2.astype('str')
X_all.HM3 = X_all.HM3.astype('str')
X_all.AM1 = X_all.AM1.astype('str')
X_all.AM2 = X_all.AM2.astype('str')
X_all.AM3 = X_all.AM3.astype('str')

#we want continous vars that are integers for our input data, so lets remove any categorical vars
def preprocess_features(X):
# Preprocesses the football data and converts catagorical variables into dummy variables. 
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
                    
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
#print(X_all.shape[0])

# Seperate data in to train (2000/01-2015/16), test (2016/17), predict set (2017/18)
X_train = X_all[:5800]
y_train = y_all[:5800]
X_test = X_all[5800:5950]
y_test = y_all[5800:5950]
#X_predict=X_all[5940:5950]
#X_predict=X_predict[['HTGD','ATGD','HTP','ATP','DiffFormPts','DiffLP']]
#print(X_predict.tail())

teams_pred = pd.concat([teams, X_all], axis=1)
print(teams_pred.head())

def fetch(home,away):
    Home.append(home)
    Away.append(away)
    teams=teams_pred.loc[(teams_pred['HomeTeam'] == home) & (teams_pred['AwayTeam']==away)]
    teams=teams[list(X_all)]
    means=teams[list(X_all)].mean()
    teams=teams.append(means,ignore_index=True)
    return teams.tail(1)
    
#fetch('Chelsea','Arsenal')

# Standardising the data.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
mask = ('HTGD','ATGD','HTP','ATP','DiffFormPts','DiffLP')
X_train.loc[:, mask] = scaler.fit_transform(X_train.loc[:, mask])
X_test.loc[:, mask] = scaler.transform(X_test.loc[:, mask])
#X_predict.loc[:, mask] = scaler.transform(X_predict.loc[:, mask])

#for measuring training time
from time import time 
# F1 score (also F-score or F-measure) is a measure of a test's accuracy. 
#It considers both the precision p and the recall r of the test to compute 
#the score: p is the number of correct positive results divided by the number of 
#all positive results, and r is the number of correct positive results divided by 
#the number of positive results that should have been returned. The F1 score can be 
#interpreted as a weighted average of the precision and recall, where an F1 score 
#reaches its best value at 1 and worst at 0.
from sklearn.metrics import f1_score

def train_classifier(clf, X_train, y_train):
    # Fits a classifier to the training data.
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    #Makes predictions using a fit classifier based on F1 score.
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    
    end = time()
    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))
    
    return f1_score(target, y_pred, average='macro'), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, X_train, y_train, X_test, y_test):
    # Train and predict using a classifer based on F1 score.
    
    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, X_train, y_train)
    print(f1, acc)
    print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
    
    f1, acc = predict_labels(clf, X_test, y_test)
    print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))

# Initialize the three models (XGBoost is initialized later)
clf_A = LogisticRegression(random_state = 42)
clf_B = SVC(random_state = 912, kernel='rbf')
#Boosting refers to this general problem of producing a very accurate prediction rule 
#by combining rough and moderately inaccurate rules-of-thumb
clf_C = xgb.XGBClassifier(seed = 82)

print(train_predict(clf_A, X_train, y_train, X_test, y_test))
print('')
print(train_predict(clf_B, X_train, y_train, X_test, y_test))
print('')
print(train_predict(clf_C, X_train, y_train, X_test, y_test))
print('')


# TODO: Import 'GridSearchCV' and 'make_scorer'
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

# TODO: Create the parameters list you wish to tune
parameters = { 'learning_rate' : [0.1],
               'n_estimators' : [40],
               'max_depth': [3],
               'min_child_weight': [3],
               'gamma':[0.4],
               'subsample' : [0.8],
               'colsample_bytree' : [0.8],
               'scale_pos_weight' : [1],
               'reg_alpha':[1e-5]
             } 

# TODO: Initialize the classifier
clf = xgb.XGBClassifier(seed=2)

# TODO: Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score, average='macro')

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf,
                        scoring=f1_scorer,
                        param_grid=parameters,
                        cv=5)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train,y_train)

# Get the estimator
clf = grid_obj.best_estimator_
print(clf)

# Report the final F1 score for training and testing after parameter tuning
f1, acc = predict_labels(clf, X_train, y_train)
print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
    
f1, acc = predict_labels(clf, X_test, y_test)
print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))

predict = clf.predict(fetch('Tottenham','Everton'))

if 'H' in predict:
    print(Home.pop())
elif 'A' in predict:
    print(Away.pop())
else:
    print("Draw")
