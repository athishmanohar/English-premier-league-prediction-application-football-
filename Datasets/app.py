from flask import Flask, render_template
from flask import request
import os
import pandas as pd
from flask import jsonify
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.metrics import f1_score




data = pd.read_csv('final_dataset.csv')
data = data[data.MW > 3]
teams = data[['HomeTeam', 'AwayTeam']]
features_labels = data[['FTR', 'HTGD','ATGD','HTP','ATP','HM1', 'HM2', 'HM3', 'AM1', 'AM2', 'AM3','DiffFormPts','DiffLP']]
X_all = features_labels.drop(['FTR'],1)
y_all = features_labels['FTR']

Home=[]
Away=[]
X_all.HM1 = X_all.HM1.astype('str')
X_all.HM2 = X_all.HM2.astype('str')
X_all.HM3 = X_all.HM3.astype('str')
X_all.AM1 = X_all.AM1.astype('str')
X_all.AM2 = X_all.AM2.astype('str')
X_all.AM3 = X_all.AM3.astype('str')
def preprocess_features(X):
    ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''
    
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
X_train = X_all[:5800]
y_train = y_all[:5800]
X_test = X_all[5800:5940]
y_test = y_all[5800:5940]
X_predict=X_all[5940:5950]

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

scaler = StandardScaler()
mask = ('HTGD','ATGD','HTP','ATP','DiffFormPts','DiffLP')
X_train.loc[:, mask] = scaler.fit_transform(X_train.loc[:, mask])
X_test.loc[:, mask] = scaler.transform(X_test.loc[:, mask])

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


app = Flask(__name__)
CORS(app)

@app.route('/')
def get_data():
    text = 'hello'
    return render_template('bar.html', text)


#app.add_url_rule('/','hello', hello_world)

@app.route("/members/<string:name>/")
def getMember(name):
    #return name
    split_data = name.split(',')
    keyword_0 = split_data[0]
    keyword_1 = split_data[1]
    predict = clf.predict(fetch(keyword_0,keyword_1))
    if 'H' in predict:
        return Home.pop()
    elif 'A' in predict:
        return Away.pop()
    else:
        return "Draw"

    



if __name__ == "__main__":
    app.run(host = '192.168.0.5', port=10235, debug=True)
