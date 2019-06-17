import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from yellowbrick.regressor import ResidualsPlot
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from matplotlib.gridspec import GridSpec



def RQ3():
    # read in the data
    over15s = pd.concat([pd.read_csv(f, index_col='Unnamed: 0') for f in glob.glob('./Data/RQ3/delay15/over15*.csv')])
    onlydelayed = pd.concat([pd.read_csv(f, index_col='Unnamed: 0') for f in glob.glob('./Data/OnlyDelayed/only-delayed*.csv')])
    normal = pd.concat([pd.read_csv(f, index_col='Unnamed: 0') for f in glob.glob('./Data/RQ3/allArrivedFlights/Features*.csv')])

    # change the delay type to boolean
    c_d_b = list()
    w_d_b = list()
    n_d_b = list()
    s_d_b = list()
    l_d_b = list()

    for row in over15s.iterrows():
        if int(row[1][9]) == 0:
            c_d_b.append(0)
        else:
            c_d_b.append(1)

        if int(row[1][10]) == 0:
            w_d_b.append(0)
        else:
            w_d_b.append(1)

        if int(row[1][11]) == 0:
            n_d_b.append(0)
        else:
            n_d_b.append(1)

        if int(row[1][12]) == 0:
            s_d_b.append(0)
        else:
            s_d_b.append(1)

        if int(row[1][13]) == 0:
            l_d_b.append(0)
        else:
            l_d_b.append(1)

    over15s['CARRIER_DELAY_BOOL'] = c_d_b
    over15s['WEATHER_DELAY_BOOL'] = w_d_b
    over15s['NAS_DELAY_BOOL'] = n_d_b
    over15s['SECURITY_DELAY_BOOL'] = s_d_b
    over15s['LATE_AIRCRAFT_DELAY_BOOL'] = l_d_b

    over15s = over15s[[x for x in over15s.columns if
                       x != 'CARRIER_DELAY' and x != 'WEATHER_DELAY' and x != 'NAS_DELAY' and x != 'SECURITY_DELAY' and x != 'LATE_AIRCRAFT_DELAY']]

    # change the column ARR_DELAY_NEW to ARR_DELAY for consistency in the onlydelayed dataframe
    new_col = [c for c in onlydelayed.columns if c != 'ARR_DELAY_NEW']
    new_col.append('ARR_DELAY')
    onlydelayed.columns = new_col
    print(onlydelayed.columns)
    over15s.head()

    # split the datasets
    datasets = [over15s, onlydelayed, normal]

    X_trains = list()
    X_tests = list()
    y_trains = list()
    y_tests = list()

    for dataset in datasets:
        X = dataset[
            [x for x in dataset.columns if x != 'ARR_DELAY' and x != 'MONTH' and x != 'DAY_OF_WEEK' and x != 'YEAR']]
        y = dataset['ARR_DELAY']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)

    # hot encode the training and testing sets
    ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)

    X_trains_encoded = list()
    X_tests_encoded = list()

    for x in X_trains:
        X_trains_encoded.append(ohe.fit_transform(x))

    for x in X_tests:
        X_tests_encoded.append(ohe.fit_transform(x))

    # let's print the summary of each regression using statsmodels (default is WITHOUT an intercept)
    for i in range(len(datasets)):
        model = sm.OLS(y_trains[i], X_trains_encoded[i]).fit()
        print(model.summary())

    # Now WITH an intercept using scikit learn, but let's just look at the over15s dataframe
    lm = linear_model.LinearRegression()
    model = lm.fit(X_trains_encoded[0], y_trains[0])
    score = lm.score(X_trains_encoded[0], y_trains[0])
    print(score)

    # Instantiate the linear model and visualizer
    ridge = Ridge()
    visualizer = ResidualsPlot(ridge)

    visualizer.fit(X_trains_encoded[0], y_trains[0])  # Fit the training data to the model
    visualizer.score(X_tests_encoded[0], y_tests[0])  # Evaluate the model on the test data
    visualizer.poof()  # Draw/show/poof the data

    # so all of the unique carriers had almost identical coefficents, so lets re-analyze without them
    n_c = list(X_trains_encoded[0].columns[17:])
    x = X_trains_encoded[0][n_c]
    # let's also add in some interaction variables based on the continuous columns
    x['DEP_TIME_X_ELAPSED_TIME'] = x['CRS_DEP_TIME'] * x['CRS_ELAPSED_TIME']
    x['DEP_TIME_X_DISTANCE'] = x['CRS_DEP_TIME'] * x['DISTANCE']
    x['ELAPSED_TIME_X_DISTANCE'] = x['CRS_ELAPSED_TIME'] * x['DISTANCE']
    y = y_trains[0]

    # rework the model
    model = sm.OLS(y, x).fit()
    print(model.summary())

    # rework the model WITH an intercept
    lm = linear_model.LinearRegression()
    model = lm.fit(x, y)
    score = lm.score(x, y)
    print(score)

    predictions = lm.predict(x)
    Y = y

    plt.scatter(Y, predictions)
    plt.xlabel('Mean delays (min)', fontsize=15)
    plt.ylabel('Predictions (min)', fontsize=15)
