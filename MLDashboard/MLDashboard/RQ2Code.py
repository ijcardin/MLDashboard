import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from IPython.display import Markdown, display

def printmd(string):
    display(Markdown(string))

def RQ2():
    # Read in data
    df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('./Data/RQ2', "*.csv"))))

    # Remove unnecessary columns
    # df = df[['YEAR', 'MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER',
    #                  'ORIGIN', 'ORIGIN_STATE_ABR','DEST', 'DEST_STATE_ABR',
    #                  'CRS_DEP_TIME','CRS_ARR_TIME','DISTANCE', 'CARRIER_DELAY',
    #                  'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']]

    # Remove all na values which indicate that flights are on-time or cancelled
    lateFlights = df.dropna()

    # Remove all flights that do not leave from IAH airport
    flightDataHouston = lateFlights[lateFlights['ORIGIN']=='IAH']

    # Create the dataframe for single label analysis
    singleLabel = flightDataHouston.copy()
    singleLabel['MAX_DELAY'] = singleLabel[
        ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']].idxmax(axis=1)
    singleLabel.head()

    # Remove unnecessary label columns
    singleLabel = singleLabel.drop(
        ['ORIGIN', 'ORIGIN_STATE_ABR', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY',
         'LATE_AIRCRAFT_DELAY'], axis=1)
    singleLabel.head()

    # Create dataframe for multilabel analysis
    multiLabel = flightDataHouston.copy()
    multiLabel['CARRIER_DELAY'] = np.where(multiLabel['CARRIER_DELAY'] >= 1, 1, 0)
    multiLabel['WEATHER_DELAY'] = np.where(multiLabel['WEATHER_DELAY'] >= 1, 1, 0)
    multiLabel['NAS_DELAY'] = np.where(multiLabel['NAS_DELAY'] >= 1, 1, 0)
    multiLabel['SECURITY_DELAY'] = np.where(multiLabel['SECURITY_DELAY'] >= 1, 1, 0)
    multiLabel['LATE_AIRCRAFT_DELAY'] = np.where(multiLabel['LATE_AIRCRAFT_DELAY'] >= 1, 1, 0)

    multiLabel.head()
    # Remove unnecessary columns
    multiLabel = multiLabel.drop(['ORIGIN', 'ORIGIN_STATE_ABR'], axis=1)

    # Complete one hot encoding for single label analysis
    # print(singleLabel.head())
    singleLabel = pd.get_dummies(singleLabel, columns=['UNIQUE_CARRIER'], prefix=['UNIQUE_CARRIER'])
    singleLabel = pd.get_dummies(singleLabel, columns=['DEST'], prefix=['DEST'])
    singleLabel = pd.get_dummies(singleLabel, columns=['DEST_STATE_ABR'], prefix=['DEST_STATE_ABR'])
    singleLabel = pd.get_dummies(singleLabel, columns=['DAY_OF_WEEK'], prefix=['DAY_OF_WEEK'])
    singleLabel = pd.get_dummies(singleLabel, columns=['MONTH'], prefix=['MONTH'])

    # Complete one hot encoding for multilabel analysis
    multiLabel = pd.get_dummies(multiLabel, columns=['UNIQUE_CARRIER'], prefix=['UNIQUE_CARRIER'])
    multiLabel = pd.get_dummies(multiLabel, columns=['DEST'], prefix=['DEST'])
    multiLabel = pd.get_dummies(multiLabel, columns=['DEST_STATE_ABR'], prefix=['DEST_STATE_ABR'])
    multiLabel = pd.get_dummies(multiLabel, columns=['DAY_OF_WEEK'], prefix=['DAY_OF_WEEK'])
    multiLabel = pd.get_dummies(multiLabel, columns=['MONTH'], prefix=['MONTH'])

    # Single Label Test/Train Sets
    single_features = singleLabel.drop('MAX_DELAY', axis='columns')
    single_label = singleLabel.MAX_DELAY

    # Multi Label Test/Train Sets
    multi_features = multiLabel.drop(
        ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'], axis='columns')
    multi_label = multiLabel[['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']]

    # Set up train and test split for single label analysis
    x_train_single, x_test_single, y_train_single, y_test_single = train_test_split(single_features, single_label,
                                                                                    test_size=0.3, shuffle=True)
    # Set up train and test split for multi label analysis
    x_train_multi, x_test_multi, y_train_multi, y_test_multi = train_test_split(multi_features, multi_label,
                                                                                test_size=0.3, shuffle=True)

    # Using pipeline for applying logistic regression and one vs rest classifier
    LogReg_pipeline = Pipeline([
        ('clf', OneVsRestClassifier(LogisticRegression(solver='sag', max_iter=4000), n_jobs=-1)),
    ])
    print("start")
    for label in multi_label:
        printmd('**Processing {} comments...**'.format(label))
        print("test1")
        # Training logistic regression model on train data
        LogReg_pipeline.fit(x_train_multi, y_train_multi[label])
        print("test2, entering prediction")
        # calculating test accuracy
        prediction = LogReg_pipeline.predict(x_test_multi)
        print('Test accuracy is {}'.format(accuracy_score(y_test_multi[label], prediction)))
        print("\n")

        print("Confusion Matrix")
        print(confusion_matrix(y_test_multi[label], prediction))
        print('\n')

        print("Classification Report")
        print(classification_report(y_test_multi[label], prediction))


RQ2()
