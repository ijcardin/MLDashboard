import pandas as pd
import glob

def getDelayedFlights():
    onlydelayed = pd.concat \
        ([pd.read_csv(f, index_col='Unnamed: 0') for f in glob.glob('./Data/delayedflights/only-delayed*.csv')])

    delayedflightcount = onlydelayed['DEST_STATE_ABR'].value_counts()
    print(delayedflightcount)

def Merge(a, b):
    out = {}
    for key in a.keys() | b.keys():
        if key in a and key in b:
            out[key] = a[key] + b[key]
        elif key in a:
            out[key] = a[key]
        else:
            out[key] = b[key]
    return out

def getOnTimeFlights():
    onTime = pd.concat \
        ([pd.read_csv(f, index_col='Unnamed: 0') for f in glob.glob('./Data/onTimeFlights/on-time*.csv')])
    onTimeflightcount = onTime['DEST_STATE_ABR'].value_counts().to_dict()

    onTime2 = pd.read_csv('./Data/on-time2012.csv')
    onTimeflightCount2 = onTime2['DEST_STATE_ABR'].value_counts().to_dict()

    finalDF = Merge(onTimeflightcount, onTimeflightCount2)
    print(finalDF)

# getDelayedFlights()
# getOnTimeFlights()

# This could is meant to run only if all the necessary files are present