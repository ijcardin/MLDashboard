import pandas as pd
import numpy as np
import glob
import os

def RQ2():
    # Read in data
    df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('./Data/RQ2', "*.csv"))))

    # Remove unnecessary columns
    df = df[['YEAR', 'MONTH', 'DAY_OF_WEEK', 'UNIQUE_CARRIER',
                     'ORIGIN', 'ORIGIN_STATE_ABR','DEST', 'DEST_STATE_ABR',
                     'CRS_DEP_TIME','CRS_ARR_TIME','DISTANCE', 'CARRIER_DELAY',
                     'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']]

    # Remove all na values which indicate that flights are on-time or cancelled
    lateFlights = df.dropna()

    # Remove all flights that do not leave from IAH airport
    flightDataHouston = lateFlights[lateFlights['ORIGIN']=='IAH']

RQ2()
