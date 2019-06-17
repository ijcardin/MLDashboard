import pandas as pd
import glob

def RQ1():
    path = r'./Data/RQ1'
    all_files = glob.glob(path + "/*.csv")

    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0).dropna()
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)

    cancelled = 0
    diverted = 0
    delayed = 0
    onTime = 0
    for i in df["LABEL"]:
        if (i == 0):
            onTime += 1
        elif (i == 1):
            delayed += 1
        elif (i == 2):
            diverted += 1
        elif (i == 3):
            cancelled += 1
    print("On Time: ", onTime)
    print("Delayed: ", delayed)
    print("Diverted: ", diverted)
    print("Cancelled: ", cancelled)

RQ1()