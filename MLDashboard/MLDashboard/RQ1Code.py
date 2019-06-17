import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


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

    # Separates the X predicters and Y output and then into training and test sets
    X = df.drop(columns=['Unnamed: 0', 'LABEL'])
    y = df['LABEL'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ohe = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)
    X_train_ohe = ohe.fit_transform(X_train)
    X_train_ohe.head()

    # Transforms the test set to numerical dummy columns
    X_test_ohe = ohe.transform(X_test)
    X_test_ohe.head()

    # KNN Classification
    classifier = KNeighborsClassifier(n_neighbors=655)
    classifier.fit(X_train_ohe, y_train)

    # Gets the predicted values for y
    y_pred = classifier.predict(X_test_ohe)
    cancelled = 0
    diverted = 0
    delayed = 0
    onTime = 0
    for i in y_pred:
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

    # Outputs a confusion matrix and classification report
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Check the test set score
    print("Test set score: {:.2f}".format(classifier.score(X_test_ohe, y_test)))
    
RQ1()