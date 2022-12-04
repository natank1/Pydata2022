
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
def plt_bar(y_test, outp):
    mm = confusion_matrix(y_test, outp)
    precision = mm[0, 0] / sum(mm[:, 0])
    recall = mm[0, 0] / sum(mm[0, :])
    p00 = [-2, -0.5, 0.5, 2, 3, 4]
    print('precc=', precision, 'rec=', recall)
    tt = []
    for pi in p00:
        tt.append(pfunc(0.94, 0.6, pi))
        print(pi, tt)
    print("ok")
    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    xx = plt.bar(p00, tt, color='maroon',
                 width=0.4)

    print(xx)
    ty = []
    for pi in [-1, 1]:
        ty.append(pfunc(0.94, 0.6, pi))
        print(pi, tt)
    print("ok")
    xy = plt.bar([-1], ty[0], color='g',
                 width=0.4)
    xy = plt.bar([1], ty[1], color='b',
                 width=0.4)

    # xy[1].set_color('b')
    # xx[4].set_color( 'g')

    plt.xlabel("P ")
    plt.ylabel("Scores")
    plt.title("Titanic P Sscores Histogram")
    plt.legend(['All', 'Harmonic', 'Algebr'])
    plt.show()
    return
def pfunc(prec0,rec0,p):
    ss=0.5*(np.power(prec0,p)+np.power(rec0,p))
    s1 = np.power(ss,1/p)
    return s1


path1= ''
train = pd.read_csv(path1+'train.csv', header=0)
test  = pd.read_csv(path1+'test.csv', header=0)
print(train.columns)
X_full = pd.concat([train.drop('Survived', axis = 1), test], axis = 0)
print (X_full.shape)
X_full.drop('PassengerId', axis = 1, inplace=True)
selector = (train.Cabin.isnull() & train.Age.isnull())
selector = (train.Cabin.isnull())
X_full['Nulls'] = X_full.Cabin.isnull().astype('int') + X_full.Age.isnull().astype('int')
print (train[selector].Survived.mean())
X_full['Cabin_mapped'] = X_full['Cabin'].astype(str).str[0]
cabin_dict = {k:i for i, k in enumerate(X_full.Cabin_mapped.unique())}
X_full.loc[:, 'Cabin_mapped'] = X_full.loc[:, 'Cabin_mapped'].map(cabin_dict)
print (cabin_dict,X_full.columns)
X_full.drop(['Age', 'Cabin'], inplace = True, axis = 1)
fare_mean = X_full[X_full.Pclass == 3].Fare.mean()
X_full['Fare'].fillna(fare_mean, inplace = True)
X_full['Embarked'].fillna('S', inplace = True)
X_full.drop(['Name', 'Ticket'], axis = 1, inplace = True)
X_dummies = pd.get_dummies(X_full, columns = ['Sex', 'Nulls', 'Cabin_mapped', 'Embarked'], drop_first= True)
X = X_dummies[:len(train)]; new_X = X_dummies[len(train):]
y = train.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = .3,
                                                    random_state = 5,
                                                   stratify = y)
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
outp =xgb.predict(X_test)
print (xgb.score(X_test, y_test))
plt_bar(y_test, outp)
