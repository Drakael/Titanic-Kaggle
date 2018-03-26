import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pydot

from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import IsolationForest
#from sklearn.externals.six import StringIO

train = pd.read_csv('train.csv')
kaggle = pd.read_csv('test.csv')


PassengerId_train = train['PassengerId'].tolist()
PassengerId = kaggle['PassengerId'].tolist()


def dprint(**kwargs):
    if kwargs is not None:
        for key, value in kwargs.iteritems():
            shape = None
            if(hasattr(value,'shape')):
                shape = value.shape
            print(key,type(value),shape,"\n",value)


def group_female_titles(elem):
    if elem in ['Ms','Miss','Mme','Mlle','Lady','Dona','Countess']:
        return 'Mrs'
    return elem

def group_titles(elem):
    if elem in ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Sir', 'Jonkheer', 'Dona']:
        return 'Rare'
    elif elem in ['Mlle','Ms']:
        return 'Miss'
    elif elem in ['Mme']:
        return 'Mrs'
    return elem
    
def set_title(df):
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].map(group_titles)
    return df


def set_age_group(df):
    bins = [0, 3, 12, 18, 60, 100]
    labels = ['Baby', 'Child', 'Teenager', 'Adult', 'Elder']
    age_groups = pd.cut(df.Age, bins, labels=labels)
    df['Age_group'] = age_groups
    return pd.concat([df, pd.get_dummies(df['Age_group'])], axis=1)

def get_random_age(elem):
    age_avg = elem.mean()
    age_std = elem.std()
    age_null_count = elem.isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    elem[np.isnan(elem)] = age_null_random_list
    elem = elem.astype(int)
    return elem

def clean_nans(df):
    df['Age'] = df['Age'].fillna(get_random_age(df['Age']))#(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'].fillna(df['Embarked'].value_counts().idxmax(), inplace=True)
    return df

def clean_visu(df):
    df = clean_nans(df)
    df['Sex'] = df['Sex'].map({'female':0, 'male':1})
    portName = {'Q':2, 'C':0, 'S':1}
    df['Embarked'] = df['Embarked'].map(portName)
    df['IsAlone'] = 0
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    exclude = ['Parch','SibSp','PassengerId','Cabin','Ticket','Name']
    df = clean_nans(df).drop(exclude, axis = 1)
    df = df.dropna()
    return df

def process_data(df):
    df = df.drop('Cabin', axis=1)
    df = clean_nans(df)
    portName = {'Q':'Queenstown', 'C':'Cherbourg', 'S':'Southampton'}
    df['Embarked'] = df['Embarked'].map(portName)
    df = pd.concat([df, pd.get_dummies(df['Embarked'])], axis=1)
    className = {1:'Classe 1', 2:'Classe 2', 3:'Classe 3'}
    df['Pclass'] = df['Pclass'].map(className)
    df = pd.concat([df, pd.get_dummies(df['Pclass'])], axis=1)
    df['Sex'] = df['Sex'].map({'female':0, 'male':1})
    df['IsAlone'] = 0
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    return df
   
def visu_isolation_tree(Axis_1,Axis_2,label,color):
    data = pd.concat([Axis_1,Axis_2],axis=1).copy()
#    print('data',type(data),"\n",data)
#    print('Axis_1 uniques',type(Axis_1),"\n",Axis_1.unique())
#    print('Axis_2 uniques',type(Axis_2),"\n",Axis_2.unique())
    clf = IsolationForest(max_samples=100, random_state=0)   
    clf.fit(data)
    xx, yy = np.meshgrid(np.linspace(Axis_1.min(), Axis_1.max(), len(Axis_1.unique())), np.linspace(Axis_2.min(), Axis_2.max(), len(Axis_2.unique())))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    a = plt.scatter(Axis_1, Axis_2, c='white', s=20, edgecolor='k')
    plt.axis('tight')
    plt.xlim((Axis_1.min(), Axis_1.max()))
    plt.ylim((Axis_2.min(), Axis_2.max()))
    #print('Axis_1',type(Axis_1),Axis_1.shape,"\n",Axis_1,"\nname",Axis_1.name)
    #print('Axis_2',type(Axis_2),Axis_2.shape,"\n",Axis_2,"\nname",Axis_2.name)
    plt.legend([a],
               [Axis_1.name+' / '+Axis_2.name],
               loc="upper left")
    plt.show()
    

visu = clean_visu(train)
print('visu',type(visu),"\n",visu.describe())
#visu_isolation_tree(visu['Age'],visu['Fare'])


def pair_grid_visu(data):
    #g = sns.pairplot(visu, hue="Survived", palette="husl")
    #g = sns.pairplot(visu, hue="Sex", palette="husl")
    g1 = sns.PairGrid(data, hue="Survived", palette="husl")
    g1 = g1.map_diag(plt.hist)
    g1 = g1.map_upper(plt.scatter)
    g1 = g1.map_lower(sns.swarmplot)
    g1 = g1.add_legend()
    
    g = sns.PairGrid(data, palette="husl")
    g = g.map_diag(plt.hist)
    #g = g.map_offdiag(plt.scatter) 
    #g = g.map_upper(plt.scatter)
    g = g.map_upper(visu_isolation_tree)
    g = g.map_lower(sns.swarmplot)
    g = g.add_legend()

#visu.plot()
visu.boxplot()
visu.hist()
pair_grid_visu(visu)
#%%
#calcul et affichage des facteurs de corrélation des variables
methods = ['pearson', 'kendall', 'spearman']
correlation = train.corr(method='pearson')
print('correlation',correlation.shape,type(correlation),"\n",correlation)


print("\n"+'correlations initiales relatives = '+"\n",correlation['Survived'].sort_values(ascending= False),"\n")
correlation = correlation['Survived'].abs().sort_values(ascending= False)
print('correlations initiales absolues = ',type(correlation),"\n",correlation,"\n","\n")

#def plot_features(ax,X_train, axis_1=0, axis_2=1, predict=None, cmap=plt.cm.Blues_r):
#    #plt.figure()
#    ax.set_title(axis_1+" / "+axis_2)
#    ax.plot(x, y)
#    #sns.swarmplot(x=axis_1, y=axis_2, data=X_train, color="Blue")
#    #plt.show()  
#done = []    
#max_figures = 20
#fig, sub = plt.subplots(4, 5)
#plt.subplots_adjust(wspace=0.4, hspace=0.4)
#for i, feature_1 in enumerate(correlation.keys()):
#    done.append(i)
#    for j, feature_2 in enumerate(correlation.keys()):
#        if i is not j and j not in done and i not in exclude and max_figures>0:
#            print('i',i)
#            print('j',j)
#            print('feature_1',feature_1)
#            print('feature_2',feature_2)
#            plot_features(ax,train,feature_1,feature_2)
#            max_figures-=1
#plt.show()



#prépare le tableau de train et le tableau de test     
train = process_data(train)
kaggle = process_data(kaggle)


'''
train = set_title(train)
train['Sexe'] = train['Sex'].map({0:'Femme',1:'Homme'})
#kaggle.sort_values(['Sex','Pclass','Age'])
plt.figure()
sns.barplot(x='Embarked',y='Fare',hue='Pclass',data=train)
plt.show()
plt.figure()
sns.countplot(x='Embarked',hue='Pclass',data=train)
plt.show()
plt.figure()
sns.countplot(x='Embarked',hue='family_size',data=train)
plt.show()
'''

#prépare un tableau avec le train + le test
train_kaggle = train.append(kaggle)
train_kaggle = set_title(train_kaggle)
#créé les colonnes "onehot" pour les titres
title_onehot = pd.get_dummies(train_kaggle['Title'])
liste_titles_onehot = title_onehot.keys()
title_onehot['PassengerId'] = train_kaggle['PassengerId']
#ajoute les colonnes "onehot" pour les titres aux tableaux de train et de test
train = train.merge(title_onehot.iloc[:len(train),:], how='outer', on='PassengerId')
kaggle = kaggle.merge(title_onehot.iloc[-len(kaggle):,:], how='outer', on='PassengerId')

#calcul et affichage des facteurs de corrélation des variables
correlation = train.corr()


print("\n"+'correlations relatives = '+"\n",correlation['Survived'].sort_values(ascending= False),"\n")
correlation = correlation['Survived'].abs().sort_values(ascending= False)
print('correlations absolues = ',type(correlation),"\n",correlation,"\n","\n")

#def plot_features(X_train, axis_1=0, axis_2=1, predict=None, cmap=plt.cm.Blues_r):
#    plt.figure()
#    plt.title(axis_1+" / "+axis_2)
#    sns.swarmplot(x=axis_1, y=axis_2, data=X_train, color="Blue")
#    plt.show()
#    
#for i, feature_1 in enumerate(correlation.keys()):
#    for j, feature_2 in enumerate(correlation.keys()):
#        if i>1 and j>1 and i is not j and feature_1 in X_cols and feature_2 in X_cols:
#            print('i',i)
#            print('j',j)
#            print('feature_1',feature_1)
#            print('feature_2',feature_2)
#            plot_features(train,feature_1,feature_2)
            


#suppression des colonnes non pertinentes
cols = ['SibSp', 'PassengerId','Embarked']
train = train.drop(cols, axis = 1)
kaggle = kaggle.drop(cols, axis = 1)

#sélection des colonnes pertinentes
#X_cols = ['Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'is_alone', 'Southampton', 'Cherbourg', 'Queenstown']
X_cols = ['Sex', 'Age', 'Fare', 'IsAlone', 'FamilySize', 'Southampton', 'Cherbourg', 'Queenstown','Classe 1','Classe 2','Classe 3']
X_cols.extend(liste_titles_onehot)
print('Colonnes sélectionnées = '+"\n",X_cols,"\n")

#création des tableaux d'entrainement et de cible, des tableaux de validation intermédiaire et le tableau à prédire au final
X = train[X_cols]
y = train['Survived']
X_kaggle = kaggle[X_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("\n",'X_train : ',"\n",type(X_train),"\n",X_train.describe(),"\n\n")
#print(X_train.columns)
X_train_columns = X_train.columns


def plot_axis(X_train, X_test=None, axis_1=0, axis_2=1, predict=None, cmap=plt.cm.Blues_r):
#    rng = np.random.RandomState(17)
#    # fit the model
#    clf = IsolationForest(max_samples=100, random_state=rng)
#    clf.fit(X_train)
#    y_pred_train = clf.predict(X_train)
#    y_pred_test = clf.predict(X_test)
#    
#    # plot the line, the samples, and the nearest vectors to the plane
#    xx, yy = np.meshgrid(np.linspace(X_train[axis_1].min(), X_train[axis_1].max(), len(X_train[axis_1].unique())), np.linspace(X_train[axis_2].min(), X_train[axis_2].max(), len(X_train[axis_2].unique())))
#    print('xx',type(xx),"\n",xx)
#    print('yy',type(yy),"\n",yy)
#    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
#    Z = Z.reshape(xx.shape)
    
#    plt.title(axis_1+" / "+axis_2)
#    #plt.contourf(xx, yy, Z, cmap=cmap)
#    
#    b1 = plt.scatter(X_train[axis_1], X_train[axis_2], c='white',
#                     s=20, edgecolor='k')
#    if X_test is not None:
#        b2 = plt.scatter(X_test[axis_1], X_test[axis_2], c='green',
#                         s=20, edgecolor='k')
#    #c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
#    #                s=20, edgecolor='k')
#    plt.axis('tight')
#    plt.xlim((X_train[axis_1].min(), X_train[axis_1].max()))
#    plt.ylim((X_train[axis_2].min(), X_train[axis_2].max()))
#    plt.legend([b1, b2],
#               [axis_1,axis_2],
#               loc="upper left")
    plt.figure()
    plt.title(axis_1+" / "+axis_2)
    sns.swarmplot(x=axis_1, y=axis_2, data=X_train, color="Blue")
    if X_test is not None:
        sns.swarmplot(x=axis_1, y=axis_2, data=X_test, color="Green")
    plt.show()


#for i, feature_1 in enumerate(correlation.keys()):
#    for j, feature_2 in enumerate(correlation.keys()):
#        if i>1 and j>1 and i is not j and feature_1 in X_cols and feature_2 in X_cols:
#            print('i',i)
#            print('j',j)
#            print('feature_1',feature_1)
#            print('feature_2',feature_2)
#            plot_axis(X_train,X_test,feature_1,feature_2)
            
            

#étalonnage des valeurs
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_kaggle = scaler.transform(X_kaggle)

names = [
        'Logistic Regression',
        'Decision Tree',
        #'Gaussian Naive Bayes',
        'MPLCLassifier',
        #'QuadraticDiscriminantAnalysis',
        'KNeighborsClassifier',
        'SVC linear 0.025',
        'SVC gamma=2 C=1',
        'SVC gamma=3 C=3',
        #'GaussianProcessClassifier',
        'RandomForestClassifier',
        'AdaBoostClassifier',
        'ExtraTreesClassifier',
        'IsolationForest']
classifier = [
        LogisticRegression(C=3),
        tree.DecisionTreeClassifier(),
        #GaussianNB(),
        MLPClassifier(alpha=1),
        #QuadraticDiscriminantAnalysis(),
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        SVC(gamma=3, C=3),
        #GaussianProcessClassifier(1.0 * RBF(1.0)),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        ExtraTreesClassifier(n_estimators=250,random_state=0),
        IsolationForest(max_samples=100, random_state=rng)]

#fonction de cross_validation
best_clf = None
best_clf_name = None
best_score = 0
idx = 0
#clf_results = pd.DataFrame([{'name':'','score':0} for x in range(10)])
clf_results = pd.DataFrame(index=range(len(classifier)), columns=['name','score'])
def run_kfold(clf, name, best_clf, best_score, best_clf_name, clf_results):
    kf = KFold(891, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print(name, " - Fold {0} accuracy: {1}".format(fold, accuracy))
    mean_outcome = np.mean(outcomes)
    std_outcome = np.std(outcomes)
    print("\n",name," - Mean Accuracy: {0}".format(mean_outcome),' +/- ',std_outcome)
    print("\n"+'---------------------------------------------------------'+"\n")
    if mean_outcome > best_score:
        best_score = mean_outcome
        best_clf = clf
        best_clf_name = name
    clf_results.at[idx,'name'] = name
    clf_results.at[idx,'score'] = mean_outcome
    return (best_clf, best_score, best_clf_name)

#entrainement des différents classifieurs
for name, clf in zip(names,classifier):
    best_clf, best_score, best_clf_name = run_kfold(clf, name, best_clf, best_score, best_clf_name, clf_results)
    idx+=1

print('Résultats des classifieurs'+"\n")
clf_results.sort_values(by='score', ascending=False, inplace=True)
print(clf_results,"\n")
if len(clf_results)>1:
    plt.figure()
    sns.barplot(x='score',y='name',data=clf_results,palette="Set1")
    plt.show()
print("\n"+'Best classifier = ',best_clf_name)
print('with score = ',best_score,"\n")

#préparation des différents paramètres du classifieur à tester
'''clf_params['Logistic Regression'] = {'C': [1, 3, 6, 10],
              'dual': [False, True],
              'penalty': ['l2', 'l1'],
              'fit_intercept': [False, True],
              'solver': ['lbfgs', 'liblinear', 'sag', 'saga']
             }'''
clf_params = {}
clf_params['Logistic Regression'] = {'C': [1, 3, 6, 10],
              'dual': [False],
              'penalty': ['l2'],
              'fit_intercept': [False, True],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
             }
clf_params['GaussianProcessClassifier'] = {'n_restarts_optimizer': [0, 1, 2],
              'warm_start': [False, True],
              'random_state': [None, 2]
             }
clf_params['RandomForestClassifier'] = {'n_estimators': [4, 6, 9],
              'max_features': ['log2', 'sqrt','auto'],
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10],
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }
clf_params['AdaBoostClassifier'] = {'n_estimators': [50, 25, 125],
              'learning_rate': [0.5,1,2],
              'algorithm': ['SAMME', 'SAMME.R'],
              'random_state': [None, 2]
             }
clf_params['ExtraTreesClassifier'] = {'n_estimators': [50, 150, 250],
              'criterion': ['gini','entropy']
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)
# Run the grid search
grid_obj = GridSearchCV(best_clf, clf_params[best_clf_name], scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)
# Set the clf to the best combination of parameters
best_clf = grid_obj.best_estimator_
# Fit the best algorithm to the data.
best_clf.fit(X_train, y_train)
#création des prédictions pour la validation intermédiaire
predictions = best_clf.predict(X_test)
print("\n",'Accuracy on test set = ', accuracy_score(y_test, predictions))


#création des prédictions sur l'échantillon à tester
predictions = best_clf.predict(X_kaggle)

#création du tableau de sortie pour Kaggle
output = pd.DataFrame({ 'PassengerId' : PassengerId, 'Survived': predictions })
output = output.set_index('PassengerId')
#output.to_csv('data.csv')

#hack pour l'affichage
kaggle['PassengerId'] = PassengerId
kaggle['Survived'] = predictions


if hasattr(best_clf, 'feature_importances_') and hasattr(best_clf, 'estimators_'):
    importances = best_clf.feature_importances_
    #print('importances',"\n",importances)
    std = np.std([tree.feature_importances_ for tree in best_clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("\n","Feature ranking:")
    for f in range(X_train.shape[1]):
        print("%d. %s (%f)" % (f + 1, X_train_columns[indices[f]], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices], color="b", yerr=std[indices], align="center")
    #plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(X_train.shape[1]), X_train_columns[indices], rotation=45)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()
    
if hasattr(best_clf, 'coef_'):
    importances = best_clf.coef_
    indices = np.argsort(importances)[::-1]
    for f in range(len(X_train_columns)):
        print("%d. %s (%f)" % (f + 1, X_train_columns[indices[0,f]], importances[0,indices[0,f]]))
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances (relative)")
    #plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.bar(range(X_train.shape[1]), importances[0,indices[0]], color="b", align="center")
    plt.xticks(range(X_train.shape[1]), X_train_columns[indices[0]], rotation=45)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()
    
    # Plot the feature importances of the forest
    df_abs = pd.DataFrame(importances[0]).abs()
    indices = np.argsort(np.array([list(df_abs[0])]))
    df_abs_sorted = df_abs.sort_values(by=0,ascending= False)
    importances = np.array([list(df_abs_sorted[0])])
    plt.figure()
    plt.title("Feature importances (absolute)")
    #plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.bar(range(X_train.shape[1]), importances[0,:], color="b", align="center")
    plt.xticks(range(X_train.shape[1]), X_train_columns[indices[0][::-1]], rotation=45)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()
    
    
    
    
    
def plot_contourf_axis(clf, X_train, X_test=None, axis_1=0, axis_2=1, predict=None, cmap=plt.cm.Blues_r):
    # plot the line, the samples, and the nearest vectors to the plane
    xx, yy = np.meshgrid(np.linspace(X_train[axis_1].min(), X_train[axis_1].max(), len(X_train[axis_1].unique())), np.linspace(X_train[axis_2].min(), X_train[axis_2].max(), len(X_train[axis_2].unique())))
    print('xx',type(xx),"\n",xx)
    print('yy',type(yy),"\n",yy)
    
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.title(axis_1," / ",axis_2)
    plt.contourf(xx, yy, Z, cmap=cmap)
    
    b1 = plt.scatter(X_train[axis_1], X_train[axis_2], c='white',
                     s=20, edgecolor='k')
    if X_test is not None:
        b2 = plt.scatter(X_test[axis_1], X_test[axis_2], c='green',
                         s=20, edgecolor='k')
    #c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
    #                s=20, edgecolor='k')
    plt.axis('tight')
    plt.xlim((X_train[axis_1].min(), X_train[axis_1].max()))
    plt.ylim((X_train[axis_2].min(), X_train[axis_2].max()))
    plt.legend([b1, b2],
               [axis_1,axis_2],
               loc="upper left")
    plt.show()

    

#for i in range(len(X_train_columns)):
#    for j in range(i,len(X_train_columns)):
#        plot_contourf_axis(best_clf,X_train,X_test,X_train_columns[i],X_train_columns[j])    

'''
#pour afficher les stats selon les données d'entraînement
data = pd.DataFrame( {'PassengerId' : PassengerId_train, 'Survived' : train['Survived']} )
data = data.set_index('PassengerId')
kaggle = train
kaggle['PassengerId'] = PassengerId_train
kaggle['Survived'] = train['Survived']
'''

liste_titres = train_kaggle['Title'].value_counts().keys()
dict_titres = dict(zip(liste_titres,range(len(liste_titres))))
kaggle = set_title(kaggle)
kaggle['Title_id'] = kaggle['Title'].map(dict_titres)

kaggle['Sexe'] = kaggle['Sex'].map({0:'Femme',1:'Homme'})
#kaggle.sort_values(['Sex','Pclass','Age'])

plt.figure()
sns.barplot(x='Pclass',y='Survived',hue='Sexe',data=kaggle)
plt.figure()

sns.barplot(x='Title',y='Survived',hue='Sexe',data=kaggle)
plt.figure()

sns.barplot(x='Parch',y='Survived',hue='Sexe',data=kaggle)
plt.figure()

sns.barplot(x='IsAlone',y='Survived',hue='Sexe',data=kaggle)
plt.figure()

sns.barplot(x='FamilySize',y='Survived',hue='Sexe',data=kaggle)
plt.figure()

#affichage du modèle appris pour la survie selon le prix du billet et le sexe
bins = [0, 15, 30, 60, 80, 500]
labels = ['0-15', '15-30', '30-60', '60-80', '+80']
fare_groups = pd.cut(kaggle.Fare, bins, labels=labels)
kaggle['fare_group_labeled'] = fare_groups
sns.barplot(x='fare_group_labeled',y='Survived',hue='Sexe',data=kaggle)
plt.figure()

#affichage du modèle appris pour la survie selon le port d'embarcation et le sexe
def get_embark(item):
    if item['Southampton'] == 1:
        return 'Southampton'
    elif item['Queenstown'] == 1:
        return 'Queenstown'
    elif item['Cherbourg'] == 1:
        return 'Cherbourg'
kaggle['Embarked'] = kaggle.apply(get_embark, axis=1)
sns.barplot(x='Embarked',y='Survived',hue='Sexe',data=kaggle)
plt.figure()

#affichage du modèle appris pour la survie selon l'âge et le sexe
bins = [0, 3, 12, 18, 55, 100]
labels = ['baby', 'child', 'teenager', 'adult', 'elder']
age_groups = pd.cut(kaggle.Age, bins, labels=labels)
kaggle['Age_group'] = age_groups
sns.barplot(x='Age_group',y='Survived',hue='Sexe',data=kaggle)
plt.show()

#sns.violinplot(x = "Fare", data=kaggle)
#plt.show()
'''
sns.violinplot(x = "Age", data=kaggle)
plt.figure()
bins = [0, 3, 6, 9, 12, 15, 18, 55, 100]
bins = np.arange(0, 81, 3)
print(bins)
#labels = ['baby', 'child', 'teenager', 'adult', 'elder']
age_groups = pd.cut(kaggle.Age, bins)
kaggle['Age_group2'] = age_groups
sns.barplot(x="Age_group2", y="Survived", hue='Sexe', data=kaggle);
plt.show()'''