"""
=================================================
MSIA Solver Class with linear and logistic regression
=================================================
On 2018 march
Author: Drakael Aradan
"""
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
#from sklearn.datasets import make_classification

#fonction utile pour le débugging
def p(mess,obj):
    """Useful function for tracing"""
    if hasattr(obj,'shape'):
        print(mess,type(obj),obj.shape,"\n",obj)
    else:
        print(mess,type(obj),"\n",obj)

#classe abstraite pour les classifieurs ( = modèles prédictifs)
class MSIAClassifier(ABC):
    """Base class for Linear Models"""
    def __init__(self, learning_rate, max_iterations, starting_thetas, range_x, n_samples):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.starting_thetas = starting_thetas
        self.predicted_thetas = None
        self.range_x = range_x
        self.n_samples = n_samples
        self.progression = None
        self.minima = None
        self.maxima = None
        self.mean = None
        self.std = None
        self.ptp = None
        self.scale_mode = 'ptp'
        self.cout_moyen = 1
    
    @abstractmethod
    def fit(self, X, Y, max_time=False, tic_time=0):
        """Fit model."""
        pass
        
    @abstractmethod
    def predict(self, X):
        """Predict using the linear model
        """
        pass

    @abstractmethod
    def regression_cost(self, model, theta, X, Y):
        """Calculate and return cost
        """
        pass

    @abstractmethod
    def randomize_model(self, theta, X, range_x, random_ratio=0.0, offsets=None):
        """Calculate target for initial training set and weights
        """
        pass

    @abstractmethod
    def plot_1_dimension(self, X, Y):
        """Plot visual for 1 dimentionnal problem
        """
        pass

    def init_attribs_from_X(self, X):
        """Stores means, stds and ranges for X
        """
        self.range_x = np.max(np.abs(X))
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.ptp = X.ptp(axis=0)
        self.minima = X.min(axis=0)
        self.maxima = X.max(axis=0)
        return self
    
    def get_mean(self):
        return self.mean
    
    def get_std(self):
        return self.std
    
    def get_ptp(self):
        return self.ptp
    
    #fonction de normalisation par minimum et maximum
    def scale_(self, X):
        """Scale X values with min and max
        """
        self.init_attribs_from_X(X)
        for i in range(X.shape[1]):
            min_ = np.min(X[:,i])
            max_ = np.max(X[:,i])
            X[:,i]-=min_
            X[:,i]/=max_-min_
        return X
    
    #fonction de normalisation par moyenne (=mean) et plage de valeurs (=range)
    def scale(self, X, on='ptp'):
        """Scale X values with means and ranges
        """
        self.init_attribs_from_X(X)
        self.scale_mode = on
        if self.scale_mode == 'ptp':
            X = (X - self.mean) / self.ptp
        else:
            X = (X - self.mean) / self.std
        return X
    
    #fonction de remise à l'échelle des poids prédis selon la normalisation initiale
    def rescale(self):
        """Rescale weights to original scale
        """
        array = []
        theta_zero = self.predicted_thetas[0].copy()
        for col, mean, std, ptp in zip(self.predicted_thetas[1:], self.mean, self.std, self.ptp):
            if self.scale_mode == 'ptp':
                theta_i = float((col) / ptp)
            else:
                theta_i = float((col) / std)
            array.append(theta_i)
            theta_zero -= theta_i * mean
        self.predicted_thetas = np.array([float(theta_zero),]+array).reshape(len(self.predicted_thetas),1)
        return self

    def linear_regression(self, theta, x):
        """linear regression method
        """
        if isinstance(x, int):
            if theta.shape[0]==len(x)+1:
                x = np.concatenate([[1,],x])
        elif type(x).__module__ == np.__name__:
            if len(x.shape) == 1:
                x = x.reshape(1,x.shape[0])
            elif len(x.shape) == 0:
                x = np.array(x).reshape(1,1)
            if theta.shape[0]==x.shape[1]+1:
                x = np.column_stack((np.ones(len(x)),x))
        else:
            print('different type!!!!',type(x))
        return np.matmul(x,theta) 

    def get_cost_derivative(self, model, theta, X, Y):
        """cost derivative calculation
        """
        if theta.shape[0]==X.shape[1]+1:
            X = np.column_stack((np.ones(self.n_samples),X))
        result = []
        diff = model(theta, X)-Y
        diff_reshaped = diff.reshape(1,X.shape[0])
        for i, t in enumerate(theta):
            #deriv = (1/self.n_samples) * np.matmul((model(theta, X)-Y).reshape(1,X.shape[0]),X[:,i])
            result.append(np.matmul(diff_reshaped,X[:,i]) / self.n_samples)
        return np.array(result).reshape(len(result),1)

    def get_cost_derivative_(self, model, theta, X, Y):
        """cost derivative calculation from Achille
        """
        if theta.shape[0]==X.shape[1]+1:
            X = np.column_stack((np.ones(self.n_samples),X))
        diff_transpose = (model(theta, X)-Y).T
        return np.array([np.sum(np.matmul(diff_transpose, X[:,i]))/(X.shape[0]) for i, t in enumerate(theta)]).reshape(len(theta),1)

    def plot_progression(self):
        """plot learning progression
        """
        if self.progression is not None:
            plt.plot(self.progression, label='progression')
            plt.legend()
            plt.show()

    def gradient_descent(self, initial_model, X, Y, max_iterations, alpha, starting_thetas=None, max_time=0, tic_time=None):
        """performs gradient descent
        """
        self.n_samples = len(X)
        if starting_thetas is None:
            self.starting_thetas = np.random.random((X.shape[1]+1,1))
        self.predicted_thetas = self.starting_thetas
        self.progression = []
        cnt = max_iterations
        cout = 1
        if tic_time is None:
            tic_time = datetime.now()
        while cnt > 0 and ((max_time==0) or (datetime.now()-tic_time).seconds<max_time):# and (len(self.progression)<=100 or (len(self.progression)>100 and np.abs(self.cout_moyen)>0.00000001)):#np.abs(cout) > 0.00000001 and 
            iteration = self.get_cost_derivative(initial_model, self.predicted_thetas, X, Y)
            iteration*= alpha
            self.predicted_thetas = self.predicted_thetas - iteration
            cout = self.regression_cost(initial_model, self.predicted_thetas, X, Y)
            self.progression.append(cout)
            self.cout_moyen = np.mean(self.progression[-100:])
            cnt-=1
        #self.plot_progression()
        #self.plot_1_dimension(X, Y)
        return self.predicted_thetas

#Classe de régression linéaire
class LinearRegression(MSIAClassifier):
    """Linear Regression Class
    """
    def __init__(self, learning_rate=3*10**-1, max_iterations=4000, starting_thetas = None, range_x = 1, n_samples = 0):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.starting_thetas = starting_thetas
        self.range_x = range_x
        self.n_samples = n_samples
        MSIAClassifier.__init__(self, self.learning_rate, self.max_iterations, self.starting_thetas, self.range_x, self.n_samples)

    def fit(self, X, Y, max_time=False, tic_time=0):
        """Linear Regression Fit
        """
        X = self.scale(X)
        self.predicted_thetas = self.gradient_descent(self.linear_regression, X, Y, self.max_iterations, self.learning_rate, max_time=max_time, tic_time=tic_time)
        self.rescale()
        return self

    def predict(self, X):
        """Linear Regression Prediction
        """
        return self.linear_regression(self.predicted_thetas, X)

    def regression_cost(self, model, theta, X, Y):
        """Linear Regression cost calculation
        """
        diff = (model(theta, X)-Y)
        return float(1/(2 * len(X)) * np.matmul(diff.T,diff))
        #return float(np.sum(np.abs(model(theta,X)-Y))/ (self.n_samples * X.shape[1] * self.range_x))
        #return float(np.sum(np.abs(model(theta,X)-Y))/ (self.n_samples * X.shape[1])) 

    def randomize_model(self, theta, X, range_x, random_ratio=0.0, offsets=None):
        """Linear Regression randomize function
        """
        self.n_samples = X.shape[0]
        if theta.shape[0]==X.shape[1]+1:
            X = np.column_stack((np.ones(self.n_samples),X))
        produit = np.matmul(X,theta)
        #option de random pour ajouter du bruit statistique
        if random_ratio != 0.0:
            produit+= (np.random.random(produit.shape)-0.5)*range_x*random_ratio
        return produit

    def plot_1_dimension(self, X, Y):
        """Linear Regression 1 dimensionnal plot
        """
        if(len(self.predicted_thetas)==2):
            plt.figure()
            plt.plot(X, Y  , 'o', label='original data')
            plt.plot(X, self.predicted_thetas[0] + self.predicted_thetas[1]*X, 'r', label='fitted line')
            plt.legend()
            plt.show()



class LogisticRegression(MSIAClassifier):
    """Logistic Regression Class
    """
    def __init__(self, learning_rate=0.5, max_iterations=4000, predicted_thetas = None, range_x = 1, n_samples = 0):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.predicted_thetas = predicted_thetas
        self.range_x = range_x
        self.n_samples = n_samples
        MSIAClassifier.__init__(self, self.learning_rate, self.max_iterations, self.predicted_thetas, self.range_x, self.n_samples)

    def fit(self,X,Y, max_time=False, tic_time=0):
        """Logistic Regression Fit
        """
        X = self.scale(X,'std')
        self.predicted_thetas = self.gradient_descent(self.sigmoid, X, Y, self.max_iterations, self.learning_rate, max_time=max_time, tic_time=tic_time)
        self.predicted_thetas/= np.absolute(self.predicted_thetas[:,0]).max()
        return self

    def predict(self,X):
        """Logistic Regression Prediction
        """
        array = self.sigmoid(self.predicted_thetas, X)
        array = list(map(lambda x: 1 if x >= 0.5 else 0, array[:,0]))
        return array

    def sigmoid(self, theta, x):
        """Logistic Regression Sigmoid function
        """
        sigmoid = 1/(1+np.exp(self.linear_regression(theta, x)*-1))
        if sigmoid.shape == (1,1):
            sigmoid = sigmoid[0][0]
        return np.round(sigmoid,2)

    def regression_cost_(self, model, theta, X, Y):
        """Logistic Regression Cost calculation (heavy)
        """
        somme = 0
        for x, y in zip(X, Y):
            sig = np.clip(self.sigmoid(theta,x),0.00000001,0.99999999)
            somme+= (y * np.log(sig)) + ( (1-y) * np.log(1 - sig) )  
        somme/= -self.n_samples    
        return float(somme) 
    
    def regression_cost__(self, model, theta, X, Y):
        """Logistic Regression Cost calculation (heavy)
        """
        somme = 0
        for x, y in zip(X, Y):
            sig = np.clip(self.sigmoid(theta,x),0.00000001,0.99999999)
            somme+= (y * ((1/sig)-1))+ ( (1-y) * ((1/(1-sig))-1) )  
        somme/= self.n_samples    
        return float(somme) 

    def regression_cost(self, model, theta, X, Y):
        """Logistic Regression Cost calculation (regular)
        """
        #cout = float(np.sum(np.absolute(model(theta,X)-Y))/ self.n_samples) 
        cout = float(np.absolute((model(theta,X)-Y)**2).mean())
        return cout

    def randomize_model(self, theta, X, range_x, random_ratio=0.0, offsets=None):
        """Logistic Regression Randomize function
            TODO: test sur random_ratio qui doit être entre 0.0 et 1.0
        """
        self.n_samples = X.shape[0]
        if theta.shape[0]==X.shape[1]+1:
            X = np.column_stack((np.ones(self.n_samples),X))
        produit = []
        for x in X:
            sig = self.sigmoid(theta,x.reshape(1,len(x)))
            val = 1 if sig > 0.5 else 0
            #option de random pour ajouter du bruit statistique
            if random_ratio != 0.0:
                val = val if np.random.random() < random_ratio else 1 - val
            produit.append(val)
        return np.array(produit,order='F').reshape(self.n_samples,1)

    def plot_1_dimension(self, X, Y):
        """Logistic Regression 1 dimensionnal plot
        """
        if(len(self.predicted_thetas)==2):
            plt.figure()
            plt.plot(X, Y, 'o', label='original data')
            x = np.linspace(-self.range_x/2,self.range_x/2,100)
            y = []
            for var in x:
                sig = self.sigmoid(self.predicted_thetas, var)
                y.append(sig)
            plt.plot(x,y,'r')
            plt.legend()
            plt.show()
            
            
class MSIASolver():
    """Solver class
    """
    def __init__(self, max_iterations=500, randomize=0.0, max_time=50, tic_time=None):
        self.__max_iterations = max_iterations
        self.__randomize = randomize
        self.__max_time = max_time
        self.__tic_time = tic_time
        self.__true_weights = None
        self.__clf = None
        self.__X = None
        self.__Y = None
        self.__n_dimensions = None
        self.__b_samples = None
        self.__use_classifier = None
        self.__range_x = None
       
    def __format_array(self, array):
        if type(array).__module__ == np.__name__:
            if len(array.shape) == 1:
                array = array.reshape(array.shape[0],1)
            elif len(array.shape) == 0:
                array = np.array(array).reshape(1,1)
        else:
            print('different type!!!!',type(X))
        return array
        
        
        
    def fit(self, X, Y, max_time=False):
        """Solver Fit
        """
        X = self.__format_array(X)
        Y = self.__format_array(Y)
        self.__X = X.copy()
        n_samples, n_dimensions = X.shape
        self.set_n_samples(n_samples)
        self.set_n_dimensions(n_dimensions)
        self.__range_x = np.max(np.abs(X))
        self.__Y = Y.copy()
        #todo: tests sur les données
        self.__choose_classifier()
        self.__clf.init_attribs_from_X(self.__X)
        self.__clf.fit(self.__X, self.__Y, max_time or self.__max_time, self.__tic_time)
        self.__clf.plot_1_dimension(X, Y)
        
        return self

    def predict(self,X):
        """Solver Prediction
        """
        return self.__clf.predict(X)
    
    def __choose_classifier(self):
        """Solver: automatic classifier choice
        """
        if self.__clf == None:
            if(self.__Y.shape[1]==1):
                self.__use_classifier = 'LinearRegression'
                min_ = self.__Y.min(axis=0)
                if (self.__Y.dtype == 'int32' or self.__Y.dtype == 'int64' or self.__Y.dtype == 'bool') and min_ >= 0:
                    unique = np.unique(self.__Y.astype(float))
                    test = True
                    for item in unique:
                        if item.is_integer() == False:
                            test = False
                    if test == True:
                        self.__use_classifier = 'LogisticRegression'
            if self.__use_classifier == 'LogisticRegression':
                self.__clf = LogisticRegression(max_iterations=self.__max_iterations)
            else:
                self.__clf = LinearRegression(max_iterations=self.__max_iterations)
                    
    def set_learning_rate(self, learning_rate):
        if self.__clf != None:
            self.__clf.learning_rate = learning_rate
            
    def set_max_iterations(self, max_iterations):
        if self.__clf != None:
            self.__clf.max_iterations = max_iterations
            
    def set_predicted_thetas(self, predicted_thetas):
        if self.__clf != None:
            self.__clf.predicted_thetas = predicted_thetas
            
    def set_range_x(self, range_x):
        if self.__clf != None:
            self.__clf.range_x = range_x
            
    def set_n_samples(self, n_samples):
        self.__n_samples = n_samples
        if self.__clf != None:
            self.__clf.n_samples = n_samples
            
    def set_n_dimensions(self, n_dimensions):
        self.__n_dimensions = n_dimensions
        if self.__clf != None:
            self.__clf.n_dimensions = n_dimensions
            
    def get_starting_thetas(self):
        if self.__clf != None:
            return self.__clf.starting_thetas
        return None
    
    def get_predicted_thetas(self):
        if self.__clf != None:
            return self.__clf.predicted_thetas
        return None
    
    def get_mean(self):
        return self.__clf.get_mean()
    
    def get_std(self):
        return self.__clf.get_std()
    
    def get_ptp(self):
        return self.__clf.get_ptp()
    
    def get_range_x(self):
        return self.__range_x
    
    def get_classifier(self):
        return self.__use_classifier
                    
    def severe_randomizer(self, class_type='LinearRegression', n_samples=50, n_dimensions=10, range_x = 10000):
        #calcul aléatoire de poids pour le modèle théorique
        self.__true_weights = (np.random.random((n_dimensions+1,1))*range_x)-(range_x/2)
        self.__range_x = range_x
        X = []
        self.n_samples = n_samples
        rand_offsets = []
        if class_type == 'LogisticRegression':
            self.__use_classifier = 'LogisticRegression'
            self.__clf = LogisticRegression(max_iterations=self.__max_iterations)
            for i in range(n_samples):
                row = []
                for j in range(n_dimensions):
                    value = (np.random.random()-0.5)*range_x
                    row.append(value)
                X.append(row)
        else:
            self.__use_classifier = 'LinearRegression'
            self.__clf = LinearRegression(max_iterations=self.__max_iterations)
            degre = np.floor(np.log10(range_x))
            if degre < 1:
                degre = 1
            rand_categories = []
            for i in range(n_dimensions):
                rand_category = np.random.randint(-1,degre)
                rand_categories.append(rand_category)
                #rand_offset = np.random.randint(0,degre)-((degre-1)/2)
                rand_offset = (np.random.random()-0.5)*10**(rand_category+2)
                rand_offsets.append(rand_offset)
                
            for i in range(n_samples):
                row = []
                for j in range(n_dimensions):
                    value = (np.random.random()-0.5)*range_x
                    value*=10**(rand_categories[j])
                    value-= rand_offsets[j]
                    row.append(value)
                X.append(row)
        
        X = np.array(X)
        Y = self.__clf.randomize_model(self.__true_weights, X, range_x, self.__randomize, rand_offsets) 
        if self.__use_classifier == 'LogisticRegression':
            self.__true_weights/= np.absolute(self.__true_weights[:,0]).max()
        return X, Y, self.__true_weights


#on ferme toutes les éventuelles fenêtres
# plt.close('all')
# #on enregistre le temps courant
# # = on démarre le chronomètre
# tic_time = datetime.now()

# #variables de base
# n_dimensions = 100
# n_samples = 666
# range_x = 1000
# max_iterations = 1500 
# randomize = 0.1 
# max_execution_time = 111
# true_weights = None

#déclaration du solveur
#solver = MSIASolver(max_iterations, randomize, max_execution_time, tic_time)

#initialisation aléatoire du set d'entrainement
#X, Y, true_weights = solver.severe_randomizer('LinearRegression', n_samples, n_dimensions, range_x)

#X = np.random.normal(0, 2, 100)
#Y = np.random.normal(0, 2, 100)

#from sklearn.datasets import fetch_california_housing
#dataset = fetch_california_housing()
#X, Y = dataset.data, dataset.target
#n_samples, n_dimensions = X.shape
#Y = Y.reshape(len(Y),1)

#degre = np.floor(np.log10(range_x))
#if degre < 1:
#    degre = 1
#rand_categories = []
#rand_offsets = []
#for i in range(n_dimensions):
#    rand_category = np.random.randint(0,degre)
#    rand_categories.append(rand_category)
#    #rand_offset = np.random.randint(0,degre)-((degre-1)/2)
#    rand_offset = (np.random.random()-0.5)#*10**rand_category
#    rand_offsets.append(rand_offset)
#X, Y = make_classification(n_samples=n_samples,
#                           n_features=n_dimensions,
#                           n_informative=n_dimensions,
#                           #scale=range_x,
#                           shift=rand_offsets,
#                           n_redundant=0,
#                           n_repeated=0,
#                           n_classes=2,
#                           random_state=np.random.randint(100),
#                           shuffle=False)
#Y = Y.reshape(len(Y),1)

#affichages préliminaires
# p('X',X)
# p('true_weights',true_weights)
# #p('theta_initial',theta_initial)
# p('Y',Y)

# #entrainement du modèle sur les données
# solver.fit(X, Y)

# #affichage finaux
# predicted_thetas = solver.get_predicted_thetas()
# print('Theta start',"\n",solver.get_starting_thetas())   
# if true_weights is not None:
#     print("Theta target\n",true_weights) 
# print('Theta end : ',"\n",predicted_thetas)
# #if type(clf).__name__ == 'LinearRegression':
# print('Means : ',"\n",solver.get_mean())
# print('StDs : ',"\n",solver.get_std())
# print('Ranges : ',"\n",solver.get_ptp()) 
# if true_weights is not None:
#     print('Erreurs : ',"\n",true_weights-predicted_thetas)
#     print('Erreur globale : ',"\n",np.sum(true_weights-predicted_thetas))
#     print('Erreur moyenne : ',"\n",np.sum(true_weights-predicted_thetas)/(len(X)))
# range_x = solver.get_range_x()
# print('Range of values :',range_x)
# print('Solver :',solver.get_classifier())
# #arrêt du chronomètre
# delta_time = (datetime.now()) - tic_time
# #affichage du temps de calcul global
# print('Script executed in',delta_time.days,'d',delta_time.seconds,'s',delta_time.microseconds,'µs')
