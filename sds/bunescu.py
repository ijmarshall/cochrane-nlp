import sklearn
from sklearn.svm import LinearSVC 
from sklearn.grid_search import GridSearchCV

class Bunescu:
    '''
    An implementation of Bunescu & Mooney [2007].
    '''
    def __init__(self, C_range=None, c_p_range=None):
        self.C_range = C_range or np.logspace(.01, 1000, 10)
        self.c_p_range = c_p_range or np.linspace(0, .5, 10)

        self.clf = LinearSVC.

    def fit(self, X, y):
        ''' 
        The objective is 

            1/2 ||w||^2 + C/L * {c_p (L_n/L) fn + c_n (L_p/L) fp}

        we just express this through the class weights
        '''

        L = X.shape[0]
        L_p = y[y>0].shape[0]
        L_n = y.shape[0] - L_p 

        ## 
        # map C and c_p pairs to equivalent class weightings
        ##
        class_weights = []
        for C in self.C_range:
            for c_p in self.c_p_range: 
                # see derivation for where these come from
                c_p_prime = 1/(2*L) * (C * c_p * L_n) 
                c_n = (L_p - c_p * L_p ) / (c_p * L_n)

                class_weights.append({1:c_p_prime, -1:c_n})

        param_grid = [{'class_weight':class_weights}]


        self.clf = GridSearchCV(LinearSVC(), param_grid, scoring="f1")
        self.clf.fit(X,y)

    def predict(self, X):
        return self.clf.predict(X)

    







