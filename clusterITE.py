# %% [markdown]
# # Mixture of Individualized Treatment Effects
# Justification for the appraoch is available [here](https://fcgrolleau.github.io/clusterITE/Mixture_of_ITEs.pdf).

# %% [markdown]
# ### Generate Data

# %%
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

# For the experts we can use random forests
from sklearn.ensemble import RandomForestRegressor

# For simplicity in this parametric simulation, we will use linear regression
from sklearn.linear_model import LinearRegression

# For the gating network, deep learning libraries are more convenient
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from keras.callbacks import EarlyStopping

# Instantiate our gating network
def tf_model(n_clusters):
    model = Sequential()
    model.add(Dense(n_clusters, use_bias=True, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

class ClusterIte():
    def __init__(self, K=3, experts=None, gating_net=None, Callback=None):
        self.K = K
        
        if experts is not None:
            self.experts = {f"ex_mod_{k}": experts for k in range(K)}
        else:
            self.experts = {f"ex_mod_{k}": LinearRegression() for k in range(K)}
        
        if gating_net is not None:
            self.gating_net = gating_net(K)
            if Callback is not None:
                self.Callback = Callback
            else:
                self.Callback = EarlyStopping(monitor='loss', patience=4)
        else:        
            self.gating_net = tf_model(K)
            self.Callback = EarlyStopping(monitor='loss', patience=4)
           
    def fit(self, X, y, maxit=100, epsi=1e-6, verbose=False, silence=False):
        X = X.copy()
        y = y.copy()
                
        # Initialize the cluster probabilities
        self.G = np.ones((len(X), self.K)) / self.K

        # Initialize the individual predictions from the experts
        self.U = np.random.normal(loc=0, scale=10, size=(len(X), self.K))

        # Broadcast our targets in a matrix for ease of use later
        Y = np.repeat(np.array(y)[:, np.newaxis], self.K, axis=1)
        # Iterate over the EM algorithm
        for it in range(maxit):
            
            ### E step ###
            # Compute individual contributions to each expert’s likelihood as
            L = stats.norm.pdf(Y, loc=self.U, scale=1)
            
            # Compute the posterior cluster probabilities as
            H = (self.G * L) / np.sum(self.G * L, axis=1)[:, np.newaxis]

            ### M step ###
            ## 1. Experts substeps
            # Fit all expert models separately
            for k, mod in enumerate(self.experts.values()):
                mod.fit(X, y, sample_weight=H[:, k])

            # Update the predictions from the expert networks as
            self.U = []
            for k, mod in enumerate(self.experts.values()):
                self.U.append(mod.predict(X))
            self.U = np.array(self.U).T
        
            ## 2. Gating network substep
            # Fit the gating network jointly
            self.gating_net.fit(X, H, epochs=50, batch_size=len(X), verbose=0, callbacks=[self.Callback])
            
            # Update the cluster probabilities as
            G_old = self.G
            self.G = self.gating_net.predict(X, verbose=0)
            
             # If verbose, print iterations and accuracies of the gating network
            if verbose and it % 10 == 0:
                print(f'Iteration {it+1} \
                    accuracy on H {(np.argmax(self.G, axis=1) == np.argmax(H, axis=1)).mean()*100:.2f}% \n', \
                    end='\r')
                
            # Check for convergence on the frobenius norm of the difference
            if np.linalg.norm(self.G - G_old, ord='fro') / (len(X) * self.K) < epsi: 
                if silence is False:
                    print(f'Converged at iteration {it+1} with accuracy on H {(np.argmax(self.G, axis=1) == np.argmax(H, axis=1)).mean()*100:.2f}%')
                break
            
            if it == maxit - 1 and silence is False:
                print(f'Reached max no. of iterations = {it+1}')
               
    def predict(self, X, verbose=False):
        pred_Gs = self.gating_net.predict(X, verbose=verbose)
        pred_Us = []
        for k, mod in enumerate(self.experts.values()):
            pred_Us.append(mod.predict(X))
        pred_Us = np.array(pred_Us).T
        
        return np.sum(pred_Gs * pred_Us, axis=1)

from sklearn.model_selection import KFold
from tqdm import tqdm

class ClusterIte_cv:
    def __init__(self, nb_folds=3, **kwargs):
        self.nb_folds = nb_folds
        self.hyperparams = kwargs
        
    def fit(self, X, y, cluster_range=range(2,4), **kwargs):
        X = X.copy()
        y = y.copy()
        
        kf = KFold(n_splits = self.nb_folds)
        res = {}
        for k in tqdm(cluster_range):
            model = ClusterIte(K=k, **self.hyperparams)

            MSEs = []
            # Iterate over the folds
            for train_index, val_index in kf.split(X):
                # Split the data into training and validation sets
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]
                
                # Train your custom model on the training data
                model.fit(X_train, y_train, silence=True, **kwargs)
                
                # Evaluate your model on the validation data
                y_preds = model.predict(X_val, verbose=False)
                cv_mse = ((y_preds - y_val)**2).mean()
                MSEs.append(cv_mse)
            res[f'K={k}'] = MSEs
        self.cv_mse = pd.DataFrame(res)
        
    def best_K_tab_fun(self):
        res = pd.DataFrame({'MSE':self.cv_mse.mean(axis=0), 'se_MSE': self.cv_mse.std(axis=0) / np.sqrt(self.nb_folds)})
        res =res.reset_index().rename(columns={'index': 'K'})
        res['K'] = res['K'].str.extract('(\d+)').astype(int)
        return res.sort_values(by='MSE')
    
    def best_K_fun(self):
        pd_best_K = self.best_K_tab_fun()
        best_K = pd_best_K.iloc[0,0]
        best_K_mse = pd_best_K['MSE'].iloc[0]
        best_K_se = pd_best_K['se_MSE'].iloc[0]
        ub = best_K_mse + best_K_se
        best_K_within_se  = pd_best_K[pd_best_K['MSE'] <= ub].sort_values(by='K').iloc[0,0]
        self.best_K = best_K; self.best_K_within_se = best_K_within_se
        return best_K, best_K_within_se
    
    def plot(self):
        df = self.best_K_tab_fun().sort_values(by='K')
        plt.figure(figsize=(6, 6))
        plt.plot(df['K'], df['MSE'], color='blue')
        plt.fill_between(df['K'], df['MSE']-df['se_MSE'], df['MSE']+df['se_MSE'], color='lightblue')
        plt.xticks(df['K'])
        plt.xlabel('Number of clusters')
        plt.ylabel('MSE')
        plt.title('Crossvalidated MSE ± se')
        self.best_K_fun()
        best_K_tab = self.best_K_tab_fun()
        plt.scatter(self.best_K_within_se, best_K_tab[best_K_tab['K'] == self.best_K_within_se]['MSE'], marker='^', color='g', label='Optima within 1SE')
        plt.scatter(self.best_K, best_K_tab[best_K_tab['K'] == self.best_K]['MSE'], marker='v', color='b', label='Apparent Optima')
        plt.legend()
        plt.show()
