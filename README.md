# clusterITE
[![Python](https://img.shields.io/static/v1?label=made%20with&message=Python&color=blue&style=for-the-badge&logo=Python&logoColor=white)](#)
[![license](https://img.shields.io/badge/license-MIT-blue)](https://github.com/fcgrolleau/clusterITE/blob/main/LICENSE)

<img src="figures/clusters.png" align="center" alt="" width="800">

Equipped with a dataset of baseline covariates $`(X_i)_{1\leq i\leq n}`$ and predictions $(Y_i)_{1\leq i\leq n}$ from an estimated individualized treatment effect (ITE, aka CATE) function, the *clusterITE* library lets you estimate the function (i.e., gating network) mapping observations to their probabilities of belonging to clusters $1,\dots,K$ of similar (true) ITE function. The *clusterITE* library, lets you conveniently pick $K$, the optimal number of cluster, via cross-validation. 

Our model and fitting algorithm are described <a href="https://fcgrolleau.github.io/clusterITE/Mixture_of_ITEs.pdf">here</a>.

```python
from clusterITE import *
```

### 1. Specify a model architecture
```python
# For the gating network, define any Keras/Tensorflow architecture of your choice
def custom_tf_model(n_clusters):
    model = Sequential()
    ## Write your favorite architecture here...
    model.add(Dense(10, use_bias=True, activation='relu'))
    model.add(Dense(10, use_bias=True, activation='relu'))
    ## ... but make sure to finish the network like so
    model.add(Dense(n_clusters, use_bias=True, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

# For the expert networks, define any sklearn architecture of your choice
# and store both expert and gating network in a dictonary
base_learners = {'experts': RandomForestRegressor(n_estimators=100, max_depth=10, max_features=10),
                 'gating_net': custom_tf_model}
```
### 2. Pick the optimal number of cluster $K$ via cross-validation
```python
# Instanciate a ClusterIte model with 5 fold cross-validation
cv_model = ClusterIte_cv(nb_folds=5, **base_learners)

# Specify a range for the no. of clusters and fit this model to the data
cv_model.fit(X,y, cluster_range = range(2,16))

# Plot the result
cv_model.plot()
```
 <img src="figures/cv.png" align="center" alt="" width="300" />

### 3. Train a clusterITE model on all the data
```python
# Instanciate a ClusterIte model with the optimal K estimated from cross-validation
final_model = ClusterIte(K=cv_model.best_K, **base_learners)

# Instanciate this model on all the training data
final_model.fit(X, y)
```

### 4. Use your fitted model for cluster prediction and evaluation on unseen data
```python
# Use the gating network of your trained model to predict the probabilities 
# of belonging to clusters 1,...,K for unseen observations
final_model.gate_predict(X_test)

# Evaluate the MSE of the final model on unseen data
mean_squared_error(final_model.predict(X_test), y_test)
````

See implementation details in <a href="https://nbviewer.org/github/fcgrolleau/clusterITE/blob/main/minimal.ipynb">minimal.ipynb</a>.

See more details and a toy simulation in <a href="https://nbviewer.org/github/fcgrolleau/clusterITE/blob/main/clusterITE.ipynb">clusterITE.ipynb</a>.
