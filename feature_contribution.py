# -*- coding: UTF-8 -*-

#################################################################################
#   __author__:       HJ van Veen info@mlwave.com                               #
#   __description__:  This algorithm returns a list of columns, sorted by their #
#                     ‘contribution’ to final classifier output.                #
#                     From "Peering into the Black Box" by Gurjeet Singh        # 
#                     http://www.ayasdi.com/blog/bigdata/5191-2/                #
#   __license__:      CC0 1.0 Universal                                         #
#################################################################################

from sklearn import datasets, ensemble
from sklearn.metrics import accuracy_score
import numpy as np

if __name__ == "__main__":

  # Sample 8x8 digit data with 10 classes
  X, y = datasets.load_digits().data, datasets.load_digits().target

  # Turn into binary classification problem
  X_bin = []
  y_bin = []
  for x, ys in zip(X,y):
    if ys == 0 or ys == 1:
      X_bin.append(x)
      y_bin.append(ys)
  X = np.array(X_bin)
  y = np.array(y_bin)
  print("Data shaped\t%sx%s"%X.shape)
  
  # Create a "complex" ensemble model
  X_train = X[:200]  
  y_train = y[:200]
  X_test = X[200:]
  y_test = y[200:]
  
  clf = ensemble.ExtraTreesClassifier(n_jobs=-1, random_state=1, n_estimators=1000)
  clf.fit(X_train,y_train)
  preds = clf.predict(X_test)
  print("Accuracy score\t%f"%accuracy_score(y_test, preds))
  
  # Imagine that we are given at dataset X which contains N points and each point 
  # in the dataset is a d-dimensional vector.
  # N = 160
  # d = 64
  # In addition, we are given the output of the classifier for each point as an 
  # N-dimensional, binary vector f (preds)
  # Let’s say that X_0 is the subset of X where f is 0 and X_1 is the subset of X 
  # where is 1.
  # Here’s a simple algorithm:
  # For each of the d-dimensions of X, find its mean and standard deviation, call 
  # it m_i and o_i for the i-th dimension respectively.
  m = np.mean(X_test, axis=0)
  o = np.std(X_test, axis=0)
  
  X = np.c_[preds, X_test]
  X_0 = X[X[:,0] == 0][:,1:]
  X_1 = X[X[:,0] == 1][:,1:]
  
  # For each of the d-dimensions of X_0, find its mean. Let’s call it u_i for the 
  # i-th dimension.
  u = np.mean(X_0, axis=0)
  
  # For each of the d-dimensions of X_1, find its mean. Let’s call it v_i for the 
  # i-th dimension.
  v = np.mean(X_1, axis=0)
  
  # Now, sort the list of columns, by |u_i-v_i|/o_i
  column_rank = abs(u-v)/(o+1) # Standard deviation o may contain zero's.
  
  print("\nContribution\tColumn")
  for c in sorted([(f,e) for e, f in enumerate( column_rank )], reverse=True):
    print("%f\t%d"%c)