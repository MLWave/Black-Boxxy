from sklearn import datasets, cross_validation, svm, calibration, ensemble, neighbors
from sklearn.metrics import accuracy_score, log_loss
import numpy as np
from collections import Counter
from scipy.misc import imsave

##################################################################################
#   __author__:       HJ van Veen <info@mlwave.com>                              #
#   __description__:  Using Dark Knowledge to explain Black Box predictions with # 
#                     Simple white-box models                                    #
#   __license__:      CC0 1.0 Universal                                          #
##################################################################################

if __name__ == "__main__":

  X, y = datasets.load_digits().data, datasets.load_digits().target
  
  # Create a test set and an explanation set
  X_train, X_test, y_train, y_test = cross_validation.train_test_split(
          X, y, test_size=0.23, random_state=1, stratify=y)
  
  X_test, X_explain, y_test, y_explain = cross_validation.train_test_split(
          X_test, y_test, test_size=0.82, random_state=1, stratify=y_test)

  print("X_train\t\t%sx%s"%(X_train.shape))
  print("X_test\t\t%sx%s"%(X_test.shape))
  print("X_explain\t%sx%s"%(X_explain.shape))

  # Create an ensemble blackbox classifier and predict test and explain set
  clf_svm = svm.SVC(probability=True, kernel="linear", random_state=1)
  clf_svm.fit(X_train, y_train)
  svm_preds = clf_svm.predict_proba(X_test)
  svm_explanations = clf_svm.predict_proba(X_explain)
  
  clf_base = ensemble.ExtraTreesClassifier(n_jobs=-1, n_estimators=1000, random_state=1)
  clf_cet = calibration.CalibratedClassifierCV(base_estimator=clf_base)
  clf_cet.fit(X_train, y_train)
  et_preds = clf_cet.predict_proba(X_test)
  et_explanations = clf_cet.predict_proba(X_explain)

  blackbox_preds = (et_preds + svm_preds) / 2.
  blackbox_explanations = (et_explanations + svm_explanations) / 2.

  print("\n\nSupport Vector Machine with linear kernel")
  print("Accuracy Score:\t%f" % accuracy_score(y_test, np.argmax(svm_preds, axis=1)))
  print("Multi-Log loss:\t%f" % log_loss(y_test, svm_preds))
  
  print("\n\nCalibrated Extremely Randomized Trees")
  print("Accuracy Score:\t%f" % accuracy_score(y_test, np.argmax(et_preds, axis=1)))
  print("Multi-Log loss:\t%f" % log_loss(y_test, et_preds))
  
  print("\n\nBlack Box Ensemble model(average(svm, rf))")
  print("Accuracy Score:\t%f" % accuracy_score(y_test, np.argmax(blackbox_preds, axis=1)))
  print("Multi-Log loss:\t%f" % log_loss(y_test, blackbox_preds))
  
  import random
  ii = random.randint(0,y_test.shape[0])
  # Find n nearest neighbours from explanation set to random new test sample prediction
  x_test, y_sample = X_test[ii], y_test[ii]
  x_pred = blackbox_preds[ii]
  print("\n\nPredicted:\t%s (%f)"%(np.argmax(x_pred), np.max(x_pred)))
  
  neigh = neighbors.NearestNeighbors(20)
  neigh.fit(blackbox_explanations)
  
  darkneigh = neigh.kneighbors([x_pred],20)[1][0] # Indices of old predictions
  darkneigh_distance = np.sum(neigh.kneighbors([x_pred],5)[0][0]) # total distance

  print("Neighbours:\t%s"%(" ".join([str(f) for f in y_explain[darkneigh]])))
  print("Distance:\t%f"%(darkneigh_distance))
  neighbor_img = np.c_[x_test.reshape((8,8)), np.zeros(64).reshape((8,8))]
  for e,p in enumerate(X_explain[darkneigh]):
    neighbor_img = np.c_[neighbor_img, p.reshape((8,8))]
  imsave('explain_digits.png', neighbor_img)
  
  
  # Take n random neighbours from explanation set
  X_random = np.array( random.sample(X_explain,80) )
  X_bin_explain = np.r_[X_explain[darkneigh], X_random]
  
  # Label dark neighbors "1" and label the random samples "0"
  y_bin_explain = np.array([1 for f in range(X_explain[darkneigh].shape[0])] + [0 for f in range(X_random.shape[0])])
  
  # Train shallow decision tree and a linear model
  from sklearn.tree import DecisionTreeClassifier, export_graphviz
  from sklearn.cross_validation import cross_val_score
  from sklearn.metrics import make_scorer, roc_auc_score
  from sklearn.linear_model import LogisticRegression
  clf_tree = DecisionTreeClassifier(random_state=1)
  
  score = cross_val_score(clf_tree, X_bin_explain, y_bin_explain, cv=5, scoring=make_scorer(roc_auc_score))
  print(score, np.mean(score), np.std(score))
  
  clf_lr = LogisticRegression()
  
  score = cross_val_score(clf_lr, X_bin_explain, y_bin_explain, cv=5, scoring=make_scorer(roc_auc_score))
  print(score, np.mean(score), np.std(score))
  
  clf_lr.fit(X_bin_explain, y_bin_explain)
  clf_tree.fit(X_bin_explain, y_bin_explain)
  print(clf_lr.predict([x_test])[0]) # Check if decision tree is eligible (this should predict '1')
  
  # Visualize Logreg features
  l = [0 for f in range(64)]
  w = 16.
  for weight, feature_id in sorted([(v,e) for e,v in enumerate(clf_lr.coef_[0])], reverse=True):
    #print weight, feature_id
    if weight > 0.05:
      l[feature_id] = w
      w -= 1
  imsave('explain_feature_importance.png', np.array(l).reshape((8,8)))

  # Visualize Decision Tree
  with open('tree-explain.dot', 'w') as dotfile:
    export_graphviz(
      clf_tree,
      dotfile)