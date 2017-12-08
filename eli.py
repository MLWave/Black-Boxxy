#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from sklearn import (datasets, metrics, model_selection, 
                     linear_model, ensemble)
import xgboost as xgb                     
import pandas as pd
import numpy as np
import eli5

from collections import defaultdict
import json
import sys

class XGBExplainer(object):
    def __init__(self, verbose=2):
        self.verbose = verbose
    
    def __repr__(self):
        return "XGBExplainer(verbose=%s)"%(self.verbose)
    
    def explain_model(self,
                      model,
                      X,
                      y,
                      feature_names=[],
                      target_names=[],
                      visualize=True,
                      importance_type="gain",
                      temporal=False,
                      max_samples=0):
        
        def _infer_problem_from_model(model):
            if "Classifier" in model.__class__.__name__:
                print("\nClassification Problem")
                return True
            else:
                print("\nRegression Problem")
                return False
            
        def _classification_report(y_test, preds):
            if self.verbose > 0:
                y_mean = np.mean(y_test)
                print("\n%s:\t%f (benchmark: 0.5)"%("ROC AUC".ljust(20),
                                  metrics.roc_auc_score(y_test, preds)))
                print("%s:\t%f (benchmark: %f)"%("Accuracy".ljust(20),
                                  metrics.accuracy_score(y_test,
                                                         preds > 0.5),
                                  y_mean if y_mean > 0.5 else 1-y_mean))
                print("%s:\t%f (benchmark: %f)\n"%("Log Loss".ljust(20),
                                  metrics.log_loss(y_test, preds),
                                  metrics.log_loss(y_test, [y_mean for p in preds])))
                if len(target_names) > 0:
                    print(metrics.classification_report(y_test, preds > 0.5, target_names=target_names))
                else:
                    print(metrics.classification_report(y_test, preds > 0.5))
                print("Confusion Matrix:\n%s"%(metrics.confusion_matrix(y_test, preds > 0.5)))
                    
        def _regression_report(y_test, preds):
            if self.verbose > 0:
                y_mean = np.mean(y_test)
                print("\nStandard Deviation of predictions: %f"%(np.std(preds)))
                print("\n%s:\t%f (benchmark: %f)"%("Mean Absolute Error".ljust(20),
                                  metrics.mean_absolute_error(y_test,
                                                              preds),
                                  metrics.mean_absolute_error(y_test, [y_mean for p in preds])))
                print("%s:\t%f (benchmark: %f)"%("Mean Squared Error".ljust(20),
                                  metrics.mean_squared_error(y_test,
                                                            preds),
                                  metrics.mean_squared_error(y_test, [y_mean for p in preds])))
                print("%s:\t%f (benchmark: %f)"%("R2-Score".ljust(20),
                                  metrics.r2_score(y_test,
                                                   preds),
                                  0.))
                print("%s:\t%f (benchmark: %f)"%("Explained Variance".ljust(20),
                                  metrics.explained_variance_score(y_test,
                                                   preds),
                                  0.))
                
        def _predict(model, X_test, classification):
            if classification:
                return model.predict_proba(X_test)[:,1]
            else:
                return model.predict(X_test)
            
        
        # Verbosity
        if self.verbose > 0:
            print("Explaining model: %s"%(model))
            
        # Infer classification or regression problem
        classification = _infer_problem_from_model(model)
                
        # Split into train and test data
        if self.verbose > 0:
            print("\nSplitting data into 67%% train and 33%% train. Is temporal? %s"%(temporal))
        if temporal:
            cut_off = int(X.shape[0] * 0.67)
            X_train, X_test, y_train, y_test = X[:cut_off],\
                                               X[cut_off:],\
                                               y[:cut_off],\
                                               y[cut_off:]
        else:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
                                                                                y,
                                                                                test_size=0.33,
                                                                                random_state=1)
        # Fitting the model
        if self.verbose > 0:
            print("\nFitting on data shaped %s. Mean target: %f"%(X_train.shape,
                                                                  np.mean(y_train)))
        model.fit(X_train, y_train)
        
        # Check if feature_names was passed or needs default names
        # Else check if length of feature names matches length of fitted features
        fscores = model.booster().get_fscore()
        if len(feature_names) == 0:
            feature_names = fscores.keys()
        else:
            if len(feature_names) != len(fscores):
                sys.exit("`feature_names` length (%d) does not equal length of"%(len(feature_names)) \
                         + " number of features used during fitting (%d)"%(len(fscores)))
        
        # Verbosity
        if self.verbose > 1:
            print("\nUsing %d features: %s"%(len(feature_names), feature_names))
        
        # Calculating feature importances
        d = eli5.formatters.as_dict.format_as_dict(
            eli5.explain_weights_xgboost(model,
                                     feature_names=feature_names,
                                     importance_type=importance_type))
        
        # Format weights and feature names as a sorted list
        l = sorted([(k["weight"], k["feature"]) for k in \
                    d["feature_importances"]["importances"]], reverse=True)
        l_names = [k[1] for k in l]
        for feature_name in feature_names:
            if feature_name not in l_names:
                l.append((0., feature_name))
        l = sorted(l, reverse=True)
        
        # Verbosity
        if self.verbose > 0:
            print("\nImportance: %s\t%s")%(importance_type.ljust(2), "Feature Name".ljust(20))
            print("%s\t%s")%(str("-"*20).rjust(20), str("-"*20).ljust(20))
            for weight, feature in l:
                print("%s\t%s")%(str("%f"%(weight)).rjust(20), str(feature).ljust(20))
        
        # Testing
        if self.verbose > 0:
            print("\nTesting on data shaped %s. Mean target: %f"%(X_test.shape,
                                                                  np.mean(y_test)))
        preds = _predict(model, X_test, classification)

        if classification:
            if self.verbose > 0:
                _classification_report(y_test, preds)
        else:
            preds = model.predict(X_test)
            if self.verbose > 0:
                _regression_report(y_test, preds)
                
        # Violin Density Plots of contributions
        if self.verbose > 0:
            print("\nCreating explanations for test set.")
            
        if max_samples == 0:
            max_samples = X_test.shape[0]
        if max_samples > X_test.shape[0]:
            if self.verbose > 0:
                print("\n`max_samples` larger than test set size." \
                      + " Resetting `max_samples` to %d\n"%(X_test.shape[0]))
            max_samples = X_test.shape[0]
        
        importances = defaultdict(list)
        values = defaultdict(list)    
        
        for i in range(max_samples):
            if self.verbose > 0:
                if i % 100 == 0:
                    print("%d/%d"%(i+1, max_samples))
            
            d = eli5.formatters.as_dict.format_as_dict(
                eli5.explain_prediction(model, X_test[i], feature_names=feature_names))
            
            for k in d["targets"][0]["feature_weights"]["neg"] \
                     + d["targets"][0]["feature_weights"]["pos"]:
                importances[k["feature"]].append(k["weight"])
                values[k["feature"]].append(k["value"])
        
        del importances["<BIAS>"]
        pos = list(reversed(range(len(importances))))
        data = [importances[k] for k in feature_names]
        
        ax = plt.figure(figsize=(16,10))
        plt.axvline(0, linestyle='-', color='k', alpha=0.1)
        plt.violinplot(data, pos, points=max_samples, vert=False, widths=0.7,
                       showmeans=True, showextrema=True, showmedians=False)
        plt.yticks(pos, [k for v, k in l], rotation='horizontal')
        plt.grid(color='b', linestyle='--', linewidth=1, alpha=0.05)
        plt.title("Feature Contributions\n", fontsize=30)
        
        if visualize:
            plt.show()

        if visualize:
            # Feature contribution by value plots
            for feature in feature_names:
                x, ys = [], []
                for val, imp in sorted(zip(values[feature], importances[feature])):
                    x.append(val)
                    ys.append(imp)

                plt.figure()
                plt.scatter(x, ys, alpha=0.5)
                plt.title(feature)
                plt.grid(color='b', linestyle='--', linewidth=1, alpha=0.05)

                trend = ensemble.RandomForestRegressor(random_state=1,
                                                       min_samples_leaf=5)
                trend.fit([[xs] for xs in x], ys)
                p = trend.predict([[xs] for xs in x])
                trend = linear_model.Ridge(random_state=1)
                trend.fit([[xs] for xs in x], ys)
                p2 = trend.predict([[xs] for xs in x])
                p = (p2+p+p)/3.
                plt.plot(x, p, color=(0,0,0), linewidth=5, alpha=0.33)
                plt.ylabel("Feature Contribution")
                plt.xlabel("Feature Value")

                plt.show()

        self.explainer = {
            "pos": pos,
            "data": data,
            "mean_X": np.mean(X, axis=0),
            "std_X": np.std(X, axis=0),
            "mean_y": np.mean(y),
            "model": model,
            "feature_names": np.array(feature_names),
            "target_names": target_names,
            "classification": classification,
            "f_weights": l,
            "max_samples": max_samples
        }
    
    def explain_sample(self, j, y="?"):
        try:
            self.explainer
        except:
            sys.exit("You need to explain the model first with `.explain_model()` before" \
                    + " you can explain examples.")
        
        print("\nExplaining sample shaped %s"%(j.shape))
        d = eli5.formatters.as_dict.format_as_dict(
                eli5.explain_prediction(self.explainer["model"],
                                        j,
                                        feature_names=[k for v, k in self.explainer["f_weights"]]))
        
        # Fetch prediction depending on classification or regression problem
        if self.explainer["classification"]:
            p = d["targets"][0]["proba"]
        else:
            p = d["targets"][0]["score"]
        
        # Fetch scatter points
        s = {}
        for k in d["targets"][0]["feature_weights"]["neg"] \
                     + d["targets"][0]["feature_weights"]["pos"]:
                s[k["feature"]] = k["weight"]
        ds = []
        for i, k in enumerate([k for v, k in self.explainer["f_weights"]]):
            if k in s:
                ds.append(s[k])
            else:
                ds.append(np.nan)

        # Plot violin scatter plot
        ax = plt.figure(figsize=(16,10))
        plt.violinplot(self.explainer["data"],
                       self.explainer["pos"],
                       points=self.explainer["max_samples"],
                       vert=False,
                       widths=0.7,
                       showmeans=True,
                       showextrema=True,
                       showmedians=False)
        plt.yticks(self.explainer["pos"], [k for v, k in self.explainer["f_weights"]],
                   rotation='horizontal')
        plt.grid(color='b', linestyle='--', linewidth=1, alpha=0.05)
        plt.scatter(ds, self.explainer["pos"], color='r')
        plt.title("Target: %s\nPredicted:%f\nMean:%f"%(y, p, self.explainer["mean_y"]))
        plt.ylabel("Feature Importance")
        plt.xlabel("Feature Contribution")
        plt.show()
        
        # Print 1+ STD statistics by order of feature importance
        std_m = np.sqrt((j - self.explainer["mean_X"])**2) / self.explainer["std_X"]
        
        s = {}
        for i, f in enumerate(self.explainer["feature_names"]):
            s[f] = (j[i], std_m[i], self.explainer["mean_X"][i])
        print("\n%s %s %s %s"%("FEATURE".ljust(20),
                             "VALUE".ljust(10),
                             "STD".ljust(10),
                             "MEAN".ljust(10)))
        for w, f in self.explainer["f_weights"]:
            if s[f][1] > 1.:
                print("%s %s %s %s"%(f.ljust(20),
                                     str(s[f][0]).ljust(10),
                                     str("%f"%(s[f][1])).ljust(10),
                                     str("%f"%(s[f][2])).ljust(10)))