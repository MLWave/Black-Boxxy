# Black Boxxy

Some experiments into explaining complex black box ensemble predictions.

## Feature contribution

`feature_contribution.py` is an implementation of the algorithm described in "[Peering into the Black Box](http://www.ayasdi.com/blog/bigdata/5191-2/ "Peering into the Black Box")" by Gurjeet Singh.

It sorts columns by contribution to a model's predictions. For our black box ensemble model we pick Extremely Randomized Trees with a 1000 estimators. For our data we use an 8x8 digit dataset with 2 classes.

### Console
```
Data shaped     360x64
Accuracy score  0.993750

Contribution    Column
1.654096        28
1.645041        20
1.607345        36
1.452777        44
1.356622        19
1.354579        38
1.330869        27
1.315076        30
1.278320        35
1.257359        43
1.191988        50
1.165835        33
1.156170        10
1.139788        46
1.087283        42
0.966231        53
0.933879        41
0.904047        45
0.861478        21
0.850846        18
0.845671        34
0.821411        12
0.697953        26
0.692177        59
0.684659        22
0.625493        25
0.590022        58
0.583654        2
0.563406        13
0.525454        37
0.495914        29
0.467407        3
0.452016        52
0.435204        17
0.422424        5
0.359402        54
0.358895        51
0.348001        49
0.335544        9
0.280788        62
0.238360        6
0.221660        11
0.166321        63
0.073130        55
0.062626        61
0.040910        1
0.035090        14
0.021596        16
0.019251        4
0.011810        60
0.011587        24
0.000000        57
0.000000        56
0.000000        48
0.000000        47
0.000000        40
0.000000        39
0.000000        32
0.000000        31
0.000000        23
0.000000        15
0.000000        8
0.000000        7
0.000000        0
```

## Dark Knowledge Neighbourhood

`dark_knowledge_neighbourhood.py` finds previous (explained) multi-class predictions which are close to the current prediction.

For data we use an 8x8 digit dataset with 10 classes. For our black box model we use an average of SVM with a linear kernel and Extremely Randomized Trees with a 1000 estimators.

### Algorithm description and output

This method introduces an "Explanation set". This is a kind of test set used to explain black box predictions. This explanation set may be labeled 
with descriptive explanations (using experts or crowd-sourcing). We store the multi-class predictions for this explanation set.

```
X_train         1383x64
X_test          75x64
X_explain       339x64

Support Vector Machine with linear kernel
Accuracy Score: 0.986667
Multi-Log loss: 0.079983


Calibrated Extremely Randomized Trees
Accuracy Score: 0.986667
Multi-Log loss: 0.070617


Black Box Ensemble model(average(svm, rf))
Accuracy Score: 0.986667
Multi-Log loss: 0.074331
```

When we receive a new test sample (![Digit 9](http://i.imgur.com/F39NgrP.png)) we create multi-class probability prediction for it with the black-box model.

```
Predicted:      9 (0.944547)
```

We then find n nearest neighbours from the explanation set (Note: we rank top n samples that received a similar prediction, we only look at predictions, not features).

We show the nearest neighbours labels (ground truth) and check if these are accurate/consistent. We also report the total distance between the test sample predictions and the nearest neighbour predictions (lower is better, if too high, increase the explanation set size).

```
Neighbours:     9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9
Distance:       0.062994
```

![Explain digits](http://i.imgur.com/QzMPLUu.png)

If the explanation set is labeled with descriptions we present the first few:

```
Explanation:    "Single loop. Full bottom curve. Slightly slanted to the left. Looks like the digit 'nine'."
```

Now we take n random samples from the explanation set and add these to the nearest prediction neighbours to create a small dataset. We label the nearest prediction samples `1` and the random samples `0`.

We then train a shallow decision tree and logistic regression (two white-box models) on this small dataset (using the original features) and report 5-fold cross-validation AUC score.

```
Decision Tree:  0.55625 AUC (+/- 0.08926)
LogReg:         0.76250 AUC (+/- 0.08970)
```

A white-box explanation model is eligible when it correctly predicts the new test sample, and has decent generalization (CV-score).

```
LogReg predicts `1` for new test sample, thus is eligible.
Decision Tree predicts `1` for new test sample, thus is eligible.
```

You can build different seeded decision trees, 'till you get an eligible model with decent generalization.

Finally, we visualize the explanation models. The Decision Tree is turned into a GraphViz file: 

![Decision Tree](http://i.imgur.com/yzsVTiX.png) 

and the LogReg model shows the features that were indicative of the sample being labeled "1": ![LogReg Feature Importance](http://i.imgur.com/WO6DHTW.png)

### Intuition

We have a 3-class non-linear classification problem with a single feature x (age) and a bit of noise added.
```
y A A A B B B C C A A
x 0 1 2 3 4 5 6 7 8 9 
```
For instance, when x == 6 then y == class C.

We have as an explanation model a simple linear model that can do only a single split. Such a model is not able to fully separate class A from class B and C (it needs at least two splits for this, `x < 3 and x > 7`).

We have a black-box model that can separate the three classes perfectly and can output multi-class predictions.

A new test sample comes in where x == 8. The black-box model predicts [0.84, 0.01, 0.15] or 84% probability of class A, 1% of class B and 15% of class C.

We find 10 nearest prediction neighbours from the explanation set. These samples likely have similar features to the new test sample. We also take 10 random samples from the explanation set.

After turning the problem into a binary classification problem we may get:

```
y 0 0 0 0 0 0 0 1 1 1
x 0 1 2 3 4 5 6 7 8 9 
```

This we *can* separate with a single split (x > 6? 1 else 0). We used the dark knowledge of a black box ensemble model to create a much simpler linear sub-problem. Neighbouring samples were found using the fact that class "C" was deemed more probable than class "B".

The white-box explanation may be: "Black-box predicted an A, because the variable 'age' is larger than 6.". Not perfect, but close enough.

### Zero-shot

We leave the class 'digit 9' from the train set and retrain our ensemble black box model. We create multi-class predictions for the explanation set (this explanation set does contain samples of class 'digit 9'). 

Now for new test samples labeled 'digit 9' we ask the ensemble to explain its prediction neighbourhood.

```
Predicted:      3 (0.803607)
Neighbours:     [9] 9 9 9 9 9 9 9 3 3 9 3 3 3 3 3 3 3 3 3
Counter({3: 11, 9: 9})
Distance:       0.449837

Predicted:      3 (0.632486)
Neighbours:     [9] 9 9 9 9 9 9 5 9 9 3 9 3 9 9 9 3 9 9 9
Counter({9: 16, 3: 3, 5: 1})
Distance:       0.748975

Predicted:      3 (0.559741)
Neighbours:     [5] 9 9 9 9 9 9 9 9 9 9 3 9 9 3 9 9 9 9 9
Counter({9: 17, 3: 2, 5: 1})
Distance:       0.558236

Predicted:      3 (0.632028)
Neighbours:     [9] 9 9 9 9 5 9 9 3 9 9 3 9 9 9 9 3 9 9 9
Counter({9: 16, 3: 3, 5: 1})
Distance:       0.524373

Predicted:      3 (0.808942)
Neighbours:     [9] 3 9 3 9 9 9 9 9 9 3 9 3 3 3 3 3 3 3 3
Counter({3: 11, 9: 9})
Distance:       0.348610

Predicted:      3 (0.750614)
Neighbours:     [9] 9 9 9 3 9 9 9 9 9 3 9 3 9 3 3 9 3 9 3
Counter({9: 13, 3: 7})
Distance:       0.444025

Predicted:      5 (0.814363)
Neighbours:     [5] 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5
Counter({5: 20})
Distance:       0.515372
```

5 out of 7 times, the 1-nearest prediction neighbour was able to present the correct class -- using predictions from a black box model that has never seen that class during training.

![zeroshot](http://i.imgur.com/1T8D1cv.png) 

### Mapping Dark Knowledge

Treat the output of a black box ensemble as a TDA function and apply Mapper on this. For data use 10-class 8x8 digit data set, for black-box ensemble Calibrated Extremely Randomized Trees with a 1000 estimators, and for mapping KeplerMapper.

We create out-of-5-fold 10-class probability predictions for the entire data set.

We try to visualize how a black-box model predicts 'digit 9'. 

- We remove all samples where black-box did not predict a 9. 
- We calculate the mean of the predictions
- We calculate the absolute difference between a sample prediction and the mean
- The ninth column abs('prediction for 9'-'mean prediction for 9') becomes our lens.
- For clustering we use k-means++ with 3 clusters, 8 overlapping intervals, and 90% overlap.

![topology](http://i.imgur.com/RMnJUKk.png) 

We can spot which samples are troublesome for the black-box. We can spot different types of 'digit 9' that received a similar prediction.

Using only the prediction column for "digit 1", one can also use Mapper to guide a cluster algorithm on the inverse image:

![topology](http://i.imgur.com/cJN5ohG.png) 

We were able to produce pleasing clusters for other digits than "digit 1", using black-box predictions for "digit 1".

Another example where clear regions with similar digits appeared:

![topology](http://i.imgur.com/WgvU6G5.png) 

Nodes with the digits '8' appeared, solely in relation to how much the black-box predicted these digits looked like the digit "1".

### References

Eligible models is from Turner's Model Explanation System. Stacked Generalization and State Space Compression is from Wolpert et al. Dark Knowledge and Semantic Output Code classification is Hinton et al. transfer and zero-shot learning. Nearest predictions is an idea from Enlitic (they use layer activations for this). Zero-data learning of digits is Larochelle et al.. Mapper is Singh et al..

## XGBoost Decision Paths

See `eli.py`. We use ELI5 library to provide explainations for XGBoost models and model predictions. This replicates to a large degree the excellent post [Interpreting Decision Trees and Random Forests](http://engineering.pivotal.io/post/interpreting-decision-trees-and-random-forests/) by Greg Tam and extends it to gradient boosted decision trees.

### Model Explanations

```Python
from sklearn import datasets

X, y, f = datasets.load_boston().data,
          datasets.load_boston().target,
          datasets.load_boston().feature_names

model = xgb.XGBRegressor(seed=0)

explainer = XGBExplainer(verbose=2)
explainer.explain_model(model,
                        X,
                        y,
                        feature_names=f,
                        max_samples=260,
                        visualize=True,
                        temporal=True)
```

```
Explaining model: XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0,
       learning_rate=0.1, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
       objective='reg:linear', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True, subsample=1)

Regression Problem

Splitting data into 67% train and 33% train. Is temporal? True

Fitting on data shaped (339, 13). Mean target: 25.142183

Using 13 features: ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
 'B' 'LSTAT']

Importance: gain        Feature Name
--------------------    --------------------
            0.687155    RM
            0.122762    LSTAT
            0.032781    TAX
            0.031916    PTRATIO
            0.025528    NOX
            0.020725    DIS
            0.019334    INDUS
            0.017637    AGE
            0.011368    B
            0.010895    CRIM
            0.007768    CHAS
            0.007123    RAD
            0.005006    ZN

Testing on data shaped (167, 13). Mean target: 17.235928

Standard Deviation of predictions: 5.028404

Mean Absolute Error :	4.519644 (benchmark: 5.770985)
Mean Squared Error  :	45.417754 (benchmark: 66.116434)
R2-Score            :	0.313064 (benchmark: 0.000000)
Explained Variance  :	0.351315 (benchmark: 0.000000)

Creating explanations for test set.

`max_samples` larger than test set size. Resetting `max_samples` to 167

1/167
101/167
```
![img](https://i.imgur.com/nWKxleD.png)

And for every feature:

![img](https://i.imgur.com/FUiTqXT.png)

#### Edit: Important Note

Always check the feature descriptions (if available). Use of feature `B` would be unethical as race is a protected variable in many jurisdictions.

```
Boston house prices dataset
---------------------------

    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's
```

### Sample explanation

```Python
df = pd.read_csv("credit-card-default.csv")
f = [f for f in df.columns if f not in ["default payment next month"]]
X, y = np.array(df[f]), np.array(df["default payment next month"])

model = xgb.XGBClassifier(seed=0)
explainer = XGBExplainer(verbose=1)
explainer.explain_model(model,
                        X,
                        y,
                        feature_names=f,
                        target_names=["non-default", "default"],
                        max_samples=260,
                        visualize=False,
                        temporal=False)

explainer.explain_sample(X[0], y=y[0])

```

```
Explaining sample shaped 23

FEATURE              VALUE      STD        MEAN
PAY_0                2          1.794564   -0.016700
PAY_2                2          1.782348   -0.133767
PAY_5                -2         1.530046   -0.266200
PAY_6                -2         1.486041   -0.291100
LIMIT_BAL            20000      1.136720   167484.322667
AGE                  24         1.246020   35.485500
MARRIAGE             1          1.057295   1.551867
```

![img](https://i.imgur.com/rqutYck.png)

### Params

Param | Description
---|---
model | XGBoostClassifier() or XGBoostRegressor(). Required.
X | 2-D NumPy array. Data. You can use the train set for this.
y | 1-D NumPy array. Targets.
feature_names | List. Human-readable feature names.
target_names | List. Human-readable target names.
max_samples | Int. Max samples to consider for plotting violin plots.
visualize | Bool. Whether to show the plot images.
temporal | Bool. Whether the data is temporal (forecasting).

### References

- Credit card default data: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
- Pivotal blog: http://engineering.pivotal.io/post/interpreting-decision-trees-and-random-forests/
- ELI5: https://github.com/TeamHG-Memex/eli5
- Decision Paths: http://blog.datadive.net/interpreting-random-forests/

## A Model Explanation System

Todo

http://www.inference.vc/accuracy-vs-explainability-in-machine-learning-models-nips-workshop-poster-review/

http://www.blackboxworkshop.org/pdf/Turner2015_MES.pdf A Model Explanation System / Ryan Turner

## Intelligible Models

Todo

https://vimeo.com/125940125 Intelligible Machine Learning Models for Health Care / Richard Caruana

http://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf Accurate Intelligible Models with Pairwise Interactions / Lou et al.

http://www.cs.cornell.edu/~yinlou/projects/gam/ Intelligible Models
