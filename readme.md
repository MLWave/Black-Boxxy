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

### References

Eligible models is from Turner's Model Explanation System. Stacking is from Wolpert's Stacked Generalization. Dark Knowledge is Hinton et al. transfer learning. Nearest predictions is an idea from Enlitic (they use layer activations for this).

## A Model Explanation System

Todo

http://www.inference.vc/accuracy-vs-explainability-in-machine-learning-models-nips-workshop-poster-review/

http://www.blackboxworkshop.org/pdf/Turner2015_MES.pdf A Model Explanation System / Ryan Turner

## Intelligible Models

Todo

https://vimeo.com/125940125 Intelligible Machine Learning Models for Health Care / Richard Caruana

http://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf Accurate Intelligible Models with Pairwise Interactions / Lou et al.

http://www.cs.cornell.edu/~yinlou/projects/gam/ Intelligible Models