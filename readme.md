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

## A Model Explanation System

Todo

http://www.inference.vc/accuracy-vs-explainability-in-machine-learning-models-nips-workshop-poster-review/

http://www.blackboxworkshop.org/pdf/Turner2015_MES.pdf A Model Explanation System / Ryan Turner

## Intelligible Models

Todo

https://vimeo.com/125940125 Intelligible Machine Learning Models for Health Care / Richard Caruana

http://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf Accurate Intelligible Models with Pairwise Interactions / Lou et al.

http://www.cs.cornell.edu/~yinlou/projects/gam/ Intelligible Models