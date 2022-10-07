# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 15:25:06 2022

@author: BACOS1
"""

# General libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Specific functions

from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
#from sklearnex import patch_sklearn, unpatch_sklearn

# Configure numpy random seed

np.random.seed(753)

cells = pd.read_csv('cells.csv')

features = (cells
                .drop('class', axis=1)
)

outcome = (cells['class'])

xTrain, xTest, yTrain, yTest = train_test_split(
        features,
        outcome,
        test_size = 0.25,
        stratify = outcome
)

treePreProcess = make_column_transformer(
        (FunctionTransformer(),
        (features
         .drop('case', axis=1)
         .columns)
         )
)

treePipeline = make_pipeline(
        treePreProcess,
        DecisionTreeClassifier()
)

paramGrid = {
        'decisiontreeclassifier__max_depth': [1, 4, 8, 11, 15],
        'decisiontreeclassifier__ccp_alpha': [0.0000000001, 0.0000000178, 0.00000316, 0.000562, 0.1]
}

treeScorer = {
    'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
    'accuray': make_scorer(accuracy_score)
}

treeTunerPar = GridSearchCV(
        treePipeline,
        paramGrid,
        cv=4,
        scoring=treeScorer,
        refit='roc_auc',
        n_jobs = -1
)

treeResPar = treeTunerPar.fit(xTrain, yTrain)

treeResPar.best_params_

unpatch_sklearn()