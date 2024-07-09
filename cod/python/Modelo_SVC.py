# -*- coding: utf-8 -*-
"""
Modulo Modelo_SVC

"""

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
from Preproceso import Preproceso  

class ModeloSVC:
    def __init__(self, data, best_params=None):
        self.data = data
        self.X = self.data.drop(columns=['isFraud'])
        self.Y = self.data['isFraud']
        self.pipeline = None
        self.best_params = best_params
        self.preproceso_instancia = Preproceso()

    def crear_pipeline(self, params=None):
        preproceso = self.preproceso_instancia.preprocesar_datos()
        if params is None:
            classifier = SVC(probability=True)
        else:
            classifier = SVC(probability=True, **params)
        self.pipeline = Pipeline([
            ('preproceso', preproceso),
            ('classifier', classifier)
        ])
        return self.pipeline

    def optimizar_hiperparametros(self):
        self.crear_pipeline()
        param_grid = {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'classifier__gamma': ['scale', 'auto']
        }
        grid_search = GridSearchCV(self.pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X, self.Y)
        self.best_params = grid_search.best_params_
        print("Mejores hiperparámetros:", self.best_params)
        return self.best_params

    def entrenar_modelo(self):
        if self.best_params is None:
            self.optimizar_hiperparametros()
        self.crear_pipeline(self.best_params)
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.25, 
                                                            random_state=7, stratify=self.Y)
        self.pipeline.fit(X_train, Y_train)
        Y_predict_proba = self.pipeline.predict_proba(X_test)[:, 1]
        roc_auc_test = roc_auc_score(Y_test, Y_predict_proba)
        print("ROC AUC del conjunto de prueba:", roc_auc_test)
        return roc_auc_test

    def validacion_cruzada(self, num_estratos=5):
        if self.best_params is None:
            self.optimizar_hiperparametros()
        self.crear_pipeline(self.best_params)
        estratos_kfold = StratifiedKFold(n_splits=num_estratos, shuffle=True, random_state=7)
        roc_auc_vector = cross_val_score(self.pipeline, self.X, self.Y, cv=estratos_kfold, 
                                         scoring='roc_auc', n_jobs=-1)
        roc_auc_promedio = np.mean(roc_auc_vector)
        print("ROC AUC promedio utilizando validación cruzada estratificada:", roc_auc_promedio)
        return roc_auc_promedio