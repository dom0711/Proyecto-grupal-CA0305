# -*- coding: utf-8 -*-
"""
Modulo Voting_Classifier

"""

from sklearn.ensemble import VotingClassifier
import numpy as np
from Modelo_XGB import ModeloXGBoost
from Modelo_Logistico import ModeloLogistico
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

class VotingClassifierModel:
    def __init__(self, data, best_params_xgb=None, best_params_logistic=None):
        self.data = data
        self.X = self.data.drop(columns=['isFraud'])
        self.Y = self.data['isFraud']
        self.best_params_xgb = best_params_xgb
        self.best_params_logistic = best_params_logistic
        self.pipeline = None

    def crear_modelos(self):
        modelo_xgb = ModeloXGBoost(self.data, self.best_params_xgb)
        modelo_logistico = ModeloLogistico(self.data, self.best_params_logistic)
        modelo_xgb.crear_pipeline(self.best_params_xgb)
        modelo_logistico.crear_pipeline(self.best_params_logistic)
        return modelo_xgb.pipeline, modelo_logistico.pipeline

    def crear_voting_classifier(self):
        pipeline_xgb, pipeline_logistic = self.crear_modelos()
        self.pipeline = VotingClassifier(
            estimators=[
                ('xgb', pipeline_xgb),
                ('logistic', pipeline_logistic)
            ],
            voting='soft'
        )
        return self.pipeline

    def entrenar_voting_classifier(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.25, random_state=7, stratify=self.Y)
        self.crear_voting_classifier()
        self.pipeline.fit(X_train, Y_train)
        Y_predict_proba = self.pipeline.predict_proba(X_test)[:, 1]
        roc_auc_test = roc_auc_score(Y_test, Y_predict_proba)
        print("ROC AUC del conjunto de prueba:", roc_auc_test)
        return roc_auc_test

    def validacion_cruzada(self, num_estratos=5):
        self.crear_voting_classifier()
        estratos_kfold = StratifiedKFold(n_splits=num_estratos, shuffle=True, random_state=7)
        roc_auc_vector = cross_val_score(self.pipeline, self.X, self.Y, cv=estratos_kfold, scoring='roc_auc', n_jobs=-1)
        roc_auc_promedio = np.mean(roc_auc_vector)
        print("ROC AUC promedio utilizando validación cruzada estratificada:", roc_auc_promedio)
        return roc_auc_promedio
