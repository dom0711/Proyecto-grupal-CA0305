# -*- coding: utf-8 -*-
"""
Modulo Voting_Classifier

"""

from sklearn.ensemble import VotingClassifier
from Modelo_XGB import ModeloXGBoost
from Modelo_Logistico import ModeloLogistico
from Modelo_Random_Forest import ModeloRandomForest
from Modelo_Lightgbm import ModeloLightGBM
from Modelo_SVC import ModeloSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np

class VotingClassifierModel:
    def __init__(self, data, preproceso):
        self.data = data
        self.preproceso = preproceso
        self._X = self.data.drop(columns=['isFraud'])
        self._Y = self.data['isFraud']
        self.pipeline = None

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, new_data):
        self._X = new_data

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, new_data):
        self._Y = new_data
        
    def __str__(self):
        return "VotingClassifierModel con múltiples estimators"

    def crear_modelos_base(self):
        modelo_xgb = ModeloXGBoost(self.data, self.preproceso)
        modelo_logistico = ModeloLogistico(self.data, self.preproceso)
        modelo_rf = ModeloRandomForest(self.data, self.preproceso)
        modelo_lgb = ModeloLightGBM(self.data, self.preproceso)
        modelo_svc = ModeloSVC(self.data, self.preproceso)
        
        modelo_xgb.crear_pipeline()
        modelo_logistico.crear_pipeline()
        modelo_rf.crear_pipeline()
        modelo_lgb.crear_pipeline()
        modelo_svc.crear_pipeline()
        
        return [
            ('xgb', modelo_xgb.pipeline),
            ('logistic', modelo_logistico.pipeline),
            ('rf', modelo_rf.pipeline),
            ('lgb', modelo_lgb.pipeline),
            ('svc', modelo_svc.pipeline)
        ]

    def crear_voting_classifier(self):
        base_estimators = self.crear_modelos_base()
        self.pipeline = VotingClassifier(estimators=base_estimators, voting='soft', n_jobs=4)
        return self.pipeline

    def entrenar_voting_classifier(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.25, random_state=7, stratify=self.Y)
        self.crear_voting_classifier()
        self.pipeline.fit(X_train, Y_train)
        Y_predict_proba = self.pipeline.predict_proba(X_test)[:, 1]
        roc_auc_test = roc_auc_score(Y_test, Y_predict_proba)
        print("ROC AUC del conjunto de prueba:", roc_auc_test)
        return roc_auc_test

    def validacion_cruzada(self, num_estratos=10):
        self.crear_voting_classifier()
        estratos_kfold = StratifiedKFold(n_splits=num_estratos, shuffle=True, random_state=7)
        roc_auc_vector = cross_val_score(self.pipeline, self.X, self.Y, cv=estratos_kfold, scoring='roc_auc', n_jobs=4)
        roc_auc_promedio = np.mean(roc_auc_vector)
        print("ROC AUC promedio utilizando validación cruzada estratificada:", roc_auc_promedio)
        return roc_auc_promedio

