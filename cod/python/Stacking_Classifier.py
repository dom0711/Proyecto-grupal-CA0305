# -*- coding: utf-8 -*-
"""
Modulo Stacking_Classifier

"""

from sklearn.ensemble import StackingClassifier
from Modelo_XGB import ModeloXGBoost
from Modelo_Logistico import ModeloLogistico
from Modelo_Random_Forest import ModeloRandomForest
from Modelo_Lightgbm import ModeloLightGBM
from Modelo_SVC import ModeloSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np

class StackingClassifierModel:
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
        return "StackingClassifierModel con base estimators y meta-modelo XGBoost"

    def crear_modelos_base(self):
        modelo_logistico = ModeloLogistico(self.data, self.preproceso)
        modelo_rf = ModeloRandomForest(self.data, self.preproceso)
        modelo_lgb = ModeloLightGBM(self.data, self.preproceso)
        modelo_svc = ModeloSVC(self.data, self.preproceso)
        
        modelo_logistico.crear_pipeline()
        modelo_rf.crear_pipeline()
        modelo_lgb.crear_pipeline()
        modelo_svc.crear_pipeline()
        
        return [
            ('logistic', modelo_logistico.pipeline),
            ('rf', modelo_rf.pipeline),
            ('lgb', modelo_lgb.pipeline),
            ('svc', modelo_svc.pipeline)
        ]

    def crear_stacking_classifier(self):
        base_estimators = self.crear_modelos_base()
        modelo_xgb = ModeloXGBoost(self.data, self.preproceso)
        modelo_xgb.crear_pipeline()
        xgb_model = modelo_xgb.pipeline.named_steps['classifier']
        self.pipeline = StackingClassifier(estimators=base_estimators, final_estimator=xgb_model, cv=10, n_jobs=4)
        return self.pipeline

    def entrenar_stacking_classifier(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.25, random_state=7, stratify=self.Y)
        self.crear_stacking_classifier()
        self.pipeline.fit(X_train, Y_train)
        Y_predict_proba = self.pipeline.predict_proba(X_test)[:, 1]
        roc_auc_test = roc_auc_score(Y_test, Y_predict_proba)
        print("ROC AUC del conjunto de prueba:", roc_auc_test)
        return roc_auc_test

    def validacion_cruzada(self, num_estratos=10):
        self.crear_stacking_classifier()
        estratos_kfold = StratifiedKFold(n_splits=num_estratos, shuffle=True, random_state=7)
        roc_auc_vector = cross_val_score(self.pipeline, self.X, self.Y, cv=estratos_kfold, scoring='roc_auc', n_jobs=4)
        roc_auc_promedio = np.mean(roc_auc_vector)
        print("ROC AUC promedio utilizando validación cruzada estratificada:", roc_auc_promedio)
        return roc_auc_promedio
