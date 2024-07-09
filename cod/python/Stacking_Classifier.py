# -*- coding: utf-8 -*-
"""
Modulo Stacking_Classifier

"""

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import lightgbm as lgb
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from Modelo_Logistico import ModeloLogistico
from Modelo_Random_Forest import ModeloRandomForest
from Modelo_Lightgbm import ModeloLightGBM
from Modelo_SVC import ModeloSVC
from Preproceso import Preproceso

class StackingClassifierModel:
    def __init__(self, data, best_params_xgb=None, best_params_logistic=None, best_params_rf=None, 
                 best_params_lgb=None, best_params_svc=None):
        self.data = data
        self.X = self.data.drop(columns=['isFraud'])
        self.Y = self.data['isFraud']
        self.best_params_xgb = best_params_xgb
        self.best_params_logistic = best_params_logistic
        self.best_params_rf = best_params_rf
        self.best_params_lgb = best_params_lgb
        self.best_params_svc = best_params_svc
        self.pipeline = None

    def crear_modelos_base(self):
        modelo_logistico = ModeloLogistico(self.data, self.best_params_logistic)
        modelo_rf = ModeloRandomForest(self.data, self.best_params_rf)
        modelo_lgb = ModeloLightGBM(self.data, self.best_params_lgb)
        modelo_svc = ModeloSVC(self.data, self.best_params_svc)
        
        modelo_logistico.crear_pipeline(self.best_params_logistic)
        modelo_rf.crear_pipeline(self.best_params_rf)
        modelo_lgb.crear_pipeline(self.best_params_lgb)
        modelo_svc.crear_pipeline(self.best_params_svc)
        
        return [
            ('logistic', modelo_logistico.pipeline),
            ('rf', modelo_rf.pipeline),
            ('lgb', modelo_lgb.pipeline),
            ('svc', modelo_svc.pipeline)
        ]

    def crear_stacking_classifier(self):
        base_estimators = self.crear_modelos_base()
        preproceso = Preproceso().preprocesar_datos()
        xgb_model = xgb.XGBClassifier(**(self.best_params_xgb if self.best_params_xgb else {}))
        
        self.pipeline = Pipeline([
            ('preproceso', preproceso),
            ('stacking', StackingClassifier(
                estimators=base_estimators,
                final_estimator=xgb_model,
                cv=5,
                n_jobs=-1))
        ])
        return self.pipeline

    def entrenar_stacking_classifier(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.25, 
                                                            random_state=7, stratify=self.Y)
        self.crear_stacking_classifier()
        self.pipeline.fit(X_train, Y_train)
        Y_predict_proba = self.pipeline.predict_proba(X_test)[:, 1]
        roc_auc_test = roc_auc_score(Y_test, Y_predict_proba)
        print("ROC AUC del conjunto de prueba:", roc_auc_test)
        return roc_auc_test

    def validacion_cruzada(self, num_estratos=5):
        self.crear_stacking_classifier()
        estratos_kfold = StratifiedKFold(n_splits=num_estratos, shuffle=True, random_state=7)
        roc_auc_vector = cross_val_score(self.pipeline, self.X, self.Y, cv=estratos_kfold, 
                                         scoring='roc_auc', n_jobs=-1)
        roc_auc_promedio = np.mean(roc_auc_vector)
        print("ROC AUC promedio utilizando validación cruzada estratificada:", roc_auc_promedio)
        return roc_auc_promedio