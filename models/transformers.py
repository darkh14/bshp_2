from typing import Optional, Any
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class Imputer(SimpleImputer):

    def __init__(self, features: list[str], cat_features: list[str]):
        super().__init__(strategy='constant', fill_value=0)
        self.cat_features: list[str] = cat_features
        self.features: list[str] = features
        self._numeric_features: list[str] = [el for el in self.features if el not in self.cat_features]

    def fit(self, X: list[dict[str, Any]], y=None):
        print('Imputer ------------ FIT')
        df = pd.DataFrame(X)        
        super().fit(df[self._numeric_features])
        print('done------------------')        
        return self

    def transform(self, X: list[dict[str, Any]]) -> pd.DataFrame:
        print('Imputer ------------ TRANSFORM')
        df = pd.DataFrame(X)
        df = df.reindex(columns=self.features)
        df[self._numeric_features] = super().transform(df[self._numeric_features])

        df['is_service'] = df['is_service'].astype('str')
        print('done------------------')
        return df


class CatTransformer(TransformerMixin):
    def __init__(self, features, cat_features, targets):
        self._cat_features = cat_features
        self._features = features
        self._targets = targets
        self._columns_dict: Optional[dict[dict[str: str]]] = None

    def fit(self, X: pd.DataFrame, y=None):
        print('CatTransformer ------------ FIT')
        df = pd.DataFrame(X)
        self._columns_dict = dict()
        
        for col in self._cat_features:
            details = df[col].unique()
            numbers = [i for i in range(len(details))]
            details_dict = dict(zip(details, numbers))
            self._columns_dict[col] = details_dict
        print('done------------------')
        return self

    def transform(self, X: pd.DataFrame):
        print('CatTransformer ------------ TRANSFORM')      
        df = pd.DataFrame(X)    
        for col in self._cat_features:
            df[col] = df[col].apply(lambda x: self._columns_dict[col][x] if x != '' else -1)
            df[col] = df[col].astype('int')
        print('done------------------')
        return df
    
    def inverse_transform(self, X: pd.DataFrame):
        print('CatTransformer ------------ INVERSE TRANSFORM')

        df = pd.DataFrame(X)
        for col in self._cat_features:
            reverse_dict = {val: key for key, val in self._columns_dict[col].items()}
            df[col] = df[col].apply(lambda x: reverse_dict[x] if x != -1 else '')
            
        df['is_service'] = df['is_service'].astype('bool')
        print('done------------------')
        return df


class ModelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, target, x_columns, y_columns):
        self.target = target
        self.estimator = estimator
        self.x_columns = x_columns
        self.y_columns = y_columns
        self.is_fitted = False
        self.fitting_mode = False

    def fit(self, X: pd.DataFrame, y=None):
        print('ModelTransformer {}------------ FIT'.format(self.target))
        X_transformed = X[self.x_columns].to_numpy()
        y_transformed = X[self.y_columns].to_numpy()[:,0]

        self.estimator.fit(X_transformed, y_transformed)
        self.is_fitted = True
        self.fitting_mode = True
        print('done------------------')

        return self

    def transform(self, X):
        print('ModelTransformer  {}------------ TRANSFORM'.format(self.target))        
        result = X if self.fitting_mode else self._transform_predict(X)   
        self.fitting_mode = False
        print('done------------------')                 
        return result
    
    def predict(self, X):
        print('ModelTransformer  {}------------ PREDICT'.format(self.target))
        result = self._transform_predict(X)
        print('done------------------')
        return result
    
    def _transform_predict(self, X):
        X_transformed = X[self.x_columns].to_numpy()

        y_pred = self.estimator.predict(X_transformed)

        result = X.copy()
        result[self.y_columns[0]] = y_pred

        return result
    
    def __sklearn_is_fitted__(self):
        return self.is_fitted


