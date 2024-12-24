from typing import Optional, Any

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import train_test_split 
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error

from sklearn.base import BaseEstimator, TransformerMixin

# def encode_objects_fit(data: list[dict]):
#     df = pd.DataFrame(data)
#     df.drop(['base_uid', 'document'], axis=1, inplace=True)
#     neworder = ['qty', 'price', 'sum', 'customer', 'operation_type', 'moving_type', 'base_document', 'agreement_name',
#                 'article_cash_flow', 'details_cash_flow', 'with_without_count', 'unit_of_count', 'year']
#     df = df.reindex(columns=neworder)
#     target_list = ['article_cash_flow', 'details_cash_flow', 'year', 'unit_of_count', 'with_without_count']
#     target_dict = {}
#     df = df.fillna(0)
#     df.with_without_count = df.with_without_count.astype('str')
#     list_cols = ['article_cash_flow', 'details_cash_flow', 'year', 'customer', 'operation_type', 'moving_type',
#                  'base_document', 'agreement_name', 'unit_of_count', 'with_without_count']
#     for col in list_cols:
#         details = df[col].unique()
#         numbers = [i for i in range(len(details))]
#         details_dict = dict(zip(details, numbers))
#         if col in target_list:
#             target_dict[col] = details_dict
#         df[col] = df[col].apply(lambda x: details_dict[x])
#         df[col] = df[col].astype('int')

#     return df, target_dict


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
        return self

    def transform(self, X: list[dict[str, Any]]) -> pd.DataFrame:
        print('Imputer ------------ TRANSFORM')
        df = pd.DataFrame(X)
        df = df.reindex(columns=self.features)
        df[self._numeric_features] = super().transform(df[self._numeric_features])

        df['is_service'] = df['is_service'].astype('str')

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

        return self

    def transform(self, X: pd.DataFrame):
        print('CatTransformer ------------ TRANSFORM')
        df = pd.DataFrame(X)
        for col in self._cat_features:
            df[col] = df[col].apply(lambda x: self._columns_dict[col][x] if x != '' else -1)
            df[col] = df[col].astype('int')

        return df
    
    def reverse_transform(self, X: pd.DataFrame):
        print('CatTransformer ------------ REVERSE TRANSFORM')

        df = pd.DataFrame(X)
        for col in self._cat_features:
            reverse_dict = {val: key for key, val in self._columns_dict[col].items()}
            df[col] = df[col].apply(lambda x: reverse_dict[x] if x != -1 else '')
            
        df['is_service'] = df['is_service'].astype('bool')
        print(df)
        return df


class ModelEstimator(RandomForestClassifier):
    def __init__(self, x_columns, y_columns):
        super().__init__(n_estimators=200, max_depth=20, min_samples_leaf=1, min_samples_split=5)
        self.x_columns = x_columns
        self.y_columns = y_columns

    def fit(self, X: pd.DataFrame, y=None):
        print('ModelEstimator ------------ FIT')
        X_transformed = X[self.x_columns].to_numpy()
        y_transformed = X[self.y_columns].to_numpy()[:,0]

        return super().fit(X_transformed, y_transformed)

    def predict(self, X):
        print('ModelEstimator ------------ Predict')
        X_transformed = X[self.x_columns].to_numpy()

        y_pred = super().predict(X_transformed)

        result = X.copy()
        result[self.y_columns[0]] = y_pred

        return result

