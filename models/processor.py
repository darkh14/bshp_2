from sklearn.pipeline import Pipeline
from typing import Optional, Any
import pickle
import pandas as pd

from models.transformers import Imputer, CatTransformer, ModelEstimator
from api_types import ModelStatuses
from db_connectors.connector import BaseConnector


class Processor:

    def __init__(self, db_connector: BaseConnector):
        self.targets = ['article_cash_flow', 'details_cash_flow', 'is_service', 'unit_of_count', 'year']
        self.named_steps: list = list()
        self.targets_settings: Optional[dict] = None
        self.db_connector: BaseConnector = db_connector
        self.status = ModelStatuses.NOTFIT
        self._read_info_from_db()

        self._set_tagret_settings()

        self.features: list = ['qty', 
                    'price', 
                    'sum', 
                    'customer', 
                    'operation_type', 
                    'moving_type', 
                    'base_document', 
                    'agreement_name',
                    'article_cash_flow', 
                    'details_cash_flow', 
                    'is_service', 
                    'unit_of_count', 
                    'year']
        
        self.cat_features: list = ['article_cash_flow', 'details_cash_flow', 'year', 'customer', 'operation_type', 'moving_type', 
                             'base_document', 'agreement_name', 'unit_of_count', 'is_service']
        
        self._pipeline: Optional[Pipeline] = None

    def _set_tagret_settings(self):

        self.targets_settings = {'article_cash_flow': {'x_columns': ['qty', 
                                                                     'price', 
                                                                     'sum', 
                                                                     'customer', 
                                                                     'operation_type', 
                                                                     'moving_type', 
                                                                     'base_document', 
                                                                     'agreement_name'], 
                                                       'y_columns': ['article_cash_flow']}}
        self.targets = ['article_cash_flow'] # , 'details_cash_flow', 'year', 'unit_of_count', 'is_service'] # list(self.targets_settings.keys())

    def _make_pipeleine(self, new=False):
        
        self.named_steps = self._get_named_steps_template()

        if new:
            for named_step in self.named_steps:
                named_step['step'] = (named_step['name'], self._get_new_estimator(named_step['name']))
        else:
            db_steps = self._get_steps_from_db()
            for named_step in self.named_steps:
                db_step = [el for el in db_steps if el['name'] == named_step['name']][0]
                transformer = self._get_object_from_binary(db_step['transformer'])

                named_step['step'] = (named_step['name'], transformer)            

        self._pipeline = Pipeline([el['step'] for el in self.named_steps])

    def _get_named_steps_template(self):
        named_steps = []
        named_steps.append({'name': 'inputer', 'target': '', 'step': None})
        named_steps.append({'name': 'cat_transformer', 'target': '', 'step': None})
        for target in self.targets:
            named_steps.append({'name': '{}_model'.format(target), 'target': target, 'step': None})
        return named_steps
    
    def _get_new_estimator(self, name):
        c_named_step = [el for el in self._get_named_steps_template() if el['name'] == name][0]
        if name == 'inputer':
            return Imputer(features=self.features, cat_features=self.cat_features)
        elif name == 'cat_transformer':
            return CatTransformer(features=self.features, cat_features=self.cat_features, targets=self.targets)
        elif c_named_step['target']:
            return ModelEstimator(self.targets_settings[self.targets[-1]]['x_columns'], 
                                  self.targets_settings[self.targets[-1]]['y_columns'])
        else:
            return None

    def fit(self, data) -> bool:
        if not self._pipeline:
            new = self.status = ModelStatuses.NOTFIT
            self._make_pipeleine(new)

        self._pipeline.fit(data)

        self._write_steps_to_db()
        self.status = ModelStatuses.FIT
        self._write_info_to_db()
        return self

    def predict(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:

        if not self._pipeline:
            self._make_pipeleine()

        result = self._pipeline.predict(data)
        cat_transformer = self._pipeline.named_steps['cat_transformer']
        result = cat_transformer.reverse_transform(result)

        return result.to_dict(orient='records')

    
    def _write_steps_to_db(self):
        self.db_connector.delete_lines('model')
        self.db_connector.delete_lines('model_data')
        data = []
        for named_step in self.named_steps:
            data_row = {}
            for key, value in named_step.items():
                if key == 'step':

                    b_data = self._get_binary_from_object(value[1])

                    data_len = len(b_data)
                    max_len = 15*1024**2
                    
                    c_ind = 0
                    chunk_ind = 0
                    while c_ind < data_len:
                        b_data_chunk = b_data[c_ind: min(c_ind+max_len, data_len)]
                        self.db_connector.set_line('model_data', {'name': named_step['name'], 'ind': chunk_ind, 'data': b_data_chunk}, 
                                                   {'name': named_step['name'], 'ind': chunk_ind})
                        c_ind += max_len
                        chunk_ind+=1
                else:
                    data_row[key] = value

            data.append(data_row)
        self.db_connector.set_lines('model', data)

    def _get_steps_from_db(self):
        db_steps = self.db_connector.get_lines('model')
        for db_step in db_steps:
            model_lines = self.db_connector.get_lines('model_data', {'name': db_step['name']})
            model_lines.sort(key=lambda x: x['ind'])

            data_list = [el['data'] for el in model_lines]
            data = b''.join(data_list)

            db_step['transformer'] = data
        return db_steps

    def _read_info_from_db(self):
        line = self.db_connector.get_line('info')
        if line:
            self.status = ModelStatuses(line['status'])

    def _write_info_to_db(self):
        self.db_connector.delete_lines('info')
        line = {'status': self.status.value}
        self.db_connector.set_line('info', line)

    def _get_binary_from_object(self, data: object):
        return pickle.dumps(data)
    
    def _get_object_from_binary(self, data: bytes):
        return pickle.loads(data)


