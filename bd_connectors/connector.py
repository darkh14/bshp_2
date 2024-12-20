from typing import Callable, Optional, Any
from functools import wraps

from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConfigurationError, OperationFailure

from settings import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_AUTH_SOURCE
from errors import DBConnectorException


class Connector():
    
    def __init__(self, base_name: str) -> None:
        self.base_name: str = base_name
        self.db = None
        self._connection_string = self._form_connection_string()
        self._connect()

    @staticmethod
    def _safe_db_action(method: Callable):
        """ Provides actions with DB with try-except. Raises DBConnectorException
        :param method: decorating method
        :return: decorated method
        """
        @wraps(method)
        def wrapper(self, *args, **kwargs):

            try:
                result = method(self, *args, **kwargs)
            except ServerSelectionTimeoutError as server_exception:
                raise DBConnectorException('MONGO DB Server connection error! ' + str(server_exception))
            except ConfigurationError as conf_exception:
                raise DBConnectorException('MONGO DB configuration error !' + str(conf_exception))
            except OperationFailure as conf_exception:
                raise DBConnectorException('MONGO DB operation error ! ' + str(conf_exception))

            return result

        return wrapper
    
    @_safe_db_action
    def get_line(self, collection_name: str, db_filter: Optional[dict[str, Any]] = None) -> Optional[dict]:
        """ See base method docs
        :param collection_name: required collection name
        :param db_filter: optional, db filter value to find line
        :return: dict of db line
        """
        collection = self._get_collection(collection_name)

        c_filter = db_filter if db_filter else None
        result = collection.find_one(c_filter, projection={'_id': False})

        return result or None
    
    @_safe_db_action
    def set_lines(self, collection_name: str, value: list[dict[str, Any]],
                  db_filter: Optional[dict[str, Any]] = None) -> bool:
        """ See base method docs
        :param collection_name: required collection name
        :param value: list of lines to set
        :param db_filter: optional, db filter value to find lines
        :return: result of setting line, True if successful
        """
        collection = self._get_collection(collection_name)

        c_filter = db_filter if db_filter else None
        result = True
        if c_filter:
            result = collection.delete_many(c_filter)
            result = bool(getattr(result, 'acknowledged'))

        if result:
            result = collection.insert_many(value)

        return bool(getattr(result, 'acknowledged'))

    @_safe_db_action
    def delete_lines(self, collection_name: str, db_filter: Optional[dict[str, Any]] = None) -> bool:
        """ See base method docs
        :param collection_name: required collection name
        :param db_filter: optional, db filter value to find lines
        :return: result of deleting lines, True if successful
        """
        c_filter = db_filter if db_filter else None
        if c_filter:
            collection = self._get_collection(collection_name)
            result = collection.delete_many(c_filter)
            result = bool(getattr(result, 'acknowledged'))
        else:
            result = self._db.drop_collection(collection_name)
            result = result is not None

        return result
    
    @_safe_db_action
    def get_count(self, collection_name: str, db_filter: Optional[dict[str, Any]] = None) -> int:
        """ See base method docs
        :param collection_name: required collection name
        :param db_filter: optional, db filter value to find lines
        :return: number of lines in collection
        """
        c_filter = db_filter if db_filter else {}

        collection = self._get_collection(collection_name)

        return collection.count_documents(c_filter)
    
    def drop_db(self) -> str:
        """Method to drop current database
        :return result of dropping"""

        result = super().drop_db()

        collection_names = self._get_collection_names()

        for collection_name in collection_names:
            self.delete_lines(collection_name)

        return result

    def _connect(self):
        
        try:
            client = MongoClient(self._connection_string)
            self.db = client[self.base_name]
        except ConfigurationError as conf_ex:
            raise ConfigurationError('Configuration error! ' + str(conf_ex))

    def _form_connection_string(self) -> str:
        """Forms connection string from setting vars
        for example 'mongodb://username:password@localhost:27017/?authSource=admin'
        :return: connection string
        """

        if DB_USER:
            result = 'mongodb://{user}:{password}@{host}:{port}/'.format(user=DB_USER, password=DB_PASSWORD,
                                                                         host=DB_HOST, port=DB_PORT)
            if DB_AUTH_SOURCE:
                result += '?authSource={auth_source}'.format(auth_source=DB_AUTH_SOURCE)
        else:
            result = 'mongodb://{host}:{port}/'.format(host=DB_HOST, port=DB_PORT)

        return result
    
    def _get_collection(self, collection_name):
        """ Gets collection object from db object
        :param collection_name: name of required collection,
        :return: collection object
        """
        return self._db.get_collection(collection_name)
    
    def _get_collection_names(self) -> list[str]:
        """Method to copy current db,
        :return list of collection names of current db
        """

        return self._db.list_collection_names()