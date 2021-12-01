from abc import ABC, abstractmethod
import os
import typing as typ
from dataclasses import dataclass

from pymongo import MongoClient

ClientConnection = typ.Union[MongoClient]


@dataclass
class Auth:
    user: str
    password: str


@dataclass
class AtlasAuth(Auth):
    pass


class Client(ABC):
    """Base class for creating a client connection to a datasource. """

    @abstractmethod
    def conn(self) -> ClientConnection:
        raise NotImplementedError()


class PerfAtlasClient(Client):
    """Client connection to perf atlas db. """

    def __init__(self):
        self.auth = AtlasAuth(user=os.getenv("PERF_DB_READ_USER"), password=os.getenv("ERF_DB_READ_PASSWORD"))

    def conn(self) -> ClientConnection:
        """Return client connection."""
        conn_str = f"mongodb+srv://{self.auth.user}:{self.auth.password}@performancedata-g6tsc.mongodb.net/expanded_metrics?readPreference=secondary&readPreferenceTags=nodeType:ANALYTICS&readConcernLevel=local"
        client = MongoClient(conn_str)

        return client
