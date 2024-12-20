from pydantic import BaseModel
from enum import Enum
from datetime import datetime

from typing import Optional


class DataRow(BaseModel):
    """
    Loading, input or output data row
    """
    document: str
    qty: float
    price: float
    sum: float
    customer: str
    operation_type: str
    moving_type: str
    base_document: str
    agreement_name: str
    article_cash_flow: str
    details_cash_flow: str
    with_without_count: str
    unit_of_count: str
    year: int


class Data(BaseModel):
    row: list[DataRow]


class ModelStatuses(Enum):
    NOTFIT = 'not_fit'
    INPROGRESS = 'in_progress'
    FIT = 'fit'
    ERROR = 'error'


class ModelInfo(BaseModel):
    base_name: str
    status: ModelStatuses
    start_date: Optional[datetime]
    finish_date: Optional[datetime]