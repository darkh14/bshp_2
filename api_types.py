from pydantic import BaseModel
from enum import Enum
from datetime import datetime

from typing import Optional


class DataRow(BaseModel):
    """
    Loading, input or output data row
    """
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
    is_service: bool
    unit_of_count: str
    year: str


class ModelStatuses(Enum):
    NOTFIT = 'not_fit'
    INPROGRESS = 'in_progress'
    FIT = 'fit'
    ERROR = 'error'


class ModelInfo(BaseModel):
    status: ModelStatuses
    fitting_start_date: Optional[datetime]
    fitting_end_date: Optional[datetime]
    error_text: str