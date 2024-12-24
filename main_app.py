import fastapi
import uvicorn

from models.processor import Processor
from db_connectors.connector import MongoConnector
from api_types import DataRow, ModelInfo, ModelStatuses
from version import VERSION

app = fastapi.FastAPI(title="BSHP App",  
                      description="Application for AI cash flow parameters predictions!",
                      version=VERSION)

@app.get('/')
async def main_page():
    """
    Root method returns html ok description
    @return: HTML response with ok micro html
    """
    return fastapi.responses.HTMLResponse('<h2>BSHP module</h2> <br> <h3>Connection established</h3>')


@app.get('/health')
def health(base_name: str) -> str:
    return 'service is working'


@app.post('/fit')
def fit(data: list[DataRow], base_name: str) -> str:
    db_connector = MongoConnector(base_name)
    processor = Processor(db_connector)
    processor.fit([el.model_dump() for el in data])
    return 'model fitting is started'


@app.post('/predict')
def predict(data: list[DataRow], base_name: str) -> list[DataRow]:
    db_connector = MongoConnector(base_name)
    processor = Processor(db_connector)
    data = processor.predict([el.model_dump() for el in data])
    d = {"qty": 0,
                                    "price": 0,
                                    "sum": 0,
                                    "customer": "string",
                                    "operation_type": "string",
                                    "moving_type": "string",
                                    "base_document": "string",
                                    "agreement_name": "string",
                                    "article_cash_flow": "string",
                                    "details_cash_flow": "string",
                                    "is_service": True,
                                    "unit_of_count": "string",
                                    "year": "string"}
    
    return data


@app.get('/get_info')
def get_info(base_name: str) -> ModelInfo:
    return ModelInfo.model_validate({'base_name': base_name,
                                     'status': ModelStatuses.NOTFIT,
                                     'start_date': None,
                                     'finish_date': None})


@app.get('/drop_fitting')
def drop_fitting(base_name: str) -> str:
    return 'fitting is dropped'


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8090, log_level="info")