import fastapi
import uvicorn
from typing import Any

from models.processor import Processor
from db_connectors.connector import MongoConnector
from api_types import DataRow, ModelInfo, ModelStatuses
from version import VERSION

app = fastapi.FastAPI(title="BSHP App",  
                      description="Application for AI cash flow parameters predictions!",
                      version=VERSION)


def fit_background(data: list[dict[str, Any]], base_name) -> bool:
    db_connector = MongoConnector('bshp_{}'.format(base_name))
    processor = Processor(db_connector)
    processor.fit([el.model_dump() for el in data])
    return True


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
def fit(data: list[DataRow], base_name: str, background_tasks: fastapi.BackgroundTasks) -> str:
    background_tasks.add_task(fit_background, data, base_name=base_name)
    return 'model fitting is started'


@app.post('/predict')
def predict(data: list[DataRow], base_name: str) -> list[DataRow]:
    db_connector = MongoConnector('bshp_{}'.format(base_name))
    processor = Processor(db_connector)
    data = processor.predict([el.model_dump() for el in data])    
    return data


@app.get('/get_info')
def get_info(base_name: str) -> ModelInfo:
    db_connector = MongoConnector('bshp_{}'.format(base_name))
    processor = Processor(db_connector)
    result = processor.get_info()
    return ModelInfo.model_validate(result)


@app.get('/drop_fitting')
def drop_fitting(base_name: str) -> str:
    db_connector = MongoConnector('bshp_{}'.format(base_name))
    processor = Processor(db_connector)
    processor.drop_fitting()    
    return 'fitting is dropped'


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8090, log_level="info")