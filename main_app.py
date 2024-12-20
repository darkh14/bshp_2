import fastapi
import uvicorn

from api_types import Data, ModelInfo, ModelStatuses
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


@app.post('/fit')
def fit(data: list[Data], base_name: str) -> str:
    return 'model fitting started'


@app.post('/predict')
def predict(data: Data, base_name: str) -> Data:
    return Data.model_validate([])


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
    uvicorn.run(app, host="127.0.0.1", port=8070, log_level="info")