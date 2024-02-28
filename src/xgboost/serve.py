import logging

from ray import serve
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Input(BaseModel):
    feature_values: List[float]

@serve.deployment(
    route_prefix = "/xgboost",
    ray_actor_options={
        "runtime_env": {
            "pip": ["xgboost==2.0.3"]
        }
    }
)
@serve.ingress(app)
class XGBoostModel:

    def __init__(self, 
                 model_path: str = "./models/xgboost_model.json",
                 log_file: str = "./logs/xgboost_model_pip.log",
                 ):
        streamhandler = logging.StreamHandler()
        streamhandler.setLevel(logging.DEBUG)

        filehandler = logging.FileHandler(log_file, mode = "w")
        filehandler.setLevel(logging.DEBUG)
        
        self.logger = logging.getLogger("sklearn")
        self.logger.setLevel(logging.DEBUG)

        self.logger.addHandler(streamhandler)
        self.logger.addHandler(filehandler)

        # try to import xgboost
        try:
            import xgboost
            self.logger.info(f"Imported xgboost=={xgboost.__version__} successfully!")
            self.logger.info(f"xgboost path: {xgboost.__file__}")
        except ImportError:
            self.logger.error("xgboost library not installed!")

        # try to import scikit-learn
        try:
            import sklearn
            self.logger.info(f"Imported sklearn=={sklearn.__version__} successfully!")
            self.logger.info(f"sklearn path: {sklearn.__file__}")
        except ImportError:
            self.logger.error("sklearn library not installed!")

        # try to import joblib
        try:
            import joblib
            self.logger.info(f"Imported joblib=={joblib.__version__} successfully!")
            self.logger.info(f"joblib path: {joblib.__file__}")
        except ImportError:
            self.logger.error("joblib library not installed!")     
        
        self.model = xgboost.Booster()
        self.model.load_model(model_path)

    @app.post("/predict")
    def predict(self, input: Input) -> int:
        import xgboost
        return self.model.predict(xgboost.DMatrix([input.feature_values])).item(0)
    
xgboost_model = XGBoostModel.bind()