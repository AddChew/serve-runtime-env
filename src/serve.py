from ray import serve
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Input(BaseModel):
    feature_values: List[float]

@serve.deployment(
    ray_actor_options={
        "runtime_env": {
            "pip": ["joblib==1.3.2", "scikit-learn==1.4.1.post1"]
        }
    }
)
@serve.ingress(app)
class SklearnModel:

    def __init__(self, model_path: str = "./models/sklearn_model.joblib"):
        import joblib
        self.model = joblib.load(model_path)

    @app.post("/predict")
    def predict(self, input: Input) -> int:
        return self.model.predict([input.feature_values]).item(0)
    
sklearn_model = SklearnModel.bind()