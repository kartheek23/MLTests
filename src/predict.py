import logging
import typing as t
import pandas as pd
from src.utils.data_management import load_pipeline
from src.utils.validation import validate_inputs
from src.utils.all_utils import read_yaml
import os
config = read_yaml('config/config.yml')
pipeline_file_name = f"{config['pipeline_save_file']}.pkl"
_price_pipe = load_pipeline(file_name=pipeline_file_name)

def make_predictions(*,input_data: t.Union[pd.DataFrame, dict],) -> dict:
    data = pd.DataFrame(input_data)
    validated_data,errors = validate_inputs(input_data=data)
    results = {"predictions": None,"errors": errors}
    if not errors:
        predictions = _price_pipe.predict(X=validated_data[config['features']])
    results = {"predictions":predictions,"errors": errors}
    return results
