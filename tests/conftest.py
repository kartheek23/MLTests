import pytest
from sklearn.model_selection import train_test_split
import pandas as pd
from src.utils.all_utils import load_dataset,read_yaml

@pytest.fixture(scope="session")
def pipeline_inputs():
    dataframe = load_dataset("train.csv")
    config = read_yaml('config/config.yml')
    X_train,X_test,y_train,y_test = train_test_split(
        dataframe[config["features"]],
        dataframe[config["target"]],
        test_size=config["test_size"],
        random_state=config["random_state"]
    )
    return X_train,X_test,y_train,y_test