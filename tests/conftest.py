import pytest
from sklearn.model_selection import train_test_split
import pandas as pd
from src.utils.all_utils import load_dataset,read_yaml
config = read_yaml('config/config.yml')

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

@pytest.fixture()
def sample_input_data():
    return load_dataset(config["test_data_file"])

@pytest.fixture()
def raw_training_data():
    return load_dataset(config['training_data_file'])