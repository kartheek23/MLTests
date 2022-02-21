from sklearn.pipeline import Pipeline
from src.utils.all_utils import read_yaml
import typing as t
import os
from pathlib import Path
import joblib

config = read_yaml('config/config.yml')

def load_pipeline(*,file_name: str) -> Pipeline:
    """ Load a persisted pipeline """
    file_path = os.path.join(config["TRAINED_MODEL_DIR"] , file_name)
    trained_model = joblib.load(filename=file_path)
    return trained_model

def save_pipeline(*,pipeline_to_persist: Pipeline) -> None:
    save_file_name = f"{config['pipeline_save_file']}.pkl"
    save_path = os.path.join(config['TRAINED_MODEL_DIR'],save_file_name)
    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist,save_path)

def remove_old_pipelines(*,files_to_keep: t.List[str]) -> None:
    do_not_delete = files_to_keep + ["__init__.py"]
    trained_model_dir = Path(config['TRAINED_MODEL_DIR'])
    for model_file in trained_model_dir.iterdir():
        if model_file not in do_not_delete:
            model_file.unlink()
