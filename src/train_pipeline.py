from sklearn.model_selection import train_test_split
from src.utils import pipeline
from src.utils.data_management import save_pipeline
from src.utils.all_utils import load_dataset
from src.utils.all_utils import read_yaml
config = read_yaml('config/config.yml')

def run_training() -> None:
    data = load_dataset(filename=config['training_data_file'])

    X_train,X_test,y_train,y_test = train_test_split(data[config['features']],
        data[config['target']],
        test_size = config['test_size'],
        random_state= config['random_state'])
    pipeline.price_pipe.fit(X_train,y_train)
    save_pipeline(pipeline_to_persist=pipeline.price_pipe)

if __name__=="__main__":
    run_training()