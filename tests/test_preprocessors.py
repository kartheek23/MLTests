from src.utils.all_utils import read_yaml
from src.utils import preprocessors as pp
def test_drop_unneccessary_features_transformer(pipeline_inputs):
    #Given
    X_train,X_test,y_train,y_test = pipeline_inputs
    config = read_yaml('config/config.yml')
    drop_features = config['drop_features']
    assert drop_features in X_train.columns
    transformer = pp.DropUnecessaryFeatures(variables_to_drop=drop_features)
    #When
    X_transformed = transformer.transform(X_train)
    #Then
    assert drop_features not in X_transformed.columns