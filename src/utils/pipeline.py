from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from feature_engine.categorical_encoders import RareLabelCategoricalEncoder
from src.utils import preprocessors as pp
from src.utils.all_utils import read_yaml
import pandas as pd

config = read_yaml('config/config.yml')
numeric_vars = config['numerical_vars']
categorical_vars=config['categorical_vars']
temporal_vars=config['temporal_vars']
drop_features=config['drop_features']
rare_label_tol=config['rare_label_tol']
rare_label_n_categories=config['rare_label_n_categories']
loss=config["loss"]
random_state=config["random_state"]
n_estimators=config["n_estimators"]


price_pipe = Pipeline([('numeric_imputer',
pp.SklearnTransformerWrapper(variables = numeric_vars,transformer = SimpleImputer(strategy='most_frequent')),
),
(
     "categorical_imputer",
            pp.SklearnTransformerWrapper(
                variables=categorical_vars,
                transformer = SimpleImputer(strategy="constant",fill_value='missing')
            ),
        ),
        (
            "temporal_variable",
            pp.TemporalVariableEstimator(
                variables = temporal_vars,
                reference_variable = drop_features
            ),
        ),
        (
            "rare_label_encoder",
            RareLabelCategoricalEncoder(
                tol=rare_label_tol,
                n_categories=rare_label_n_categories,
                variables=categorical_vars
            ),
        ),
        (
            "categorical_encoder",
            pp.SklearnTransformerWrapper(
                variables=categorical_vars,
                transformer=OrdinalEncoder(),
            ),
        ),
        (
            "drop_features",
            pp.DropUnecessaryFeatures(
                variables_to_drop=drop_features
            ),
        ),
        ("gb_model",GradientBoostingRegressor(loss=loss,random_state=random_state,n_estimators=n_estimators)),])

