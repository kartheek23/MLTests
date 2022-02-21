import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class SklearnTransformerWrapper(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None,transformer=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables
        
        self.transformer = transformer
    def fit(self,X:pd.DataFrame,y:pd.Series=None):
        self.transformer.fit(X[self.variables])
        return self
    def transform(self,X:pd.DataFrame) -> pd.DataFrame:
        X=X.copy()
        X[self.variables] = self.transformer.transform(X[self.variables])
        return X

class DropUnecessaryFeatures(BaseEstimator,TransformerMixin):
    def __init__(self,variables_to_drop=None):
        self.variables = variables_to_drop
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X=X.copy()
        X=X.drop(self.variables,axis=1)
        return X

class TemporalVariableEstimator(BaseEstimator,TransformerMixin):
    def __init__(self,variables=None,reference_variable=None):
        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables
        self.reference_variable = reference_variable
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]
        return X