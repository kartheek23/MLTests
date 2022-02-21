from regression_model.predict import make_prediction as alt_make_prediction
from src.predict import make_predictions
from sklearn.metrics import mean_squared_error
from src.utils.all_utils import read_yaml
config = read_yaml('config/config.yml')

def test_prediction_quality_against_benchmark(raw_training_data,sample_input_data):
    #Given
    input_df = raw_training_data.drop(config['target'],axis=1)
    output_df = raw_training_data[config['target']]
    #Setting rough benchmarks (You would tweak depending on your model)
    benchmark_flexibility = 50000
    # Setting ndigits to -4 will round to the nearest 10000 i.e. 210,000
    benchmark_lower_boundary = (
        round(output_df.iloc[0],ndigits=-4) - benchmark_flexibility
    )
    benchmark_upper_boundary = (
        round(output_df.iloc[0],ndigits=-4) + benchmark_flexibility
    )

    #When
    subject = make_predictions(input_data=input_df[0:1])

    #Then
    assert subject is not None
    prediction = subject.get("predictions")[0]
    assert isinstance(prediction,float)
    assert prediction>benchmark_lower_boundary
    assert prediction<benchmark_upper_boundary

def test_prediction_quality_against_another_model(raw_training_data,sample_input_data):
    #Given
    input_df = raw_training_data.drop(config['target'],axis=1)
    output_df = raw_training_data[config['target']]
    current_predictions = make_predictions(input_data=input_df)
    # the older model has these variable names reversed
    input_df.rename(
        columns={
            "FirstFlrSF": "1stFlrSF",
            "SecondFlrSF": "2ndFlrSF",
            "ThreeSsnPortch": "3SsnPorch",
        },
        inplace=True,
    )
    alternative_predictions = alt_make_prediction(input_data=input_df)

    #When
    current_mse = mean_squared_error(
        y_true=output_df.values,y_pred=current_predictions['predictions']
    )

    alternative_mse = mean_squared_error(
        y_true=output_df.values,y_pred=alternative_predictions['predictions']
    )

    #Then
    assert current_mse < alternative_mse
