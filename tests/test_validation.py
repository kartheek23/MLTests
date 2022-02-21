from src.utils.validation import validate_inputs

def test_validate_inputs(sample_input_data):
    #When
    validated_inputs, errors = validate_inputs(input_data=sample_input_data)
    #Then
    assert not errors

    assert len(sample_input_data) == 1459
    assert len(validated_inputs) == 1457

def test_validate_inputs_identifies_errors(sample_input_data):
    #Given
    test_inputs = sample_input_data.copy()
    test_inputs.at[1,"BldgType"] = 50
    #When
    validated_inputs,errors = validate_inputs(input_data=test_inputs)
    #Then
    assert errors
    assert errors[1] == {"BldgType":["Not a valid string."]}

