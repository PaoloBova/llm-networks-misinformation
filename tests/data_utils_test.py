import datetime
import pytest
from src.data_utils import save_name, extract_data
import uuid

class TestSaveName:
    def test_save_name_1(self):
        params = {'a': 1, 'b': 2.2, 'c': 'three', 'd': True, 'e': None}
        assert save_name(params) == 'a_1__b_2p2__c_three__d_1__e_None'

    def test_save_name_2(self):
        params = {'a': -1, 'b': -2.2, 'c': 'minus', 'd': False, 'e': None}
        assert save_name(params) == 'a_minus1__b_minus2p2__c_minus__d_0__e_None'

    def test_save_name_3(self):
        params = {'a': 1+2j, 'b': 2-2j, 'c': 'complex', 'd': True, 'e': None}
        assert save_name(params) == 'a_(1add2j)__b_(2minus2j)__c_complex__d_1__e_None'

    def test_save_name_4(self):
        params = {'a': datetime.datetime.now(), 'b': 2.2, 'c': 'datetime', 'd': True, 'e': None}
        assert save_name(params) == 'a_{}__b_2p2__c_datetime__d_1__e_None'.format(params['a'].isoformat())

    def test_save_name_5(self):
        params = {'a': 1, 'b': 2.2, 'c': 'three', 'd': True, 'e': datetime.datetime.now()}
        assert save_name(params) == 'a_1__b_2p2__c_three__d_1__e_{}'.format(params['e'].isoformat())

    def test_save_name_6(self):
        params = {'a': -1, 'b': -2.2, 'c': 'minus', 'd': False, 'e': datetime.datetime.now()}
        assert save_name(params) == 'a_minus1__b_minus2p2__c_minus__d_0__e_{}'.format(params['e'].isoformat())

    def test_save_name_7(self):
        params = {'a': 1+2j, 'b': 2-2j, 'c': 'complex', 'd': True, 'e': datetime.datetime.now()}
        assert save_name(params) == 'a_(1add2j)__b_(2minus2j)__c_complex__d_1__e_{}'.format(params['e'].isoformat())

    def test_save_name_8(self):
        params = {'a': datetime.datetime.now(), 'b': 2.2, 'c': 'datetime', 'd': True, 'e': 1+2j}
        assert save_name(params) == 'a_{}__b_2p2__c_datetime__d_1__e_(1add2j)'.format(params['a'].isoformat())

    def test_save_name_9(self):
        params = {'a': 1, 'b': 2.2, 'c': 'three', 'd': True, 'e': 1+2j}
        assert save_name(params) == 'a_1__b_2p2__c_three__d_1__e_(1add2j)'

    def test_save_name_10(self):
        params = {'a': -1, 'b': -2.2, 'c': 'minus', 'd': False, 'e': 1+2j}
        assert save_name(params) == 'a_minus1__b_minus2p2__c_minus__d_0__e_(1add2j)'

    def test_save_name_empty_string_value(self):
        params = {'a': ''}
        assert save_name(params) == 'a_empty_string'

    def test_save_name_empty_string_key(self):
        params = {'': 'value'}
        with pytest.raises(TypeError):
            save_name(params)

    def test_save_name_unsupported_key_type(self):
        params = {1: 'value'}
        assert save_name(params) == '1_value'

    def test_save_name_unsupported_value_type(self):
        params = {"unsupported": set([1, 2, 3])}  # Sets are not supported by save_name
        with pytest.raises(ValueError, match="No parameters to save"):
            save_name(params)

    def test_save_name_callable_value(self):
        params = {'a': print}
        assert save_name(params) == 'a_print'

    def test_save_name_complex_value(self):
        params = {'a': 1+2j}
        assert save_name(params) == 'a_(1add2j)'

    def test_save_name_datetime_value(self):
        params = {'a': datetime.datetime.now()}
        assert save_name(params) == 'a_{}'.format(params['a'].isoformat())
        
    def test_save_name_callable_value(self):
        params = {'a': print}
        assert save_name(params) == 'a_module.print'

    def test_save_name_custom_function_value(self):
        def custom_function():
            return "Hello, World!"
        params = {'a': custom_function}
        assert save_name(params) == 'a_custom_function'

    def test_save_name_lambda_function_value(self):
        params = {'a': lambda x: x + 2}
        assert save_name(params) == 'a_<lambda>'

    def test_save_name_class_value(self):
        class MyClass:
            pass
        params = {'a': MyClass}
        assert save_name(params) == 'a_MyClass'

    def test_save_name_class_method_value(self):
        class MyClass:
            def my_method(self):
                pass
        params = {'a': MyClass().my_method}
        assert save_name(params) == 'a_MyClass.my_method'

    def test_save_name_uuid_value(self):
        params = {'a': uuid.uuid4()}
        assert save_name(params) == f'a_{str(params["a"])}'

    def test_save_name_excludes_sensitive_keys(self):
        params = {"num_simulations": 10, "temperature": 0.5, "api_key": "123456"}
        sensitive_keys = ["api_key"]
        filename = save_name(params, sensitive_keys=sensitive_keys)
        assert "api_key" not in filename
        assert "123456" not in filename

    def test_save_name_with_callable(self):
        def my_function():
            pass

        params = {
            "num_simulations": 10,
            "nested": {
                "callable": my_function
            }
        }
        sensitive_keys = []
        filename = save_name(params, sensitive_keys=sensitive_keys, max_depth=2, max_length=100)
        assert filename == "nested_begin_dict_callable_my_function_end_dict__num_simulations_10"

class TestExtractData:
    def test_extract_data(self):
        # Test with valid data
        message = '{"guess": "42", "reasoning": "It is the answer to life, the universe, and everything."}'
        data_format = {"guess": str, "reasoning": str}
        expected_output = [{"guess": "42", "reasoning": "It is the answer to life, the universe, and everything."}]
        assert extract_data(message, data_format) == expected_output

        # Test with invalid data (wrong keys)
        message = '{"wrong_key": "42", "reasoning": "It is the answer to life, the universe, and everything."}'
        assert extract_data(message, data_format) == []

        # Test with invalid data (wrong types)
        message = '{"guess": 42, "reasoning": "It is the answer to life, the universe, and everything."}'
        assert extract_data(message, data_format) == []

    def test_extract_data_multiple_no_whitespace(self):
        # Test with multiple valid data but no whitespace between them
        message = '{"guess": "42", "reasoning": "It is the answer to life, the universe, and everything."}' \
                   '{"guess": "43", "reasoning": "It is one more than the answer to life, the universe, and everything."}'
        data_format = {"guess": str, "reasoning": str}
        expected_output = [{"guess": "42", "reasoning": "It is the answer to life, the universe, and everything."},
                           {"guess": "43", "reasoning": "It is one more than the answer to life, the universe, and everything."}]
        assert extract_data(message, data_format) == expected_output

    def test_extract_data_multiple_with_whitespace(self):
        # Test with multiple valid data with whitespace between them
        message = '{"guess": "42", "reasoning": "It is the answer to life, the universe, and everything."}, ' \
                   '{"guess": "43", "reasoning": "It is one more than the answer to life, the universe, and everything."}'
        data_format = {"guess": str, "reasoning": str}
        expected_output = [{"guess": "42", "reasoning": "It is the answer to life, the universe, and everything."},
                           {"guess": "43", "reasoning": "It is one more than the answer to life, the universe, and everything."}]
        assert extract_data(message, data_format) == expected_output
    
    message = '{"guess": "42", "reasoning": "It is the answer to life, the universe, and everything.", "details": {"nested": "object"}}' \
          '{"guess": "43", "reasoning": "It is one more than the answer to life, the universe, and everything.", "details": {"nested": "object"}}'

    def test_extract_data_multiple_nested_no_whitespace(self):
        # Test with multiple nested valid data with whitespace between them
        message = '{"guess": "42", "reasoning": "It is the answer to life, the universe, and everything."}, ' \
                   '{"guess": "43", "reasoning": "It is one more than the answer to life, the universe, and everything."}'
        data_format = {"guess": str, "reasoning": str}
        expected_output = [{"guess": "42", "reasoning": "It is the answer to life, the universe, and everything."},
                           {"guess": "43", "reasoning": "It is one more than the answer to life, the universe, and everything."}]
        assert extract_data(message, data_format) == expected_output

    def test_extract_data_empty(self):
        # Test with an empty message
        message = ''
        data_format = {"guess": str, "reasoning": str}
        assert extract_data(message, data_format) == []

        # Test with a message that contains no JSON strings
        message = 'This is a message with no JSON strings.'
        assert extract_data(message, data_format) == []

    def test_extract_data_invalid_json_corrigible(self):
        # Test with a message that contains an invalid JSON string that does not prevent extraction
        message = '{"guess": "42", "reasoning": "It is the answer to life, the universe, and everything."{}'
        data_format = {"guess": str, "reasoning": str}
        assert extract_data(message, data_format) == []

        message = '{"guess": "42", "reasoning": "It is the answer to life, the universe, and everything."}}'
        data_format = {"guess": str, "reasoning": str}
        expected_output = [{"guess": "42", "reasoning": "It is the answer to life, the universe, and everything."}]
        assert extract_data(message, data_format) == expected_output

    def test_extract_data_missing_keys(self):
        # Test with a message that contains a JSON string missing keys
        message = '{"guess": "42"}'
        data_format = {"guess": str, "reasoning": str}
        assert extract_data(message, data_format) == []

    def test_extract_data_extra_keys(self):
        # Test with a message that contains a JSON string with extra keys
        message = '{"guess": "42", "reasoning": "It is the answer to life, the universe, and everything.", "extra_key": "extra_value"}'
        data_format = {"guess": str, "reasoning": str}
        assert extract_data(message, data_format) == []