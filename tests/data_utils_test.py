import datetime
import hashlib
import pytest
from src.data_utils import save_name
import uuid

def test_save_name_1():
    params = {'a': 1, 'b': 2.2, 'c': 'three', 'd': True, 'e': None}
    assert save_name(params) == 'a_1__b_2p2__c_three__d_1__e_None'

def test_save_name_2():
    params = {'a': -1, 'b': -2.2, 'c': 'minus', 'd': False, 'e': None}
    assert save_name(params) == 'a_minus1__b_minus2p2__c_minus__d_0__e_None'

def test_save_name_3():
    params = {'a': 1+2j, 'b': 2-2j, 'c': 'complex', 'd': True, 'e': None}
    assert save_name(params) == 'a_(1add2j)__b_(2minus2j)__c_complex__d_1__e_None'

def test_save_name_4():
    params = {'a': datetime.datetime.now(), 'b': 2.2, 'c': 'datetime', 'd': True, 'e': None}
    assert save_name(params) == 'a_{}__b_2p2__c_datetime__d_1__e_None'.format(params['a'].isoformat())

def test_save_name_5():
    params = {'a': 1, 'b': 2.2, 'c': 'three', 'd': True, 'e': datetime.datetime.now()}
    assert save_name(params) == 'a_1__b_2p2__c_three__d_1__e_{}'.format(params['e'].isoformat())

def test_save_name_6():
    params = {'a': -1, 'b': -2.2, 'c': 'minus', 'd': False, 'e': datetime.datetime.now()}
    assert save_name(params) == 'a_minus1__b_minus2p2__c_minus__d_0__e_{}'.format(params['e'].isoformat())

def test_save_name_7():
    params = {'a': 1+2j, 'b': 2-2j, 'c': 'complex', 'd': True, 'e': datetime.datetime.now()}
    assert save_name(params) == 'a_(1add2j)__b_(2minus2j)__c_complex__d_1__e_{}'.format(params['e'].isoformat())

def test_save_name_8():
    params = {'a': datetime.datetime.now(), 'b': 2.2, 'c': 'datetime', 'd': True, 'e': 1+2j}
    assert save_name(params) == 'a_{}__b_2p2__c_datetime__d_1__e_(1add2j)'.format(params['a'].isoformat())

def test_save_name_9():
    params = {'a': 1, 'b': 2.2, 'c': 'three', 'd': True, 'e': 1+2j}
    assert save_name(params) == 'a_1__b_2p2__c_three__d_1__e_(1add2j)'

def test_save_name_10():
    params = {'a': -1, 'b': -2.2, 'c': 'minus', 'd': False, 'e': 1+2j}
    assert save_name(params) == 'a_minus1__b_minus2p2__c_minus__d_0__e_(1add2j)'

def test_save_name_empty_string_value():
    params = {'a': ''}
    assert save_name(params) == 'a_empty_string'

def test_save_name_empty_string_key():
    params = {'': 'value'}
    with pytest.raises(TypeError):
        save_name(params)

def test_save_name_unsupported_key_type():
    params = {1: 'value'}
    assert save_name(params) == '1_value'

def test_save_name_unsupported_value_type():
    params = {"unsupported": set([1, 2, 3])}  # Sets are not supported by save_name
    with pytest.raises(ValueError, match="No parameters to save"):
        save_name(params)

def test_save_name_callable_value():
    params = {'a': print}
    assert save_name(params) == 'a_print'

def test_save_name_complex_value():
    params = {'a': 1+2j}
    assert save_name(params) == 'a_(1add2j)'

def test_save_name_datetime_value():
    params = {'a': datetime.datetime.now()}
    assert save_name(params) == 'a_{}'.format(params['a'].isoformat())
    
def test_save_name_callable_value():
    params = {'a': print}
    assert save_name(params) == 'a_module.print'

def custom_function():
    return "Hello, World!"

def test_save_name_custom_function_value():
    params = {'a': custom_function}
    assert save_name(params) == 'a_custom_function'

def test_save_name_lambda_function_value():
    params = {'a': lambda x: x + 2}
    assert save_name(params) == 'a_<lambda>'

class MyClass:
    pass

def test_save_name_class_value():
    params = {'a': MyClass}
    assert save_name(params) == 'a_MyClass'

class MyClass:
    def my_method(self):
        pass

def test_save_name_class_method_value():
    params = {'a': MyClass().my_method}
    assert save_name(params) == 'a_MyClass.my_method'

def test_save_name_uuid_value():
    params = {'a': uuid.uuid4()}
    assert save_name(params) == f'a_{str(params["a"])}'

def test_save_name_excludes_sensitive_keys():
    params = {"num_simulations": 10, "temperature": 0.5, "api_key": "123456"}
    sensitive_keys = ["api_key"]
    filename = save_name(params, sensitive_keys=sensitive_keys)
    assert "api_key" not in filename
    assert "123456" not in filename

def test_save_name_with_callable():
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