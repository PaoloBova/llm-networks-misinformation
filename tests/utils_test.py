from src.utils import multi, method

import pytest
import numpy as np
from src.utils import is_expandable, find_expandable_items, multi, method
from src.utils import get_in, assoc_in, expand_parameters, get_keypaths
from src.utils import build_grid_from_axes

# Test for get_in function
def test_get_in():
    d = {'a': {'b': {'c': 42}}}
    assert get_in(d, ['a', 'b', 'c']) == 42
    assert get_in(d, ['a', 'b', 'd'], default='not found') == 'not found'
    assert get_in(d, ['a', 'x', 'y'], default=None) == None
    assert get_in(d, [], default='empty path') == d

# Test for assoc_in function
def test_assoc_in():
    d = {'a': {'b': {'c': 1}}}
    new_d = assoc_in(d, ['a', 'b', 'c'], 42)
    assert new_d == {'a': {'b': {'c': 42}}}
    assert d == {'a': {'b': {'c': 1}}}  # original dictionary should not be modified

    new_d = assoc_in(d, ['a', 'b', 'd'], 99)
    assert new_d == {'a': {'b': {'c': 1, 'd': 99}}}

    new_d = assoc_in(d, ['a', 'x', 'y'], 100)
    assert new_d == {'a': {'b': {'c': 1}, 'x': {'y': 100}}}

    new_d = assoc_in(d, [], 200)
    assert new_d == d  # empty keypath should return the original dictionary

# Test for is_expandable function
def test_is_expandable():
    assert is_expandable([1, 2, 2, 3]) == True
    assert is_expandable([1, 1, 1]) == True
    assert is_expandable("string") == False
    assert is_expandable({"key": "value"}) == False
    assert is_expandable([1, [2, 3]]) == False  # non-hashable elements
    assert is_expandable([]) == False  # empty list
    
    # Additional tests involving numpy arrays
    assert is_expandable(np.array([1, 2, 2, 3])) == True
    assert is_expandable(np.array([1, 1, 1])) == True
    assert is_expandable(np.array([])) == False  # empty numpy array
    assert is_expandable(np.array([[1, 2], [2, 3]])) == True  # 2D numpy array with unique elements
    assert is_expandable(np.array([[1, 1], [1, 1]])) == True  # 2D numpy array with identical elements

# Test for find_expandable_items function
def test_find_expandable_items():
    d = {'a': {'b': [1, 2, 2, 3], 'c': {'d': [1, 1, 1]}}}
    result = list(find_expandable_items(d))
    expected = [(['a', 'b'], (4,)), (['a', 'c', 'd'], (3,))]
    assert result == expected

    d = {'a': {'b': {'c': [1, 2, 3]}, 'd': [4, 5, 6]}}
    result = list(find_expandable_items(d))
    expected = [(['a', 'b', 'c'], (3,)), (['a', 'd'], (3,))]
    assert result == expected

    d = {'a': {'b': {'c': 1}, 'd': 1}}
    result = list(find_expandable_items(d))
    expected = []
    assert result == expected

class TestExpandParameters:
    def test_expand_parameters_basic(self):
        """Tests basic parameter expansion with repeat values dropped."""
        params = {'a': [1, 2, 2],
                'b': 3,
                'c': {'d': [4, 5],
                        'e': {'f': [6, 7, 7]}}}
        result = expand_parameters(params,
                                ['a', ['c', 'd'], ['c', 'e', 'f']],
                                verbose=True,
                                drop_repeats=True)
        expected = {'a': np.array([1, 1, 1, 1, 2, 2, 2, 2]),
                    'b': 3,
                    'c': {'d': np.array([4, 4, 5, 5, 4, 4, 5, 5]),
                        'e': {'f': np.array([6, 7, 6, 7, 6, 7, 6, 7])}}}
        assert np.array_equal(result['a'], expected['a'])
        assert result['b'] == expected['b']
        assert np.array_equal(result['c']['d'], expected['c']['d'])
        assert np.array_equal(result['c']['e']['f'], expected['c']['e']['f'])
    
    def test_expand_parameters_no_drop_repeats(self):
        """Tests parameter expansion without dropping repeat values"""
        params = {'a': [1, 2, 2], 'b': 3}
        with pytest.warns(UserWarning, match="Repeat values found at a. Consider using drop_repeats=True if this is unintended."):
            result = expand_parameters(params, ['a'], verbose=True, drop_repeats=False)
    
        expected = {'a': np.array([1, 2, 2]), 'b': 3}
        assert np.array_equal(result['a'], expected['a'])
        assert result['b'] == expected['b']

    def test_expand_parameters_with_scalars(self):
        """Scalars: Tests parameter expansion with scalar values."""
        params = {'a': 1, 'b': 3}
        result = expand_parameters(params,
                                   ['a', 'b'],
                                   verbose=True,
                                   drop_repeats=True)
        expected = {'a': np.array([1]), 'b': np.array([3])}
        assert result['a'] == expected['a']
        assert result['b'] == expected['b']
        assert np.array_equal(result['a'], expected['a'])
        assert np.array_equal(result['b'], expected['b'])

    def test_expand_parameters_nested(self):
        """Tests parameter expansion with nested structures."""
        params = {'a': {'b': {'c': [1, 2, 2]}}}
        result = expand_parameters(params, [['a', 'b', 'c']], verbose=True, drop_repeats=True)
        expected = {'a': {'b': {'c': np.array([1, 2])}}}
        assert np.array_equal(result['a']['b']['c'], expected['a']['b']['c'])

    def test_expand_parameters_key_error(self):
        """Tests handling of a key not found in the parameters."""
        params = {'a': [1, 2, 2], 'b': 3}
        result = expand_parameters(params, ['c'],
                                   verbose=True,
                                   drop_repeats=True)
        # Drop repeats only affects the values of expanded keys and 'a' was not
        # expanded in this case
        expected = {'a': np.array([1, 2, 2]), 'b': 3}
        assert np.array_equal(result['a'], expected['a'])
        assert result['b'] == expected['b']
        assert 'c' not in result

    def test_expand_parameters_value_error(self):
        """Tests handling of a value that cannot be converted to a numpy array."""
        params = {'a': [1, [2, 3]], 'b': 3}
        result = expand_parameters(params, ['a'], verbose=True, drop_repeats=True)
        expected = {'a': [1, [2, 3]], 'b': 3}
        assert result['a'] == expected['a']
        assert result['b'] == expected['b']
        
    def test_expand_parameters_empty(self):
            """Tests handling of an empty parameters dictionary."""
            params = {}
            result = expand_parameters(params, [], verbose=True, drop_repeats=True)
            expected = {}
            assert result == expected

    def test_expand_parameters_non_expandable(self):
        """Tests handling of non-expandable parameters."""
        params = {'a': 1, 'b': 3}
        result = expand_parameters(params, ['a'], verbose=True, drop_repeats=True)
        expected = {'a': 1, 'b': 3}
        assert result == expected

    def test_expand_parameters_combinations(self):
        """Tests generation of all combinations of expanded parameters."""
        params = {'a': [1, 2], 'b': [3, 4]}
        result = expand_parameters(params, ['a', 'b'], verbose=True, drop_repeats=True)
        expected_a = np.array([1, 1, 2, 2])
        expected_b = np.array([3, 4, 3, 4])
        assert np.array_equal(result['a'], expected_a)
        assert np.array_equal(result['b'], expected_b)

    def test_expand_parameters_size_warning(self):
        """Tests that a MemoryError is raised when the size exceeds the threshold"""
        params = {'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}
        expand_keys = ['a', 'b', 'c']
        
        with pytest.raises(MemoryError, match="The size of the expanded arrays"):
            expand_parameters(params, expand_keys, verbose=True, drop_repeats=True, size_limit=4)

class TestGetKeypaths:
    
    def test_get_keypaths_basic(self):
        """Test basic functionality with a simple nested dictionary."""
        d = {
            'a': 1,
            'b': {
                'c': 2,
                'd': 3
            },
            'e': {
                'f': {
                    'g': 4
                }
            }
        }
        expected = [['a'], ['b', 'c'], ['b', 'd'], ['e', 'f', 'g']]
        result = get_keypaths(d)
        assert result == expected

    def test_get_keypaths_empty(self):
        """Test with an empty dictionary."""
        d = {}
        expected = []
        result = get_keypaths(d)
        assert result == expected

    def test_get_keypaths_single_level(self):
        """Test with a single-level dictionary."""
        d = {
            'a': 1,
            'b': 2,
            'c': 3
        }
        expected = [['a'], ['b'], ['c']]
        result = get_keypaths(d)
        assert result == expected

    def test_get_keypaths_mixed_types(self):
        """Test with a dictionary containing mixed types."""
        d = {
            'a': 1,
            'b': {
                'c': [1, 2, 3],
                'd': 'string'
            },
            'e': {
                'f': {
                    'g': 4.5
                }
            }
        }
        expected = [['a'], ['b', 'c'], ['b', 'd'], ['e', 'f', 'g']]
        result = get_keypaths(d)
        assert result == expected

    def test_get_keypaths_with_lists(self):
        """Test with a dictionary containing lists as values."""
        d = {
            'a': [1, 2, 3],
            'b': {
                'c': [4, 5],
                'd': 6
            },
            'e': {
                'f': {
                    'g': [7, 8, 9]
                }
            }
        }
        expected = [['a'], ['b', 'c'], ['b', 'd'], ['e', 'f', 'g']]
        result = get_keypaths(d)
        assert result == expected

class TestBuildParameterGrid:
    
    def test_build_grid_from_axes_with_range(self):
        expected = np.array([[0, 0],
                             [0, 1],
                             [0, 2],
                             [1, 0],
                             [1, 1],
                             [1, 2]])
        result = build_grid_from_axes([range(2), range(3)])
        assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

    def test_build_grid_from_axes_with_arange(self):
        expected = np.array([[0, 0],
                             [0, 1],
                             [0, 2],
                             [1, 0],
                             [1, 1],
                             [1, 2]])
        result = build_grid_from_axes([np.arange(0, 2, 1), np.arange(0, 3, 1)])
        assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

    def test_build_grid_from_axes_with_list(self):
        expected = np.array([[0.1, 5.],
                             [0.1, 4.],
                             [0.7, 5.],
                             [0.7, 4.],
                             [0.8, 5.],
                             [0.8, 4.]])
        result = build_grid_from_axes([[0.1, 0.7, 0.8], [5, 4]])
        assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

    def test_build_grid_from_axes_with_numpy_array(self):
        expected = np.array([[0.1, 5.],
                             [0.1, 4.],
                             [0.7, 5.],
                             [0.7, 4.],
                             [0.8, 5.],
                             [0.8, 4.]])
        result = build_grid_from_axes([np.array([0.1, 0.7, 0.8]), np.array([5, 4])])
        assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

    def test_build_grid_from_axes_with_large_range(self):
        with pytest.raises(ValueError):
            build_grid_from_axes([range(500000), range(200000)], override=False)

# Multi-method tests for area function
@multi
def area(shape):
    return shape.get('type')

@method(area, 'square')
def area(square):
    return square['width'] * square['height']

@method(area, 'circle')
def area(circle):
    return circle['radius'] ** 2 * 3.14159

@method(area)
def area(unknown_shape):
    raise Exception("Can't calculate the area of this shape")

def test_area_square():
    assert area({'type': 'square', 'width': 1, 'height': 1}) == 1
    assert area({'type': 'square', 'width': 2, 'height': 3}) == 6

def test_area_circle():
    assert pytest.approx(area({'type': 'circle', 'radius': 0.5})) == 0.7853975
    assert pytest.approx(area({'type': 'circle', 'radius': 1})) == 3.14159

def test_area_unknown():
    with pytest.raises(Exception, match="Can't calculate the area of this shape"):
        area({'type': 'rhombus'})
