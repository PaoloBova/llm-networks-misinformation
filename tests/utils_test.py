from src.utils import multi, method

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

# fastcore.test.test_eq(area({'type': 'square', 'width': 1, 'height': 1}), 1)
# fastcore.test.test_close(area({'type': 'circle', 'radius': 0.5}), 0.7853975)
# with fastcore.test.ExceptionExpected():
#     area({'type': 'rhombus'})