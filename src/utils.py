

# **Disclaimer:** Unlike the code above, this code is not my invention. All credit
# goes to Adam Bard for coming up with this (and Guido for writing an earlier
# implementation). Adam Bard made this code freely available at 
# https://adambard.com/blog/implementing-multimethods-in-python/.

# Alternatively, this package exists but only works based on type hints: https://pypi.org/project/multimethod/#description.
# In my opinion, the clojure dispatch function approach is far more versatile. Big thanks to Adam Bard for implementing this.

def multi(dispatch_fn):
    def _inner(*args, **kwargs):
        return _inner.__multi__.get(
            dispatch_fn(*args, **kwargs),
            _inner.__multi_default__
        )(*args, **kwargs)
    
    _inner.__dispatch_fn__ = dispatch_fn
    _inner.__multi__ = {}
    _inner.__multi_default__ = lambda *args, **kwargs: None  # Default default
    return _inner

def method(dispatch_fn, dispatch_key=None):
    def apply_decorator(fn):
        if dispatch_key is None:
            # Default case
            dispatch_fn.__multi_default__ = fn
        else:
            dispatch_fn.__multi__[dispatch_key] = fn
        return dispatch_fn
    return apply_decorator