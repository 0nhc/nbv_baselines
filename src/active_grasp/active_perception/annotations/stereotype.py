# --- Classes --- #

def dataset():
    pass

def module():
    pass

def pipeline():
    pass

def runner():
    pass

def factory():
    pass

# --- Functions --- #

evaluation_methods = {}
def evaluation_method(eval_type):
    def decorator(func):
        evaluation_methods[eval_type] = func
        return func
    return decorator


def loss_function():
    pass


# --- Main --- #

    