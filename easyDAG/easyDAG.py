import warnings
import operator as op
from copy import deepcopy
import enum
from typing import NamedTuple

class PipeTypeWarning(UserWarning):
    pass

def do_eval(obj, **kwargs):
    method = '__eval__'
    if hasattr(obj, method):
        if not isinstance(obj, RawStep):
            s = "an object that is not a pipeline is getting evaluated"
            warnings.warn(PipeTypeWarning(s))
        eval_method = getattr(obj, method)
        result = eval_method(**kwargs)
        return result
    return obj

def do_eval_uncached(dag, **kwargs):
    dag = deepcopy(dag)
    return do_eval(dag, **kwargs)

def multi_eval(*objects, **kwargs):
    return tuple(do_eval(obj, **kwargs) for obj in objects)

def are_equal(obj1, obj2):
    method = '__equal__'
    obj1_has_equal = hasattr(obj1, method)
    obj2_has_equal = hasattr(obj2, method)
    if (not obj1_has_equal) and (not obj2_has_equal):
        if isinstance(obj1, Exception) and isinstance(obj2, Exception):
            return repr(obj1)==repr(obj2)
        return obj1==obj2
    
    # one of the two is a step, the other is not
    if obj1_has_equal and not obj2_has_equal:
        return False
    if obj2_has_equal and not obj1_has_equal:
        return False
    
    # implement the new class style defer of the check to the second object
    # if it is a subclass of the first one
    if issubclass(obj2.__class__, obj1.__class__):
        return getattr(obj2, method)(obj1)
    else:
        return getattr(obj1, method)(obj2)



# ███████ ████████ ███████ ██████       ██████ ██       █████  ███████ ███████
# ██         ██    ██      ██   ██     ██      ██      ██   ██ ██      ██
# ███████    ██    █████   ██████      ██      ██      ███████ ███████ ███████
#      ██    ██    ██      ██          ██      ██      ██   ██      ██      ██
# ███████    ██    ███████ ██           ██████ ███████ ██   ██ ███████ ███████

class Tokens(enum.Enum):
    NO_PREVIOUS_RESULT = enum.auto()
    FUNCTION_IDX = enum.auto()
    CACHE_IDX = enum.auto()
    
class RawStep:
    def __init__(self, function, *args, **kwargs):
        self._args = list(args)
        self._kwargs = kwargs
        self._function = function
        self._last_result = Tokens.NO_PREVIOUS_RESULT

    def __eval__(self, **kwargs):
        if self._last_result is not Tokens.NO_PREVIOUS_RESULT:
            return self._last_result
        function = do_eval(self._function, **kwargs)
        # if the function evaluates to a string (not a callable) then it represents a parameter!
        if isinstance(function, str):
            name = function
            if name in kwargs:
                #self._last_result = kwargs[name]
                #return self._last_result
                return kwargs[name]
            else:
                # if the variable is not defined, return itself
                # so that it allows to do currying of the function
                return self
        # if the result is not a string, then it should be a callable and used as a proper function
        elif callable(function):
            args_eval = [do_eval(arg, **kwargs) for arg in self._args]
            kwargs_eval = {name: do_eval(arg, **kwargs)
                            for name, arg in self._kwargs.items()}
            # try to execute the function with the params, but don't freak out
            try:
                self._last_result = function(*args_eval, **kwargs_eval)
            except Exception as e:
                # fail gracefully, if there is an exception, return that
                self._last_result = e
            return self._last_result
        # things that do not eval not strings or callable are not supported yet
        else:
            raise TypeError("the first parameter of the Step"\
                            " should be evaluate to a string"\
                            " or a callable, {} found".format(function))

    def __equal__(self, other):
        same_function = are_equal(self._function, other._function)
        if not same_function:
            return False
        same_args = all(are_equal(a1, a2) for a1, a2 in zip(self._args, other._args))
        if not same_args:
            return False
        # keys might be in different order!
        # keys are simple strings, don't need to be parsed
        self_keys = set(self._kwargs.keys())
        other_keys = set(other._kwargs.keys())
        same_keys = self_keys == other_keys
        if not same_keys:
            return False
        # values need to be parsed, and in the same order
        self_val = [self._kwargs[k] for k in self_keys]
        other_val = [other._kwargs[k] for k in self_keys]
        same_vals = all(are_equal(v1, v2) for v1, v2 in zip(self_val, other_val))
        if not same_vals:
            return False
        return True

    def __repr__(self):
        s = "{}({}{}{})"
        cls_str = self.__class__.__qualname__
        f_str = repr(self._function)
        arg_str = ''
        if len(self._args):
            arg_str = ', ' + ", ".join(repr(a) for a in self._args)
        kwarg_str = ''
        if len(self._kwargs):
            elements = (f"{k}={repr(v)}" for k, v in self._kwargs.items())
            kwarg_str = ', ' + ", ".join(elements)
        return s.format(cls_str, f_str, arg_str, kwarg_str)
    
    def __copy__(self):
        return self.__class__(self._function, *(self._args), **(self._kwargs))

    def __deepcopy__(self, memo=None):
        f = deepcopy(self._function)
        a = deepcopy(self._args)
        k = deepcopy(self._kwargs)
        result = self.__class__(f, *a, **k)
        result._last_result = deepcopy(self._last_result)
        return result
    
    def __bool__(self):
        raise NotImplementedError("DAGS don't have a defined truth value")
        
        


class Step(RawStep):
    #  ██████  ██████       ██ ███████  ██████ ████████     ██ ███    ██ ████████ ███████ ██████  ███████  █████   ██████ ███████
    # ██    ██ ██   ██      ██ ██      ██         ██        ██ ████   ██    ██    ██      ██   ██ ██      ██   ██ ██      ██
    # ██    ██ ██████       ██ █████   ██         ██        ██ ██ ██  ██    ██    █████   ██████  █████   ███████ ██      █████
    # ██    ██ ██   ██ ██   ██ ██      ██         ██        ██ ██  ██ ██    ██    ██      ██   ██ ██      ██   ██ ██      ██
    #  ██████  ██████   █████  ███████  ██████    ██        ██ ██   ████    ██    ███████ ██   ██ ██      ██   ██  ██████ ███████

    def __getattr__(self, name):
        return self.__class__(getattr, self, name)

    def __call__(self, *args, **kwargs):
        return self.__class__(self, *args, **kwargs)

    def __getitem__(self, keyname):
        return self.__class__(op.getitem, self, keyname)

    #  ██████  █████  ███████
    # ██      ██   ██ ██
    # ██      ███████ ███████
    # ██      ██   ██      ██
    #  ██████ ██   ██ ███████

    # most of these are just an indirect call to the operator module functions
    # can't shortcut the reverse call because I can't assume that in general
    # the results are simmetric (such as sum between strings)

    def __eq__(self, other):
        return self.__class__(op.eq, self, other)

    def __ne__(self, other):
        return self.__class__(op.ne, self, other)

    def __add__(self, other):
        return self.__class__(op.add, self, other)

    def __radd__(self, other):
        return self.__class__(op.add, other, self)
    
    def __sub__(self, other):
        return self.__class__(op.sub, self, other)
    
    def __rsub__(self, other):
        return self.__class__(op.sub, other, self)

    def __mul__(self, other):
        return self.__class__(op.mul, self, other)

    def __rmul__(self, other):
        return self.__class__(op.mul, other, self)

    def __pow__(self, other):
        return self.__class__(op.pow, self, other)

    def __rpow__(self, other):
        return self.__class__(op.pow, other, self)
    
    def __truediv__(self, other):
        return self.__class__(op.truediv, self, other)
    
    def __rtruediv__(self, other):
        return self.__class__(op.truediv, other, self)
    
    def __abs__(self):
        return self.__class__(abs, self)
    
    def __neg__(self):
        return self.__class__(op.neg, self)
    
    def __pos__(self):
        return self.__class__(op.pos, self)
    
    def __invert__(self):
        return self.__class__(op.invert, self)
    
    def __and__(self, other):
        return self.__class__(op.__and__, self, other)
    
    def __rand__(self, other):
        return self.__class__(op.__and__, other, self)
    
    def __or__(self, other):
        return self.__class__(op.__or__, self, other)
    
    def __ror__(self, other):
        return self.__class__(op.__or__, other, self)
    
    def __lt__(self, other):
        return self.__class__(op.lt, self, other)
    
    def __le__(self, other):
        return self.__class__(op.le, self, other)
    
    def __gt__(self, other):
        return self.__class__(op.gt, self, other)
    
    def __ge__(self, other):
        return self.__class__(op.ge, self, other)
    
    def __floordiv__(self, other):
        return self.__class__(op.floordiv, self, other)
    
    def __rfloordiv__(self, other):
        return self.__class__(op.floordiv, other, self)
    
    def __mod__(self, other):
        return self.__class__(op.mod, self, other)
    
    def __rmod__(self, other):
        return self.__class__(op.mod, other, self)
    
    def __lshift__(self, other):
        return self.__class__(op.lshift, self, other)
    
    def __rlshift__(self, other):
        return self.__class__(op.lshift, other, self)
    
    def __rshift__(self, other):
        return self.__class__(op.rshift, self, other)
    
    def __rrshift__(self, other):
        return self.__class__(op.rshift, other, self)
    
    def __xor__(self, other):
        return self.__class__(op.xor, self, other)
    
    def __rxor__(self, other):
        return self.__class__(op.xor, other, self)
    
    def __matmul__(self, other):
        return self.__class__(op.matmul, self, other)
    
    def __rmatmul__(self, other):
        return self.__class__(op.matmul, other, self)

#complex()         object.__complex__(self)
#int()             object.__int__(self)
#long()            object.__long__(self)
#float()           object.__float__(self)
#oct()             object.__oct__(self)
#hex()             object.__hex__(self)
# round            object.__round__(self)
#__floor__(self) Implements behavior for math.floor(), i.e., rounding down to the nearest integer.
#__ceil__(self) Implements behavior for math.ceil(), i.e., rounding up to the nearest integer.
#__trunc__(self) Implements behavior for math.trunc(), i.e., truncating to an integral.
        
def InputVariable(name, **kwargs):
    return Step(name, **kwargs)

def variables(*names, **kwargs):
    return [Step(name, **kwargs) for name in names]

def is_dag(dag):
    return isinstance(dag, RawStep)

def is_variable(dag):
    return isinstance(dag, RawStep) and isinstance(dag._function, str)

def is_cached(dag):
    last_result = dag._last_result
    if is_dag(last_result):
        return True
    return  last_result != Tokens.NO_PREVIOUS_RESULT
    

# ██████  ██ ██████  ███████ ██      ██ ███    ██ ███████
# ██   ██ ██ ██   ██ ██      ██      ██ ████   ██ ██
# ██████  ██ ██████  █████   ██      ██ ██ ██  ██ █████
# ██      ██ ██      ██      ██      ██ ██  ██ ██ ██
# ██      ██ ██      ███████ ███████ ██ ██   ████ ███████


# ███████ ██   ██ ████████ ██████   █████   ██████ ████████ ██  ██████  ███    ██
# ██       ██ ██     ██    ██   ██ ██   ██ ██         ██    ██ ██    ██ ████   ██
# █████     ███      ██    ██████  ███████ ██         ██    ██ ██    ██ ██ ██  ██
# ██       ██ ██     ██    ██   ██ ██   ██ ██         ██    ██ ██    ██ ██  ██ ██
# ███████ ██   ██    ██    ██   ██ ██   ██  ██████    ██    ██  ██████  ██   ████


def to_dict(expr):
    """transform the DAG in a dict of dicts"""
    if isinstance(expr, Step):
        elements = dict()
        elements[Tokens.FUNCTION_IDX] = to_dict(expr._function)
        elements[Tokens.CACHE_IDX] = to_dict(expr._last_result)
        for i, a in enumerate(expr._args):
            elements[i] = to_dict(a)
        for k, v in  expr._kwargs.items():
            elements[k] = to_dict(v)
        return elements
    else:
        return expr
    
def from_dict(processed):
    # it's not a dict, so just return the value
    if not isinstance(processed, dict):
        return processed
    # it's just a normal dict, so just return the value
    if Tokens.FUNCTION_IDX not in processed:
        return processed
    f = from_dict(processed[Tokens.FUNCTION_IDX])
    c = from_dict(processed[Tokens.CACHE_IDX])
    numerical_keys = [k for k in processed if isinstance(k, int)]
    a = [from_dict(processed[k]) for k in sorted(numerical_keys)]
    other_keys = [k for k in processed if isinstance(k, str)]
    kv = [from_dict(processed[k]) for k in other_keys]
    result = Step(f, *a, *kv)
    result._last_result = c
    return result

def unroll(step, _base=None):
    """return the adjacency list of the DAG from the starting node"""
    if isinstance(step, RawStep):
        yield (step, _base, None)
        for subdag, base, pos in unroll(step._function, step):
            pos = pos if pos is not None else Tokens.FUNCTION_IDX
            yield subdag, base, pos
        for idx, a in enumerate(step._args):
            for subdag, base, pos in unroll(a, step):
                pos = pos if pos is not None else idx
                yield subdag, base, pos 
        for k, v in step._kwargs.items():
            for subdag, base, pos in unroll(v, step):
                pos = pos if pos is not None else k
                yield subdag, base, pos 

def reset_computation(*dags):
    """reset the computed value for all the nodes in the DAG"""
    for dag in dags:
        for step, *_ in unroll(dag):
            step._last_result = Tokens.NO_PREVIOUS_RESULT
    return dags

def clear_cache_from_errors(dag, force=False):
    if not is_dag(dag):
        return dag
    if not force and not isinstance(dag._last_result, Exception):
        return dag
    dag._last_result = Tokens.NO_PREVIOUS_RESULT
    clear_cache_from_errors(dag._function, force=True)
    for arg in dag._args:
        clear_cache_from_errors(arg, force=True)
    for value in dag._kwargs.values():
        clear_cache_from_errors(value, force=True)
    return dag

def replace_in_DAG(dag, to_find, to_replace):
    if not is_dag(dag):
        return dag
    if are_equal(dag, to_find):
        return to_replace
    dag._function = replace_in_DAG(dag._function, to_find, to_replace)
    dag._args = [replace_in_DAG(a, to_find, to_replace) 
                 for a in dag._args]
    dag._kwargs = {k:replace_in_DAG(v, to_find, to_replace) 
                   for k, v in dag._kwargs.items()}
    return dag

def simplify(dag):
    for subdag1, base1, position1 in unroll(dag):
        for subdag2, base2, position2 in unroll(dag):
            if (base1 is None) or (base2 is None):
                continue
            if not are_equal(subdag1, subdag2):
                continue
            if position2 == Tokens.FUNCTION_IDX:
                base2._function = subdag1
            elif isinstance(position2, str):
                base2._kwargs[position2] = subdag1
            elif isinstance(position2, int):
                base2._args[position2] = subdag1
    return dag

class OperationCount(NamedTuple):
    n_of_nodes: int
    n_of_operations: int
    n_cached: int
    n_variables: int
    n_free_variables: int
    
def count_operations(dag):
    unrolled = [d for d, *_ in unroll(dag)]
    n_of_nodes = len(unrolled)
    unique_nodes = {id(d) for d in unrolled}
    n_of_operations = len(unique_nodes)
    cached = {id(d) for d in unrolled if is_cached(d)}
    n_cached = len(cached)
    variables = {id(d) for d in unrolled if is_variable(d)}
    n_variables = len(variables)
    free_variables = len(variables - cached)
    return OperationCount(n_of_nodes,
                          n_of_operations,
                          n_cached,
                          n_variables,
                          free_variables)

# %%

def get_free_variables(dag):
    """given the DAG, search for all the variables and return their names"""
    # results = [s for s, *t in unroll(step) if is_variable(s)]
    # reduced = []
    # for element in results:
    #     for e in reduced:
    #         if are_equal(element, e):
    #             break
    #     else:
    #         reduced.append(element)
    # return reduced
    variables = {d._function for d, *_ in unroll(dag) if is_variable(d)}
    return variables
    

def find_elements(obj, results):
    """given the adjacency list of the DAG, return the positions of OBJ"""
    for idx, el in enumerate(results):
        (element, *_) = el
        if are_equal(obj, element):
            yield idx
