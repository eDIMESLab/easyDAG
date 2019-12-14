# python -m pytest --cov=easyDAG
from easyDAG import Step
from easyDAG import InputVariable, Singleton
from easyDAG import do_eval, are_equal, do_eval_uncached
from easyDAG import unroll, reset_computation
from easyDAG import find_elements, get_free_variables, process
from easyDAG.easyDAG import _NO_PREVIOUS_RESULT

# %%
from functools import partial, reduce
import operator as op
from copy import deepcopy, copy
# %%
import pytest

# %%

Sum = partial(Step, lambda *args: sum(args))
Mul = partial(Step, lambda *args: reduce(op.mul, args, 1))
Pow = partial(Step, op.pow)

Tuple = partial(Step, lambda *args: tuple(args))
List  = partial(Step, lambda *args: list(args))
Set  = partial(Step, lambda *args: set(args))
DictKV  = partial(Step, lambda a, b: dict(zip(a, b)))
Dict  = partial(Step, dict)

Map = partial(Step, map)
Filter = partial(Step, filter)
Reduce = partial(Step, reduce)

# %%
# ████████ ███████ ███████ ████████      ██████  ██████  ██████  ███████
#    ██    ██      ██         ██        ██      ██    ██ ██   ██ ██
#    ██    █████   ███████    ██        ██      ██    ██ ██   ██ █████
#    ██    ██           ██    ██        ██      ██    ██ ██   ██ ██
#    ██    ███████ ███████    ██         ██████  ██████  ██████  ███████
 
# %%

@pytest.fixture
def a():
    return InputVariable('a')

@pytest.fixture
def b():
    return InputVariable('b')

@pytest.fixture
def c():
    return InputVariable('c')

@pytest.fixture
def d():
    return InputVariable('d')

# %%
def test_eval_curry(a, b):
    e1 = do_eval(a+b, a=1)
    assert do_eval(e1, b=4) == 5

def test_curry_different_expressions(a, b, c, d):
    r = a*d + b*c
    r2 = do_eval(r, a=1, c=3, d=4)
    r3 = 4+b*3
    assert are_equal(r2, r3)

def test_subclass_deferred_execution():
    class TestStep(Step):
        def __radd__(self, other):
            return super().__add__(other*2)
        
    a = Step('a')
    b = TestStep('b')
    c = a + b
    assert do_eval(c, a=1, b=2) == 4 # a*2 + b, driven by b

def test_basic_equality():
    a = InputVariable('a')
    b = InputVariable('b')
    c = InputVariable('a')

    assert not are_equal(a, b)
    assert are_equal(a, c)
    assert are_equal(a+b, c+b)
    assert are_equal(2*(a+b), 2*(c+b))

def test_equality_for_singletons():
    a = InputVariable('a')
    b = InputVariable('b')
    c = Singleton('a')    
    assert not are_equal(a, b)
    assert not are_equal(a, c)
    assert are_equal(c+b, c+b)
    assert not are_equal(a+c, c+a)

def test_variable_identity_independence():
    a = InputVariable('a')
    b = InputVariable('a')
    assert do_eval(a*b, a=5, b=3) == 25

def test_programmatically_create_model():
    def make_pipeline(name_a='a', name_b='b'):
        a = InputVariable(name_a)
        b = InputVariable(name_b)
        return a+b

    R = make_pipeline('c', 'd')
    assert do_eval(R, c=1, d=2) == 3

def test_data_structures_list(a, b, c):
    L = List(a, b, c)
    assert do_eval(L, a=1, b=2, c=3) == [1, 2, 3]
    
def test_data_structures_dict(a, b):
    D = DictKV(a, b)
    assert do_eval(D, a=['a', 'b'], b=[1, 2]) == {'a':1, 'b':2}

def test_static_map_filter_reduce(a):
    def add1(v):
        return v+1
    def is_gt_3(v):
        return v>3
    mapping = Step(map, add1, a)
    filtering = Step(filter, is_gt_3, mapping)
    s2 = Step(list, filtering)
    assert do_eval(s2, a=[1, 2, 3, 4, 5]) == [4, 5, 6]

def test_dynamic_map_filter_reduce(a):
    def add1(v):
        return v+1
    def is_gt_3(v):
        return v>3
    mapping = Step(map, add1, a)
    filtering = Step(filter, is_gt_3, mapping)
    s2 = Step(list, filtering)
    assert do_eval(s2, a=range(6)) == [4, 5, 6]

def test_iteration_over_data(a):
    def square_iter(v): return [i**2 for i in v if (i**2)>1]
    s = Step(square_iter, a)
    assert do_eval(s, a=[1, 2, 3]) == [4, 9]

def test_get_variable_value(a):
    assert do_eval(a, a=1, b=4) == 1

def test_equalities(a, b):
    assert do_eval_uncached(a == b+1, a=2, b=1) == True
    assert do_eval_uncached(a != b+1, a=3, b=2) == False
    assert do_eval_uncached(a != b+1, a=2, b=2) == True

def test_warn_extraneous_classes():
    class Pippo:
        def __eval__(self, *args, **kwargs):
            return 1
    with pytest.warns(UserWarning):
        do_eval(Pippo())

def test_function_interface(a, b):
    def add1(v):
        return v+1
    res = a(b)
    assert do_eval(res, a=add1, b=3) == 4

def test_collection_interface(a, b):
    res = a[b]
    assert do_eval(res, a=[0, 1, 4, 9], b=3) == 9

def test_stateful_append(a, b):
    # this is weird, as it might create errors, don't give stateful functions!
    res = a.append(b)
    stringa = [0]
    do_eval(res, a=stringa, b=1)
    assert stringa == [0, 1]

def test_second_order_call(a, b):
    res = a.index(b)
    values = (1, 2, 3, 4, 5, 6)
    assert do_eval(res, a=values, b=2) == 1

def test_give_normal_function(a):
    def add1(v):
        return v+1
    s = Step(add1, v=a)
    assert do_eval(s, a=3) == 4

def test_CAS_simple_add(a):
    s = a + 1
    assert do_eval(s, a=3, b=2) == 4

def test_CAS_add_between_variables(a, b):
    s = a+b
    assert do_eval(s, a=3, b=2)==5

def test_SUM_partial(a, b):
    s = Sum(a, b)
    assert do_eval(s, a=1, b=2) == 3
    s = Sum(a, 3)
    assert do_eval(s, a=1, b=2) == 4

def test_CAS_sum_between_strings_asymmetric(a):
    r1 = "hello" + a
    r2 = a + "hello"
    v1 = do_eval(r1, a='world')
    v2 = do_eval(r2, a='world')
    assert v1!=v2
    assert v1 == 'helloworld'
    assert v2 == 'worldhello'

def test_multiple_results(a, b):
    s1 = Sum(a, b)
    s2 = Sum(a, 1)
    res = Tuple(s1, s2)
    assert do_eval(res, a=1, b=3) == (4, 2)

def test_normal_partial_values_in_step(a):
    Pow = partial(Step, lambda n, p: n**p)
    p = Pow(n=a, p=2)
    assert do_eval(p, a=5) == 25

def test_CAS_sums_and_prods(a, b, c, d):
    mul_1 = Mul(a, d)
    mul_2 = Mul(b, c)
    sum_1 = Sum(mul_1, mul_2)
    assert do_eval(sum_1, a=1, b=2, c=2, d=4) == 8
    r = a*d + b*c
    assert do_eval(r, a=1, b=2, c=2, d=4) == 8

def test_CAS_power(a, b):
    r1 = a**b
    r2 = b**a
    assert do_eval(r1, a=2, b=3) == 8
    assert do_eval(r2, a=2, b=3) == 9

def test_returning_function():
    a = InputVariable('base')
    b = InputVariable('power')

    def make_power_function(pow):
        def make_power(base):
            return base**pow
        return make_power

    make_pow = Step(make_power_function, a)
    do_math = Step(make_pow, b)
    assert do_eval(do_math, base=2, power=3, non_used=4) == 9

def test_unorthodox_variable_names():
    data = {'a': 1, '--b--': 2}
    a = InputVariable('a')
    b = InputVariable('--b--')
    assert do_eval(a+b, **data) == 3

def test_caching_results(a):
    num_of_executions = 0
    def echo_increase(x):
        nonlocal num_of_executions
        num_of_executions +=1
        return x+1

    s = Step(echo_increase, a)
    assert do_eval(s, a=1) == 2
    assert do_eval(s, a=2) == 2
    reset_computation(s)
    assert do_eval(s, a=2) == 3
    assert num_of_executions == 2
    
def test_resect_part_of_the_DAG(a, b, c, d):
    r = a*d + b*c
    r2 = do_eval(r, a=1, c=3, d=4)
    r3 = 4+b*3
    assert are_equal(r2, r3)
    
    def _replace_idx(obj, idx, value):
        obj = list(obj)
        obj[idx] = value
        return tuple(obj)
        
    assert do_eval(r3, b=1) == 7
    reset_computation(r3)
    r3._args[1] =  b*4
    assert do_eval(r3, b=1) == 8
    
def test_find_subtrees(a, b, c, d):
    r = (a*b)*d + c*(a*b)
    
    adj_list = list(unroll(r))
    res = list(find_elements(a*b, adj_list))
    assert len(res)==2
    
    r2 = Step((lambda x, y: x+y), x=(a*b)*d, y=c*(a*b))
    adj_list = list(unroll(r2))
    res = list(find_elements(a*b, adj_list))
    assert len(res)==2
    
def test_exception_management(a):
    res = do_eval(1/a, a=0)
    isinstance(res, ZeroDivisionError)
    res = do_eval(1/a +1 , a=0)
    isinstance(res, Exception)

def test_deepcopy_cache_no_interaction(a):
    b = 1/a +1
    c = deepcopy(b)
    res = do_eval(c , a=0)
    
    for step, parent in unroll(b):
        assert step._last_result is _NO_PREVIOUS_RESULT
        
    for step, parent in unroll(c):
        assert step._last_result is not _NO_PREVIOUS_RESULT
        
    assert c._last_result is res


def test_deferred_equality_to_subclass():
    class Useless(Step):
        def __equal__(self, other):
            return True
        
    a = InputVariable("a")
    b = Useless('b')
    
    assert are_equal(a, b) == True
    assert are_equal(b, a) == True
    

def test_error_unclear_function_type():
    a = InputVariable('a')
    b = Step(a, 1, 2)
    with pytest.raises(TypeError):
        do_eval(b, a=4)

def test_fluent_interface():
    a = InputVariable('a')
    expr = a.replace("hello", "ciao").replace("world", "mondo")
    res = do_eval(expr, a='hello world!')
    assert res == "ciao mondo!"


def test_dynamic_variable_generation_surprising():
    """this is a weird one but logically correct
    during evaluation, given that "a" is replaced by a string,
    b becomes a variable and thus the result of the evaluation
    is a new pipeline.
    once that pipeline is evluated, the result is just the function,
    as the parameters of the inputvariables are not used.
    one can also do everything in one step to male it weirder.
    """
    a = InputVariable('a')
    b= Step(a, 1, 2)
    res = do_eval(b, a="adios", adios=op.add)
    assert res(1, 2) == 3
    
    a = InputVariable('a')
    b= Step(a, 1, 2)
    partial_dag = do_eval(b, a="adios")
    res = do_eval(partial_dag, adios=op.add)
    assert res(1, 2) == 3


def test_basic_copy_shallow_and_deep():
    b = InputVariable('b')
    a = InputVariable('a', meta=b)
    a1 = copy(a)
    a2 = deepcopy(a)
    assert are_equal(a, a1)
    assert are_equal(a, a2)
    assert a._kwargs['meta'] is a1._kwargs['meta'] 
    assert a._kwargs['meta'] is not a2._kwargs['meta'] 
    assert are_equal(a._kwargs['meta'], a2._kwargs['meta'])

def test_equality_independent_key_ordering():
    a1 = InputVariable('a', meta=1, type=int)
    a2 = InputVariable('a', meta=1, type=int)
    a3 = InputVariable('a', type=int, meta=1)
    assert are_equal(a1, a2)
    assert are_equal(a1, a3)

def test_equality_need_same_keys():
    a1 = InputVariable('a', meta=1, type=int)
    a4 = InputVariable('a', meta=1, type=int, time=0)
    assert not are_equal(a1, a4)
    
def test_equality_values():
    a1 = InputVariable('a', meta=1, type=int)
    a2 = InputVariable('a', meta=2, type=float)
    assert not are_equal(a1, a2)

def test_dag_processing_into_dict():
    a = InputVariable('a')
    assert process(a) == {'$function$': 'a'}
    b = a+2
    assert process(b) == {'$function$': op.add,
                          0: {'$function$': 'a'},
                          1: 2}
    c = Step("random", a=1, b=2)
    process(c) == {'$function$': 'random', 
                   'a': 1, 
                   'b': 2}

def test_get_free_variables(a, b, c):
    expr = a*c + c*b + a*2 +5
    free_vars = get_free_variables(expr)
    assert len(free_vars) == 3


def test_repr():
    a = Step('a', 1, b=2)
    assert repr(a) == "Step('a', 1, b=2)"

def test_eval_uncached(a):
    expr = 1 + 2*a
    assert do_eval_uncached(expr, a=2) == 5
    assert do_eval_uncached(expr, a=3) == 7
    
# CAS TESTING

def test_CAS_rpow(a):
    assert do_eval(2**a, a=3) == 8
    
def test_CAS_truediv(a):
    assert do_eval(a/2, a=8) == 4
    
def test_CAS_abs(a):
    assert do_eval(abs(a), a=-2) == 2

def test_CAS_pos(a):
    assert do_eval(+a, a=-2) == +(-2)

def test_CAS_neg(a):
    assert do_eval(-a, a=2) == -2
    
def test_CAS_invert(a):
    assert do_eval(~a, a=2) == ~2
    
def test_CAS_sub(a):
    assert do_eval(a-2, a=3) == 1

def test_CAS_rsub(a):
    assert do_eval(3-a, a=3) == 0
    
def test_CAS_and(a):
    assert do_eval(a & True, a=False) == False
    
def test_CAS_or(a):
    assert do_eval(a | True, a=False) == True
    
def test_CAS_rand(a):
    assert do_eval(True & a, a=False) == False
    
def test_CAS_ror(a):
    assert do_eval(True | a, a=False) == True
    
def test_CAS_xor(a):
    assert do_eval(a ^ True, a=False) == True
  
def test_CAS_rxor(a):
    assert do_eval(True ^ a, a=False) == True
    
def test_CAS_floordiv(a):
    assert do_eval(a //2, a=5) == 2
  
def test_CAS_rfloordiv(a):
    assert do_eval(5//a, a=2) == 2
    
def test_CAS_rshift(a):
    assert do_eval(a >>2, a=5) == (5>>2)
  
def test_CAS_rrshift(a):
    assert do_eval(2>>a, a=3) == (2>>3) 
    
def test_CAS_lshift(a):
    assert do_eval(a <<2, a=5) == (5<<2)
  
def test_CAS_rlshift(a):
    assert do_eval(2<<a, a=3) == (2<<3) 
    
def test_CAS_mod(a):
    assert do_eval(a %2, a=5) == 1
  
def test_CAS_rmod(a):
    assert do_eval(5%a, a=2) == 1
    
def test_matmul():
    """given that numpy implements the matmul operation but does not
    conceive that they might not know how to handle it, it can't be the
    first object, need a trick to test the interface"""
    class Vector:
        def __init__(self, *args):
            self.args = args
            
        def __matmul__(self, other):
            if not isinstance(other, Vector):
                return NotImplemented
            return sum(i*j for i, j in zip(self.args, other.args))
        
    v1 = Vector(1, 2)
    v2 = Vector(1, 2)
    assert v1@v2 == 5
    a = InputVariable('a')
    assert do_eval(a @ v1, a=v2) == v1@v2
    a = InputVariable('a')
    assert do_eval(v1 @ a, a=v2) == v1@v2

def test_CAS_disequalities(a):
    assert do_eval_uncached(a>5, a=6)
    assert do_eval_uncached(a>=5, a=6)
    assert do_eval_uncached(a<8, a=6)
    assert do_eval_uncached(a<=8, a=6)


