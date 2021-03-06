# python -m pytest --cov=easyDAG
from easyDAG import Step, Template
from easyDAG import Tokens, simplify
from easyDAG import do_eval, are_equal, do_eval_uncached, clear_cache
from easyDAG import replace_in_DAG, match, is_dag
from easyDAG import find_elements, get_free_variables, to_dict, from_dict
from easyDAG import count_operations, multi_eval, variables, is_variable
# %%
from functools import partial, reduce
import operator as op
from copy import deepcopy, copy
# %%
from cmath import isnan # needed for some tests
import pytest
from hypothesis import given, example, assume
import hypothesis.strategies as st
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
    return Step('a')

@pytest.fixture
def b():
    return Step('b')

@pytest.fixture
def c():
    return Step('c')

@pytest.fixture
def d():
    return Step('d')

# %%
unicode_categories = ('Nd', # Number, decimal digit
                      'Lu', # Letter, uppercase
                      'Ll', # Letter, uppercase
                      'Pc', # Punctuation, connector, Includes "_" underscore
                      'Pd', # Punctuation, dash
                      )
legitimate_chars = st.characters(whitelist_categories=(unicode_categories))
names_strategy = st.text(alphabet=legitimate_chars,
                         min_size=1, max_size=100)
general_values_strategy = st.one_of(st.text(),
                       st.floats(),
                       )

numerical_value = st.one_of(st.floats(),
                            #st.complex_numbers(),
                            st.integers(min_value=-2**10, max_value=2**10),
                            #st.decimals(),
                            #st.fractions(),
                            )

mathematical_functions = st.one_of(st.just(op.add),
                                   st.just(op.mul),
                                   st.just(op.eq),
                                   st.just(op.ne),
                                   st.just(op.sub),
                                   st.just(op.pow),
                                   st.just(op.truediv),
                                   st.just(op.__and__),
                                   st.just(op.__or__),
                                   st.just(op.lt),
                                   st.just(op.le),
                                   st.just(op.gt),
                                   st.just(op.ge),
                                   st.just(op.floordiv),
                                   st.just(op.mod),
                                   st.just(op.lshift),
                                   st.just(op.rshift),
                                   st.just(op.xor),
                                   )

@given(name=names_strategy, value=general_values_strategy)
@example(name='a', value=1)
def test_get_variable_value(name, value):
    variable = Step(name)
    values = {name: value} 
    assert do_eval(variable, **values) is value


@given(name_a=names_strategy, 
        name_b=names_strategy,
        value_a=numerical_value,
        value_b=numerical_value,
        op_math_function=mathematical_functions,
        )
@example(name_a='a', name_b='b', value_a=1, value_b=2, op_math_function=op.add)
def test_math_interface_hyp(name_a, name_b, value_a, value_b, op_math_function):
    try:
        expected = op_math_function(value_a, value_b)
    except Exception:
        assume(False)
    try:
        is_nan_value = isnan(expected)
    except OverflowError:
        is_nan_value = False
    assume(not is_nan_value)
    assume(name_a != name_b)
    a = Step(name_a)
    b = Step(name_b)
    value_dict = {name_a: value_a, name_b: value_b}
    expr_1 = Step(op_math_function, a, b)
    expr_2 = op_math_function(a, b)
    assert isinstance(expr_1, Step)
    assert isinstance(expr_2, Step)
    assert are_equal(expr_1, expr_2)
    observed_1 = do_eval(expr_1, **value_dict)
    observed_2 = do_eval(expr_2, **value_dict)
    assert isinstance(observed_1, type(expected))
    assert isinstance(observed_2, type(expected))
    assert  observed_1 == expected
    assert  observed_2 == expected
    
    value_a_dict = {name_a: value_a}
    expr_direct = op_math_function(a, value_b)
    assert isinstance(expr_direct, Step)
    observed = do_eval(expr_direct, **value_a_dict)
    assert isinstance(observed, type(expected))
    assert  observed == expected
    
    value_b_dict = {name_b: value_b}
    expr_inverse = op_math_function(value_a, b)
    assert isinstance(expr_inverse, Step)
    observed = do_eval(expr_inverse, **value_b_dict)
    assert isinstance(observed, type(expected))
    assert  observed == expected
    

#TODO: should test multiple operations by creting a queue of values and
#operations and evaluate the whole stack

@given(name_a=names_strategy, 
        name_b=names_strategy,
        value_a=numerical_value,
        value_b=numerical_value,
        op_math_function=mathematical_functions,
        )
@example(name_a='a', name_b='b', value_a=1, value_b=2, op_math_function=op.add)
def test_eval_curry_hyp(name_a, name_b, value_a, value_b, op_math_function):
    try:
        expected = op_math_function(value_a, value_b)
    except Exception:
        assume(False)
    try:
        is_nan_value = isnan(expected)
    except OverflowError:
        is_nan_value = False
    assume(not is_nan_value)
    assume(name_a != name_b)
    a = Step(name_a)
    b = Step(name_b)
    expr = Step(op_math_function, a, b)
    a_dict = {name_a: value_a}
    b_dict = {name_b: value_b}
    curried = do_eval(expr, **a_dict)
    assert isinstance(curried, Step)
    observed = do_eval(curried, **b_dict)
    assert isinstance(observed, type(expected))
    assert  observed == expected

# %%
def test_equality_fails(a):
    e1 = a+1
    e2 = a+1
    with pytest.raises(NotImplementedError):
        is_truthy = bool(e1==e2)
        assert is_truthy
        

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
    a = Step('a')
    b = Step('b')
    c = Step('a')

    assert not are_equal(a, b)
    assert are_equal(a, c)
    assert are_equal(a+b, c+b)
    assert are_equal(2*(a+b), 2*(c+b))

def test_variable_identity_independence():
    a = Step('a')
    b = Step('a')
    assert do_eval(a*b, a=5, b=3) == 25

def test_programmatically_create_model():
    def make_pipeline(name_a='a', name_b='b'):
        a = Step(name_a)
        b = Step(name_b)
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

def test_are_equal_exception_special_case():
    e1 = ValueError('a')
    e2 = ValueError('a')
    e3 = ValueError('b')
    assert are_equal(e1, e2)
    assert not are_equal(e1, e3)

def test_iteration_over_data(a):
    def square_iter(v): return [i**2 for i in v if (i**2)>1]
    s = Step(square_iter, a)
    assert do_eval(s, a=[1, 2, 3]) == [4, 9]

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
    a = Step('base')
    b = Step('power')

    def make_power_function(pow):
        def make_power(base):
            return base**pow
        return make_power

    make_pow = Step(make_power_function, a)
    do_math = Step(make_pow, b)
    assert do_eval(do_math, base=2, power=3, non_used=4) == 9

def test_unorthodox_variable_names():
    data = {'a': 1, '--b--': 2}
    a = Step('a')
    b = Step('--b--')
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
    clear_cache(s, force=True)
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
    clear_cache(r3, force=True)
    r3._args[1] =  b*4
    assert do_eval(r3, b=1) == 8
    
def test_find_subtrees(a, b, c, d):
    r = (a*b)*d + c*(a*b)
    
    res = list(find_elements(a*b, r))
    assert len(res)==2
    
    r2 = Step((lambda x, y: x+y), x=(a*b)*d, y=c*(a*b))
    res = list(find_elements(a*b, r2))
    assert len(res)==2

def test_unroll_contains_objects_simple_operations(a, b):
    expr = a + b
    assert len(list(expr))==3
    
def test_unroll_contains_objects_generic_operations(a, b, c):
    expr = Step(a, b, c)
    assert len(list(expr))==4
    
    
def test_simple_unroll(a, b, c, d):
    expr_base = Step(c, d)
    expr = Step(expr_base, x=a, y=b)
    for subdag, basedag, position in expr:
        assert isinstance(subdag, Step)
        if basedag is None:
            assert position is None
            continue
        assert isinstance(basedag, Step)
        if position == Tokens.FUNCTION_IDX:
            assert subdag is basedag._function
        elif isinstance(position, str):
            assert subdag is basedag._kwargs[position]
        elif isinstance(position, int):
            assert subdag is basedag._args[position]
        else:
            assert False

def test_exception_management(a):
    res = do_eval(1/a, a=0)
    isinstance(res, ZeroDivisionError)
    res = do_eval(1/a +1 , a=0)
    isinstance(res, Exception)

def test_deepcopy_cache_no_interaction(a):
    def is_variable(s):
        return isinstance(s._function, str)
    
    a = Step('a')
    b = 1/a +1
    c = deepcopy(b)
    res = do_eval(c , a=0)
    d = deepcopy(c)
    
    assert is_variable(a)
    assert not is_variable(b)
    assert not is_variable(c)
    assert not is_variable(d)
    
    for step, parent, position in b:
        if not is_variable(step):
            assert step._last_result is Tokens.NO_PREVIOUS_RESULT
        
    for step, parent, position in c:
        if not is_variable(step):
            assert step._last_result is not Tokens.NO_PREVIOUS_RESULT
    assert c._last_result is res
    
    for step, parent, position in d:
        if not is_variable(step):
            assert step._last_result is not Tokens.NO_PREVIOUS_RESULT
    # can't test for equality, as there is no guarantee that the copied
    # objects will have an equality method that returns true for copied objects
    # such as for exceptions
    #assert d._last_result == res


def test_deferred_equality_to_subclass():
    class Useless(Step):
        def __equal__(self, other):
            return True
        
    a = Step("a")
    b = Useless('b')
    
    assert are_equal(a, b) == True
    assert are_equal(b, a) == True
    

def test_error_unclear_function_type():
    a = Step('a')
    b = Step(a, 1, 2)
    with pytest.raises(TypeError):
        do_eval(b, a=4)

def test_fluent_interface():
    a = Step('a')
    expr = a.replace("hello", "ciao").replace("world", "mondo")
    res = do_eval(expr, a='hello world!')
    assert res == "ciao mondo!"


def test_dynamic_variable_generation_surprising():
    """this is a weird one but logically correct
    during evaluation, given that "a" is replaced by a string,
    b becomes a variable and thus the result of the evaluation
    is a new pipeline.
    once that pipeline is evluated, the result is just the function,
    as the parameters of the Steps are not used.
    one can also do everything in one step to male it weirder.
    """
    a = Step('a')
    b= Step(a, 1, 2)
    res = do_eval(b, a="adios", adios=op.add)
    assert res(1, 2) == 3


def test_basic_copy_shallow_and_deep():
    b = Step('b')
    a = Step('a', meta=b)
    a1 = copy(a)
    a2 = deepcopy(a)
    assert are_equal(a, a1)
    assert are_equal(a, a2)
    assert a._kwargs['meta'] is a1._kwargs['meta'] 
    assert a._kwargs['meta'] is not a2._kwargs['meta'] 
    assert are_equal(a._kwargs['meta'], a2._kwargs['meta'])

def test_equality_independent_key_ordering():
    a1 = Step('a', meta=1, type=int)
    a2 = Step('a', meta=1, type=int)
    a3 = Step('a', type=int, meta=1)
    assert are_equal(a1, a2)
    assert are_equal(a1, a3)

def test_equality_need_same_keys():
    a1 = Step('a', meta=1, type=int)
    a4 = Step('a', meta=1, type=int, time=0)
    assert not are_equal(a1, a4)
    
def test_equality_values():
    a1 = Step('a', meta=1, type=int)
    a2 = Step('a', meta=2, type=float)
    assert not are_equal(a1, a2)

def test_dag_processing_into_dict():
    a = Step('a')
    assert to_dict(a) == {Tokens.FUNCTION_IDX: 'a',
                          Tokens.CACHE_IDX: Tokens.NO_PREVIOUS_RESULT}
    b = a+2
    assert to_dict(b) == {Tokens.FUNCTION_IDX: op.add,
                          Tokens.CACHE_IDX: Tokens.NO_PREVIOUS_RESULT,
                          0: {Tokens.FUNCTION_IDX: 'a',
                              Tokens.CACHE_IDX: Tokens.NO_PREVIOUS_RESULT},
                          1: 2}
    do_eval(b, a=2)
    assert to_dict(b) == {Tokens.FUNCTION_IDX: op.add,
                          Tokens.CACHE_IDX: 4,
                          0: {Tokens.FUNCTION_IDX: 'a',
                              Tokens.CACHE_IDX: Tokens.NO_PREVIOUS_RESULT},
                          1: 2}
    c = Step("random", a=1, b=2)
    to_dict(c) == {Tokens.FUNCTION_IDX: 'random', 
                   Tokens.CACHE_IDX: Tokens.NO_PREVIOUS_RESULT,
                   'a': 1, 
                   'b': 2}

def test_from_dict():
    a = Step('a')
    b = 2**a + 1
    p = to_dict(b)
    e = from_dict(p)
    assert are_equal(e, b)

def test_from_dict_with_dict_params():
    a = Step('a')
    b = Step(dict.get, {'dog': 1, 'cat': 2}, a)
    assert do_eval(b, a='cat') == 2
    p = to_dict(b)
    e = from_dict(p)
    assert are_equal(e, b)
    assert are_equal(b._last_result, e._last_result)

def test_get_free_variables(a, b, c):
    expr = a*c + c*b + a*2 +5
    free_vars = get_free_variables(expr)
    assert len(free_vars) == 3
    assert free_vars == {'a', 'b', 'c'}

def test_replace_in_dag():
    a = Step('a')
    b = Step('b')
    expr1 = 2**a + a*2 + 1
    expr2 = replace_in_DAG(deepcopy(expr1), a, b)
    res1 = do_eval(expr1, a=1, b=2)
    res2 = do_eval(expr2, a=2, b=1)
    assert are_equal(res1, res2)    


def test_repr():
    a = Step('a', 1, b=2)
    assert repr(a) == "Step('a', 1, b=2)"

def test_eval_uncached(a):
    expr = 1 + 2*a
    assert do_eval_uncached(expr, a=2) == 5
    assert do_eval_uncached(expr, a=3) == 7

def test_clear_cache_from_errors():
    a = Step('a')
    b = Step('b')
    expr = 1/b + 1/a
    do_eval(expr, a=0, b=1)
    expected_1 = {Tokens.FUNCTION_IDX: op.add,
                  Tokens.CACHE_IDX: TypeError("unsupported operand type(s) for +: 'int' and 'ZeroDivisionError'"),
                  0: {Tokens.FUNCTION_IDX: op.truediv,
                      Tokens.CACHE_IDX: 1,
                      0: 1,
                      1: {Tokens.FUNCTION_IDX: 'b',
                          Tokens.CACHE_IDX: Tokens.NO_PREVIOUS_RESULT,
                          }
                      },
                  1: {Tokens.FUNCTION_IDX: op.truediv,
                      Tokens.CACHE_IDX: ZeroDivisionError('division by zero'),
                      0: 1,
                      1: {Tokens.FUNCTION_IDX: 'a',
                          Tokens.CACHE_IDX: Tokens.NO_PREVIOUS_RESULT,
                          }
                      }
                  }
    assert are_equal(from_dict(expected_1), from_dict(expected_1))
    assert are_equal(expr, from_dict(expected_1))
    clean_expr = clear_cache(expr)
    expected_2 = {Tokens.FUNCTION_IDX: op.add,
                  Tokens.CACHE_IDX: Tokens.NO_PREVIOUS_RESULT,
                  0: {Tokens.FUNCTION_IDX: op.truediv,
                      Tokens.CACHE_IDX: 1,
                      0: 1,
                      1: {Tokens.FUNCTION_IDX: 'b',
                          Tokens.CACHE_IDX: Tokens.NO_PREVIOUS_RESULT,
                          }
                      },
                  1: {Tokens.FUNCTION_IDX: op.truediv,
                      Tokens.CACHE_IDX: Tokens.NO_PREVIOUS_RESULT,
                      0: 1,
                      1: {Tokens.FUNCTION_IDX: 'a',
                          Tokens.CACHE_IDX: Tokens.NO_PREVIOUS_RESULT,
                          }
                      }
                  }
    assert are_equal(clean_expr, from_dict(expected_2))

def test_clear_cache_not_action_without_errors(a):
    expr = a+1
    res_1 = do_eval(expr, a=2)
    clear_cache(expr)
    assert expr._last_result is res_1
    

def test_clear_cache_from_errors_with_kwargs(a, b, c):
    def partial_pow(p):
        if not isinstance(p, int):
            raise ValueError("should be int")
        def my_pow(i):
            return i**p
        return my_pow
    pow = Step(partial_pow, p=b)
    expr = Step(sorted, a, key=pow) * (c+1)
    do_eval(expr, a=[1, 3, -2], b='2', c=1)
    clear_cache(expr)
    to_dict(expr)
    assert do_eval(expr, a=[1, 3, -2], b=2, c=1) == [1, -2, 3, 1, -2, 3]


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
    a = Step('a')
    assert do_eval(a @ v1, a=v2) == v1@v2
    a = Step('a')
    assert do_eval(v1 @ a, a=v2) == v1@v2

def test_CAS_disequalities(a):
    assert do_eval_uncached(a>5, a=6)
    assert do_eval_uncached(a>=5, a=6)
    assert do_eval_uncached(a<8, a=6)
    assert do_eval_uncached(a<=8, a=6)

def test_simplify_DAG():
    a1 = Step('a')
    a2 = Step('a')
    b = Step('b')
    c = Step('c')
    d1 = Step('d')
    d2 = Step('d')
    
    expr_1 = Step(d1, a1, b)
    expr_2 = Step(d2, a2, b)
    expr_3 = Step(lambda x, y: x*y, x=c, y=expr_2)
    expr = expr_1/expr_3
    
    old_expr = deepcopy(expr)
    new_expr = simplify(expr)
    assert are_equal(new_expr, old_expr)
    len_old = len({id(subdag) for subdag, *_ in old_expr})
    len_new = len({id(subdag) for subdag, *_ in expr})
    assert len_old == 11
    assert len_new == 7
    assert len_old >= len_new

def test_count_expressions():
    a = Step('a')
    b = Step('b')
    c = Step('c')
    dag = a + b + c
    do_eval(dag, a=1, b=2)
    res = count_operations(dag)
    assert res.n_of_nodes == 5
    assert res.n_of_operations == 5
    assert res.n_cached == 2
    assert res.n_variables == 3
    assert res.n_free_variables == 3

def test_multi_eval(a, b):
    def import_math():
        import math as _math
        return _math
    math = Step(import_math)
    def operation(x, y):
        z = x**2 + y**2
        sz = math.sqrt(z)
        return x/sz, y/sz
    xn, yn = operation(a, b)
    xn_v, yn_v = multi_eval(xn, yn, a=3, b=4)
    assert xn_v == 0.6
    assert yn_v == 0.8
    
def test_multiple_variables():
    a, b, c = variables('a', 'b', 'c', type=int)
    assert are_equal(a, Step('a', type=int))
    assert are_equal(b, Step('b', type=int))
    assert are_equal(c, Step('c', type=int))
    
def test_is_variable(a, b):
    assert is_variable(a)
    assert not is_variable(a+b)
    assert not is_variable(3)
    
    
def test_Template_basic():
    v = Template('v')
    assert are_equal(v, 1)
    assert are_equal(v.meaning, 1)
    assert repr(v) == "Template(name='v', 1)"
    
    
def test_Template_in_step():
    v = Template('v')
    expr = Step(op.add, 1, 2)
    test = Step(op.add, 1, v)
    assert are_equal(test, expr)
    assert are_equal(expr, test)
    assert are_equal(v.meaning, 2)
    
def test_Template_in_step_repeated_false():
    v = Template('v')
    expr = Step(op.add, 1, 2)
    test = Step(op.add, v, v)
    assert not are_equal(test, expr)
    assert not are_equal(expr, test)
    assert are_equal(v.meaning, 1)
    
def test_Template_in_step_repeated_true():
    v = Template('v')
    expr = Step(op.add, 1, 1)
    test = Step(op.add, v, v)
    assert are_equal(test, expr)
    assert are_equal(expr, test)
    assert are_equal(v.meaning, 1)
    
def test_Template_in_step_with_input():
    a = Step('a')
    b = Step('b')
    v = Template('v')
    expr = Step(op.add, a, b)
    test = Step(op.add, a, v)
    assert are_equal(expr, test)
    assert are_equal(test, expr)
    assert are_equal(v.meaning, b)
    
def test_Template_in_step_with_input_whole_operations():
    a = Step('a')
    b = Step('b')
    v = Template('v')
    expr = Step(op.add, 1, a+b)
    test = Step(op.add, 1, v)
    assert are_equal(expr, test)
    assert are_equal(test, expr)
    assert are_equal(v.meaning, a+b)
    
    
def test_Template_in_step_with_input_partial_operations():
    a = Step('a')
    b = Step('b')
    v = Template('v')
    expr = Step(op.add, 1, a+b)
    test = Step(op.add, 1, a+v)
    assert are_equal(expr, test)
    assert are_equal(test, expr)
    assert are_equal(v.meaning, b)
    
def test_Template_in_step_with_input_partial_operations_invert():
    a = Step('a')
    b = Step('b')
    v = Template('v')
    expr = Step(op.add, 1, a+b)
    test = Step(op.add, 1, v+b)
    assert are_equal(expr, test)
    assert are_equal(test, expr)
    assert are_equal(v.meaning, a)
    
def test_Template_in_step_with_input_False():
    a = Step('a')
    b = Step('b')
    v = Template('v')
    expr = Step(op.add, 1, a+b)
    test = Step(op.add, 1, a*v)
    assert not are_equal(expr, test)
    assert not are_equal(test, expr)
    assert are_equal(v.meaning, Tokens.NO_PREVIOUS_RESULT)

def test_template_is_DAG():
    v = Template('v')
    assert is_dag(v)

def test_multiple_templates():
    a = Step('a')
    b = Step('b')
    v = Template('v')
    t = Template('t')
    expr = 1+a+b
    test = v+a+t
    assert are_equal(expr, test)
    assert are_equal(v.meaning, 1)
    assert are_equal(t.meaning, b)
    
def test_template_iterable():
    v = Template('v')
    assert list(v) == [(v, None, None)]
    
def test_iteration_dag_with_template():
    a = Step('a')
    v = Template('v')
    
    test = a+v
    unroll = list(test)    
    assert len(unroll)==3
    assert unroll[-1] == (v, test, 1)
    
def test_pattern_matching():
    a = Step('a')
    b = Step('b')
    v = Template('v')
    
    expr = 2*(a+b)+3*(a+2*b)
    test = a+v
    assert not are_equal(expr, test)
    
    results = list(match(test, expr))
    assert len(results)==2
    assert are_equal(results[0][Tokens.FUNCTION_IDX][0], a+b)
    assert are_equal(results[0][Tokens.FUNCTION_IDX][1], 2*(a+b))
    assert are_equal(results[0][Tokens.FUNCTION_IDX][2], 1)
    assert are_equal(results[0]['v'], b)
    
    assert are_equal(results[1][Tokens.FUNCTION_IDX][0], a+2*b)
    assert are_equal(results[1][Tokens.FUNCTION_IDX][1], 3*(a+2*b))
    assert are_equal(results[1][Tokens.FUNCTION_IDX][2], 1)
    assert are_equal(results[1]['v'], 2*b)
