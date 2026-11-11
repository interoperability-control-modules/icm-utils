"""Tests for value transforms: one function or a transform(...) sequence applied to a single value, with inverses,
register references and the missing-reference policy."""
import pytest

from icm.transforms import MISSING, NotInvertible, ValueTransform, compile_value_transform


def test_single_function_is_the_modbus_tk_form():
    t = compile_value_transform('scale(0.001)')
    assert isinstance(t, ValueTransform) and t.functions == ('scale',) and t.invertible
    assert t.forward(2500) == 2.5 and t.inverse(2.5) == 2500


def test_sequence_applies_left_to_right_and_inverts_right_to_left():
    t = compile_value_transform('transform(scale(0.1), add(-40))')
    assert t.forward(500) == 10.0
    assert t.inverse(10.0) == 500.0
    assert t.functions == ('scale', 'add') and t.result_type is float and t.fidelity == 1.0


def test_wrapper_around_one_function_equals_the_bare_form():
    assert compile_value_transform('transform(scale(0.1))').forward(5) == compile_value_transform('scale(0.1)').forward(5)


def test_bare_comma_list_is_rejected_with_the_wrapped_form():
    with pytest.raises(ValueError, match=r'transform\(f1'):
        compile_value_transform('scale(0.1), add(-40)')


def test_source_path_and_unknown_function_and_argument_markers_are_rejected():
    with pytest.raises(ValueError, match='no source path'):
        compile_value_transform('transform[x](scale(0.1))')
    with pytest.raises(ValueError, match="unknown transform function 'nosuch'"):
        compile_value_transform('nosuch(1)')
    with pytest.raises(ValueError, match='marks an argument'):
        compile_value_transform('literal(3)')


def test_empty_text_is_the_identity():
    t = compile_value_transform('')
    assert t.identity and t.forward(7) == 7 and t.inverse(7) == 7


def test_none_input_yields_missing_not_an_error():
    assert compile_value_transform('scale(0.1)').forward(None) is MISSING


def test_non_invertible_chain_names_the_function():
    t = compile_value_transform('transform(scale(0.1), absolute(), add(1))')
    assert not t.invertible and t.non_invertible == ('absolute',)
    with pytest.raises(NotInvertible, match="'absolute' has no inverse"):
        t.inverse(3)


def test_comparisons_yield_booleans_and_report_their_primary_name():
    assert compile_value_transform('eq(7)').functions == ('equals',)
    assert compile_value_transform('lt(7)').forward(3) is True and compile_value_transform('ge(7)').forward(3) is False
    assert compile_value_transform('in_range(1, 5)').forward(5) is True
    t = compile_value_transform('ne(0)')
    assert t.result_type is bool and not t.invertible


def test_result_type_is_the_last_declared():
    assert compile_value_transform('transform(mod10k(True), cast_value(int))').result_type is int
    assert compile_value_transform('transform(scale(0.1), round(0))').result_type is int
    assert compile_value_transform('transform(scale(0.1), round(2))').result_type is float
    assert compile_value_transform('add(1)').result_type is None      # unchanged: whatever came in


def test_scale_reg_reads_the_values_mapping_and_declares_its_reference():
    t = compile_value_transform('scale_reg(divisor)')
    assert t.references == frozenset({'divisor'})
    assert t.forward(1000, {'divisor': 10}) == 100 and t.inverse(100, {'divisor': 10}) == 1000


def test_sunssf_scales_by_a_power_of_ten_and_treats_the_sentinel_as_absent():
    t = compile_value_transform('sunssf(W_SF)')
    assert t.references == frozenset({'W_SF'})
    assert t.forward(1234, {'W_SF': -1}) == 123.4 and t.forward(12, {'W_SF': 2}) == 1200
    assert t.inverse(123.4, {'W_SF': -1}) == 1234.0
    assert t.forward(1234, {'W_SF': -32768}) is MISSING          # sunssf "not implemented"
    assert t.forward(1234, {}) is MISSING                          # not yet read: held
    assert t.inverse(123.4, {}) is MISSING


def test_missing_reference_policy_pass_leaves_the_value_unchanged():
    held = compile_value_transform('scale_reg_pow_10(SF)')
    passed = compile_value_transform('scale_reg_pow_10(SF)', missing_reference='pass')
    assert held.forward(50, {}) is MISSING and passed.forward(50, {}) == 50
    assert held.missing_reference == 'hold' and passed.missing_reference == 'pass'
    assert held.forward(50, {'SF': 1}) == 500 == passed.forward(50, {'SF': 1})
    with pytest.raises(ValueError):
        compile_value_transform('scale(1)', missing_reference='maybe')


def test_point_with_register_functions_keep_the_value_and_invert():
    values = {'offset': 5, 'factor': 4}
    assert compile_value_transform('add_reg(offset)').forward(10, values) == 15
    assert compile_value_transform('add_reg(offset)').inverse(15, values) == 10
    assert compile_value_transform('subtract_reg(offset)').forward(10, values) == 5
    assert compile_value_transform('multiply_reg(factor)').forward(10, values) == 40
    assert compile_value_transform('divide_reg(factor)').forward(10, values) == 2.5
    assert compile_value_transform('divide_reg(factor)').forward(10, {'factor': 0}) is MISSING   # division by zero: undefined


def test_of_functions_define_a_derived_point_from_other_registers():
    values = {'a': 1, 'b': 2, 'c': 3}
    assert compile_value_transform('sum_of(a, b, c)').forward(None, values) == 6     # the point's own value is not used
    assert compile_value_transform('mean_of(a, b, c)').forward(None, values) == 2
    assert compile_value_transform('min_of(a, b, c)').forward(None, values) == 1
    assert compile_value_transform('max_of(a, b, c)').forward(None, values) == 3
    t = compile_value_transform('sum_of(a, b, c)')
    assert t.references == frozenset({'a', 'b', 'c'}) and not t.invertible and t.fidelity == 0.0
    assert compile_value_transform('sum_of(a, missing)').forward(None, values) == 1       # absent registers are left out
    assert compile_value_transform('sum_of(missing)').forward(None, values) is MISSING


def test_of_functions_accept_numeric_literals_and_the_two_markers():
    assert compile_value_transform('all_equal_of(3, foo, bar)').forward(None, {'foo': 3, 'bar': 3}) is True
    assert compile_value_transform('all_equal_of(3, foo, bar)').forward(None, {'foo': 3, 'bar': 4}) is False
    assert compile_value_transform('all_equal_of(3, foo, bar)').forward(None, {'foo': 3}) is MISSING   # a comparison with nothing
    assert compile_value_transform('sum_of(100, foo)').forward(None, {'foo': 1}) == 101
    assert compile_value_transform("all_equal_of(literal('ON'), a, b)").forward(None, {'a': 'ON', 'b': 'ON'}) is True
    t = compile_value_transform('sum_of(reg_name(3), foo)')
    assert t.references == frozenset({'3', 'foo'}) and t.forward(None, {'3': 10, 'foo': 1}) == 11
    assert compile_value_transform('all_of(a, b)').forward(None, {'a': 1, 'b': 'true'}) is True
    assert compile_value_transform('any_of(a, b)').forward(None, {'a': 0, 'b': 0}) is False


def test_of_result_can_be_chained_and_a_sequence_may_follow():
    assert compile_value_transform('transform(sum_of(a, b), scale(0.1))').forward(None, {'a': 7, 'b': 3}) == 1.0


def test_map_translates_an_enumeration_and_inverts_when_one_to_one():
    t = compile_value_transform("map(0, 'OFF', 1, 'ON')")
    assert t.forward(1) == 'ON' and t.inverse('OFF') == 0 and t.forward(5) is MISSING
    with_default = compile_value_transform("map(0, 'OFF', 1, 'ON', 'UNKNOWN')")
    assert with_default.forward(5) == 'UNKNOWN' and not with_default.invertible


def test_arithmetic_rounding_and_bit_functions():
    assert compile_value_transform('subtract(3)').forward(10) == 7 and compile_value_transform('subtract(3)').inverse(7) == 10
    assert compile_value_transform('divide(4)').forward(10) == 2.5 and compile_value_transform('divide(4)').inverse(2.5) == 10
    assert compile_value_transform('negate()').forward(3) == -3
    assert compile_value_transform('power(2)').forward(3) == 9 and compile_value_transform('power(2)').inverse(9) == 3
    assert compile_value_transform('root(2)').forward(9) == 3
    assert compile_value_transform('log(10)').forward(1000) == pytest.approx(3) and compile_value_transform('log(10)').inverse(3) == pytest.approx(1000)
    assert compile_value_transform('modulo(7)').forward(10) == 3 and not compile_value_transform('modulo(7)').invertible
    assert compile_value_transform('round(1)').forward(2.345) == 2.3 and compile_value_transform('round(0)').forward(2.5) == 2
    assert compile_value_transform('floor()').forward(2.7) == 2 and compile_value_transform('ceil()').forward(2.1) == 3
    assert compile_value_transform('clamp(0, 10)').forward(15) == 10 and compile_value_transform('clamp(0, 10)').inverse(7) == 7
    assert compile_value_transform('bit(2)').forward(0b0100) is True and compile_value_transform('bit(1)').forward(0b0100) is False
    assert compile_value_transform('bits(4, 4)').forward(0xAB) == 0xA and compile_value_transform('bits(4, 4)').inverse(0xA) == 0xA0
    assert compile_value_transform('mask(15)').forward(0xAB) == 0xB
    assert compile_value_transform('shift(4)').forward(1) == 16 and compile_value_transform('shift(4)').inverse(16) == 1
    assert compile_value_transform('invert(8)').forward(0xF0) == 0x0F and compile_value_transform('invert(8)').inverse(0x0F) == 0xF0
    assert compile_value_transform('bool()').forward('yes') is True and compile_value_transform('not()').forward(1) is False
    with pytest.raises(ValueError):
        compile_value_transform('divide(0)')


def test_compiled_transforms_are_cached_by_text_and_options():
    assert compile_value_transform('scale(0.5)') is compile_value_transform('scale(0.5)')
    assert compile_value_transform('scale(0.5)') is not compile_value_transform('scale(0.5)', missing_reference='pass')


def test_parameters_bind_at_compile_time():
    t = compile_value_transform("param('mrid')", context={'mrid': 'abc'})
    assert t.forward(0) == 'abc' and t.fidelity == 0.0


def test_defines_value_marks_register_only_chains():
    assert compile_value_transform('sum_of(a, b)').defines_value
    assert compile_value_transform('transform(mean_of(a, b), scale(0.1))').defines_value
    assert not compile_value_transform('scale_reg(a)').defines_value
    assert not compile_value_transform('transform(scale(0.1), sum_of(a))').defines_value
    assert not compile_value_transform('').defines_value


def test_scale_decimal_int_signed_matches_modbus_tk():
    t = compile_value_transform('scale_decimal_int_signed(0.001)')
    assert t.forward(1500) == 1.5 and t.forward(-1500) == pytest.approx(0.001 * (0 - (-1500 + 32768)))
    assert t.forward(-32668) == pytest.approx(-0.1)                    # a small negative is stored near -32768
    assert t.inverse(1.5) == 1500.0
    for raw in (-32668, -1500, -1):                                   # the inverse undoes the encoding exactly
        assert t.inverse(t.forward(raw)) == pytest.approx(raw)
    assert compile_value_transform("scale_decimal_int_signed('0.01')").forward(250) == 2.5


def test_invalid_and_finite_hold_sentinels_out_and_pass_writes_through():
    t = compile_value_transform('invalid(-32768)')
    assert t.forward(-32768) is MISSING and t.forward(1234) == 1234 and t.inverse(1234) == 1234 and t.invertible
    assert compile_value_transform('invalid(0xFFFF, 0xFFFE)').forward(65535) is MISSING
    assert compile_value_transform("invalid('N/A')").forward('N/A') is MISSING
    chain = compile_value_transform('transform(invalid(-32768), sunssf(W_SF))')
    assert chain.forward(-32768, {'W_SF': -1}) is MISSING and chain.forward(1234, {'W_SF': -1}) == 123.4
    assert chain.inverse(123.4, {'W_SF': -1}) == 1234.0 and chain.result_type is float
    f = compile_value_transform('finite()')
    assert f.forward(float('nan')) is MISSING and f.forward(float('inf')) is MISSING and f.forward(2.5) == 2.5
    with pytest.raises(ValueError):
        compile_value_transform('invalid()')


def test_value_of_defines_a_derived_point_from_one_register():
    t = compile_value_transform('transform(value_of(Status), bit(3))')
    assert t.defines_value and t.references == frozenset({'Status'}) and not t.invertible
    assert t.forward(None, {'Status': 0b1000}) is True and t.forward(None, {'Status': 0b0111}) is False
    assert t.forward(None, {'Status': None}) is MISSING
    assert compile_value_transform('value_of(W)').forward(None, {'W': 42}) == 42
    with pytest.raises(ValueError, match='name of a register'):
        compile_value_transform('value_of(3)')
