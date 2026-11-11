"""Library of functions usable inside transform expressions (``icm.transforms.functions``), e.g. ``transform[x](multiple(7), add(9))``.

Each public function is called at parse time with the literal arguments written in the expression
(integers, or bare/quoted words as strings) and returns a convtools conversion that is applied to
the value found at the expression's source path.
"""
import contextvars
import math
import time
import uuid

from convtools import conversion as c


class _Missing:
    """Sentinel for a source field that is absent from the input message."""
    __slots__ = ()

    def __repr__(self):
        return '<MISSING>'


MISSING = _Missing()

# Label under which the transform parser stores the element enclosing the expression currently being
# parsed: the message root at top level, or the current element inside a repeated ("name[#]") group.
# The parser sets it while parsing each expression, so library functions may capture scope() at build time.
scope_label = contextvars.ContextVar('scope_label', default='each0')


ROOT_LABEL = 'each0'

# Per-resource values a transform may refer to with ``param('name')`` (a device's OpenFMB mRID, for example).
# The parser sets it while compiling a chain for one resource, so library functions bind the value at build time.
parse_context = contextvars.ContextVar('parse_context', default={})

# What a function that reads another register does when that register is absent: ``'pass'`` leaves the value
# unchanged (a resource pattern, where an absent factor usually means the driver already applied it) or ``'hold'``
# yields MISSING so the value is not produced at all (a driver point, where the registry is known and absence never
# means that). Bound at build time by whoever compiles the expression; ``'pass'`` is the historical behaviour.
missing_reference = contextvars.ContextVar('missing_reference', default='pass')


def _absent():
    """Conversion for "the referenced register is not there" under the current missing-reference policy."""
    return c.this if missing_reference.get() == 'pass' else c.naive(MISSING)


class _Literal:
    """An argument that is a value, not a register name (see ``literal``)."""
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f'literal({self.value!r})'


class _RegName:
    """An argument that is a register name even though it looks like a number (see ``reg_name``)."""
    __slots__ = ('name',)

    def __init__(self, name):
        self.name = str(name)

    def __repr__(self):
        return f'reg_name({self.name!r})'


def literal(value):
    """
        Mark an argument of an ``_of`` function as a value rather than a register name: ``all_equal_of(literal('ON'),
         state_a, state_b)``. Numbers are values without the marker; bare and quoted words are register names.
    """
    return _Literal(value)


def reg_name(name):
    """Mark an argument of an ``_of`` function as a register name even though it is all digits: ``sum_of(reg_name(3), foo)``."""
    return _RegName(name)


def _operand(arg):
    """Conversion for one argument of a function over several registers: numbers and ``literal()`` are values, bare or
    quoted words and ``reg_name()`` are registers resolved against the enclosing element."""
    if isinstance(arg, _Literal):
        return c.naive(arg.value), None
    if isinstance(arg, _RegName):
        return _register(arg.name), arg.name
    if isinstance(arg, bool) or not isinstance(arg, (int, float, str)):
        raise TypeError(f'arguments of functions over several registers are numbers, register names, literal() or '
                        f'reg_name(); got {arg!r}')
    if isinstance(arg, (int, float)):
        return c.naive(arg), None
    return _register(arg), arg


def scope():
    """Conversion yielding the element that encloses the expression being built (see ``scope_label``)."""
    return c.label(scope_label.get())


def scoped_path(path: str):
    """Conversion resolving a quoted dotted path: against the enclosing element, or against the whole message
    when the path starts with ``/`` (``'/705_V_SF'`` reaches a flat driver point from inside a repeated group)."""
    if path.startswith('/'):
        return c.call_func(_resolve_path, c.label(ROOT_LABEL), path[1:])
    return c.call_func(_resolve_path, scope(), path)


def approx(fidelity: float | int | str):
    """
        Annotation for an approximate mapping: copies the value unchanged at runtime but records that only
         ``fidelity`` (0 to 1) of its meaning carries over, e.g. ``transform[DCTE, RtnSrvDlyTim](approx(0.5))``
         where a delay register stands in for a random delay. The registry uses it when weighting paths.
    """
    conv = c.this.pipe(c.this)
    conv.fidelity = float(fidelity)
    conv.inverse = c.this.pipe(c.this)
    return conv


def _resolve_path(value, path: str):
    """Follow a dotted path through dicts and lists; digits index lists. Returns None if absent."""
    for segment in path.split('.'):
        try:
            if isinstance(value, (list, tuple)):
                value = value[int(segment)]
            else:
                value = value[segment]
        except (KeyError, IndexError, TypeError, ValueError):
            return None
    return value


def multiple(multiplier: float | int):
    """Builds a conversion and its inverse."""
    conv = c.this * multiplier
    conv.inverse = c.this / multiplier
    return conv

def add(addend: float | int):
    """Builds a conversion and its inverse."""
    conv = (c.this + addend)
    conv.inverse = c.this - addend
    return conv

def scale_decimal_int_signed(multiplier):
    """
        Scale float value that is stored as a decimal number, not using standard signing rollover,
         as the PM800 Power Factor Registers.
    """
    multiplier = _numeric(multiplier)
    # A negative register holds -(magnitude) - 32768: the magnitude is -(raw + 32768), then scaled. Conversions define
    # the arithmetic operators but not their reflected forms, so the conversion goes on the left.
    conv = c.if_(c.this < 0, (c.naive(0) - (c.this + 32768)) * multiplier, c.this * multiplier)
    # The inverse undoes exactly that: -(value / multiplier) - 32768 for a negative value. modbus_tk's inverse used
    # 0xFFFF here and never round-tripped (it also referred to an undefined name, so it never ran).
    conv.inverse = c.if_(c.this < 0, (c.naive(0) - (c.this / float(multiplier))) - 32768, c.this / float(multiplier))
    return conv

def _numeric(arg):
    """Accept a number written as an int, float or string (the modbus_tk register maps pass strings)."""
    if isinstance(arg, (int, float)) and not isinstance(arg, bool):
        return arg
    try:
        return int(str(arg), 10)
    except ValueError:
        return float(arg)


def _decimal_places(*values) -> int:
    """Total number of decimal digits written in the given factors (integers contribute none)."""
    digits = 0
    for value in values:
        if isinstance(value, float):
            text = repr(value)
            if '.' in text and 'e' not in text:
                digits += len(text.split('.')[1])
    return digits


def _fix_decimals(product, *factors):
    """Round a product to the decimal places of its factors, as modbus_tk's ``transform_func_helper`` does,
    so 0.001 * 12345 gives 12.345 rather than 12.345000000000001."""
    try:
        return round(product, _decimal_places(*factors))
    except TypeError:
        return product


def scale(multiplier: float | int | str):
    """
        Scale a register value by a constant multiplier, fixing floating point noise to the decimal
         places of the operands (modbus_tk ``scale``). The inverse divides; division by zero gives None
         and a non-numeric value is passed through.
    """
    multiplier = _numeric(multiplier)
    conv = c.call_func(_fix_decimals, c.this * multiplier, c.this, multiplier)
    conv.inverse = (c.try_(c.this / float(multiplier))
                    .except_(ZeroDivisionError, None)
                    .except_(TypeError, c.this))
    return conv


def scale_int(multiplier):
    multiplier = _numeric(multiplier)
    conv = (c.this * multiplier).as_type(int)
    conv.inverse = c.try_((c.this / c.naive(multiplier).as_type(float)).as_type(int)).except_(ZeroDivisionError, None)
    return conv


def _register(path: str):
    """Conversion reading a scaling register by quoted dotted path (see ``scoped_path``)."""
    return scoped_path(str(path))


def _reference_name(path) -> str:
    """The register a function reads, as a plain name: ``'/705_V_SF'`` and ``'705_V_SF'`` both name ``705_V_SF``."""
    return str(path).lstrip('/')


def scale_reg(register: str):
    """
        Divide the value by another register's value, named by a quoted dotted path resolved against the
         element enclosing the expression (modbus_tk ``scale_reg``), e.g. ``scale_reg('701.W_SF')``. When the
         scaling register is absent the value passes through unchanged, so a message from a driver that has
         already applied the factor (and no longer publishes it) is not altered; division by zero gives None.
         The inverse multiplies by the register.
    """
    reg = _register(register)
    conv = c.if_(reg.is_(None), _absent(), c.try_(c.this / reg).except_(ZeroDivisionError, None))
    conv.inverse = c.if_(reg.is_(None), _absent(), c.this * reg)
    conv.references = frozenset({_reference_name(register)})
    return conv


def scale_reg_pow_10(register: str):
    """
        Multiply the value by 10 to the power of another register's value (modbus_tk ``scale_reg_pow_10``),
         which is how SunSpec scale factor registers work: ``scale_reg_pow_10('701.W_SF')`` with W_SF = -1
         scales by 0.1. The register is a quoted dotted path resolved against the element enclosing the
         expression. When it is absent the value passes through unchanged (see ``scale_reg``). The inverse divides.
    """
    reg = _register(register)
    factor = c.naive(10.0) ** reg
    conv = c.if_(reg.is_(None), _absent(), c.call_func(_fix_decimals, c.this * factor, c.this, factor))
    conv.inverse = c.if_(reg.is_(None), _absent(), c.this / factor)
    conv.references = frozenset({_reference_name(register)})
    return conv


SUNSSF_NOT_IMPLEMENTED = -32768      # the sunssf type's "not implemented" value (SunSpec Device Information Model, 6.4.8)


def sunssf(register: str):
    """
        Scale by a SunSpec scale factor: the ``sunssf`` register named by ``register`` holds a signed base-10 exponent and
         the engineering value is the register times 10 to that power (``sunssf(W_SF)``). ``scale_reg_pow_10`` plus the
         type's semantics: a factor of -32768 is "not implemented" and counts as absent, so the value is not scaled by
         10 to a large negative power. The inverse divides.
    """
    reg = _register(register)
    unavailable = reg.is_(None) | (reg == SUNSSF_NOT_IMPLEMENTED)
    factor = c.naive(10.0) ** reg
    conv = c.if_(unavailable, _absent(), c.call_func(_fix_decimals, c.this * factor, c.this, factor))
    conv.inverse = c.if_(unavailable, _absent(), c.this / factor)
    conv.references = frozenset({_reference_name(register)})
    return conv


def _with_register(register: str, forward, inverse):
    reg = _register(register)
    conv = c.if_(reg.is_(None), _absent(), forward(reg))
    conv.inverse = c.if_(reg.is_(None), _absent(), inverse(reg))
    conv.references = frozenset({_reference_name(register)})
    return conv


def add_reg(register: str):
    """Add another register's value to the value (``add_reg(offset)``); the inverse subtracts it."""
    return _with_register(register, lambda reg: c.this + reg, lambda reg: c.this - reg)


def subtract_reg(register: str):
    """Subtract another register's value from the value; the inverse adds it."""
    return _with_register(register, lambda reg: c.this - reg, lambda reg: c.this + reg)


def multiply_reg(register: str):
    """Multiply the value by another register's value; the inverse divides (None when the register is zero)."""
    return _with_register(register, lambda reg: c.this * reg,
                          lambda reg: c.try_(c.this / reg).except_(ZeroDivisionError, None))


def divide_reg(register: str):
    """Divide the value by another register's value (None when it is zero); the inverse multiplies."""
    return _with_register(register, lambda reg: c.try_(c.this / reg).except_(ZeroDivisionError, None),
                          lambda reg: c.this * reg)


# ---- Functions over several registers: the point's own value is not carried ----

def _present(values):
    return [v for v in values if v is not None and v is not MISSING]


def _sum_of(values):
    present = _present(values)
    return sum(present) if present else MISSING


def _mean_of(values):
    present = _present(values)
    return sum(present) / len(present) if present else MISSING


def _min_of(values):
    present = _present(values)
    return min(present) if present else MISSING


def _max_of(values):
    present = _present(values)
    return max(present) if present else MISSING


def _all_equal_of(values):
    if any(v is None or v is MISSING for v in values) or not values:
        return MISSING
    return all(v == values[0] for v in values[1:])


def _all_of(values):
    if any(v is None or v is MISSING for v in values) or not values:
        return MISSING
    return all(_flag(v) for v in values)


def _any_of(values):
    if any(v is None or v is MISSING for v in values) or not values:
        return MISSING
    return any(_flag(v) for v in values)


def _over(reducer, args):
    if not args:
        raise TypeError(f'{reducer.__name__.strip("_")} needs at least one register name or value.')
    operands = [_operand(a) for a in args]
    conv = c.call_func(reducer, c([op for op, _ in operands]))
    conv.references = frozenset(_reference_name(name) for _, name in operands if name is not None)
    conv.defines_value = True         # the value transform pipeline must not stop it on a None input
    return conv


def _value_of(values):
    present = _present(values)
    return present[0] if present else MISSING


def value_of(register):
    """The value of one other register, as the start of a derived point's chain: ``transform(value_of(Status), bit(3))``
    publishes bit 3 of ``Status`` as a point of its own, without a second register at the same address. MISSING while
    the register has no value. Defines a derived point: the point's own value is not used."""
    if isinstance(register, (int, float, bool)) or isinstance(register, _Literal):
        raise ValueError('value_of() takes the name of a register, not a literal value.')
    return _over(_value_of, (register,))


def sum_of(*args):
    """The sum of the named registers (and any literal numbers): ``sum_of(phase_a, phase_b, phase_c)``. Registers
    without a value are left out; MISSING when none has one. Defines a derived point: the point's own value is not used."""
    return _over(_sum_of, args)


def mean_of(*args):
    """The mean of the named registers' values (missing ones left out); the register-name form of ``mean``."""
    return _over(_mean_of, args)


def min_of(*args):
    """The smallest of the named registers' values (and any literal numbers)."""
    return _over(_min_of, args)


def max_of(*args):
    """The largest of the named registers' values (and any literal numbers)."""
    return _over(_max_of, args)


def all_equal_of(*args):
    """Whether the named registers (and any literal) all hold the same value: ``all_equal_of(3, foo, bar)`` is "foo and
    bar both equal 3". MISSING when any of them has no value."""
    return _over(_all_equal_of, args)


def all_of(*args):
    """Whether every named register is true (as ``cast_value(bool)`` reads it). MISSING when any has no value."""
    return _over(_all_of, args)


def any_of(*args):
    """Whether any named register is true. MISSING when any has no value."""
    return _over(_any_of, args)


def no_op():
    """Copy the value unchanged (modbus_tk ``no_op``); its own inverse."""
    conv = c.this.pipe(c.this)
    conv.inverse = c.this.pipe(c.this)
    return conv


def _sentinel(arg):
    """A sentinel as written in a registry: a number, a hexadecimal such as ``0x8000``, or any other text as is."""
    if isinstance(arg, (int, float)) and not isinstance(arg, bool):
        return arg
    text = str(arg).strip()
    try:
        return int(text, 0)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def invalid(*values):
    """
        No value when the raw value is one of ``values``: a device's "not implemented" or error sentinel, such as
         SunSpec's ``invalid(-32768)`` for an int16 and ``invalid(65535)`` for a uint16 (``0x8000`` and ``0xFFFF`` are
         accepted spellings). Any other value passes through, and so does a write: the inverse is the identity. First in
         a chain, it keeps a sentinel from being scaled into a number: ``transform(invalid(-32768), sunssf(W_SF))``.
    """
    if not values:
        raise ValueError('invalid() needs at least one sentinel value.')
    sentinels = tuple(_sentinel(v) for v in values)
    conv = c.call_func(lambda v: MISSING if v is None or v in sentinels else v, c.this)
    conv.inverse = c.this.pipe(c.this)
    return conv


def finite():
    """
        No value when the raw value is NaN or infinite (SunSpec's float32 "not implemented" is NaN); other values pass
         through, and so does a write.
    """
    conv = c.call_func(lambda v: MISSING if v is None or (isinstance(v, float) and not math.isfinite(v)) else v, c.this)
    conv.inverse = c.this.pipe(c.this)
    return conv


def _word(index: int):
    """The ``index``-th 16 bit register of an unsigned integer packed from several registers (0 is the lowest)."""
    return (c.this // (0x10000 ** index)) % 0x10000


def _decimal_group(index: int):
    """The ``index``-th group of four decimal digits of an integer (0 is the lowest)."""
    return (c.this // (10000 ** index)) % 10000


def _mod10k(order: list[int]):
    """Build a MOD10K conversion: each 16 bit register holds four decimal digits (0 to 9999).

    ``order`` lists the register indexes (0 is the lowest 16 bits of the packed integer) from the most
    significant group of digits to the least. Only valid for non-negative values, as in the modbus_tk
    driver; the inverse packs the digit groups back into registers.
    """
    conv = sum((_word(w) * 10000 ** (len(order) - 1 - i) for i, w in enumerate(order)), c.naive(0))
    conv.inverse = sum((_decimal_group(len(order) - 1 - i) * 0x10000 ** w for i, w in enumerate(order)), c.naive(0))
    return conv


def _flag(arg) -> bool:
    return arg if isinstance(arg, bool) else _convert_to_boolean(arg)


def mod10k(reverse=False):
    """
        Decode the ION INT32-M10K format: two registers, each holding four decimal digits, high register
         first (``mod10k(True)`` for meters that store them reversed). Positive values only.
    """
    return _mod10k([0, 1] if _flag(reverse) else [1, 0])


def mod10k64(reverse=False):
    """Decode the PM800 64 bit M10K format: four registers of four decimal digits, lowest register first."""
    return _mod10k([3, 2, 1, 0] if _flag(reverse) else [0, 1, 2, 3])


def mod10k48(reverse=False):
    """Decode the PM800 INT48-M10K format: three of the four registers of a 64 bit word, as modbus_tk does
    (the second lowest register first; reversed, the lowest register first)."""
    return _mod10k([0, 1, 2] if _flag(reverse) else [1, 2, 3])

def _convert_to_boolean(v):
    v = v.decode('utf8') if isinstance(v, bytes) else v
    v = v.strip().lower() if isinstance(v, str) else v
    match v:
        case True:
            return True
        case False:
            return False
        case x if x in ('t', 'y', 'true', 'yes', '1', 1):
            return True
        case x if x in ('f', 'n', 'false', 'no', '0', 0, ''):
            return False
        case _:
            raise ValueError(f'Unable to convert {v} to Boolean.')

def cast_value(type_name):
    return c.this.dispatch(
        type_name,
        {
            'bool': c.call_func(_convert_to_boolean, c.this),
            'str': c.this.as_type(str),
            'int': c.this.as_type(int),
            'float': c.this.as_type(float),
            'list': c.this.as_type(list),
            'tuple': c.this.as_type(tuple),
            'dict': c.this.as_type(dict)
        }
    )


def mean(*paths):
    """
        Average of several fields of the current value. Each argument is a path to a field, with nested
         keys separated by dots (e.g., "phsA.mag"). Fields that are missing or None are ignored;
         the result is None if no fields have a value.
    """
    fields = [c.item(*path.split('.'), default=None) for path in paths]
    return c(fields).pipe(c.aggregate(c.ReduceFuncs.Average(c.this, where=c.this.is_not(None))))


def _take(sequence, count):
    if count is None or not isinstance(sequence, (list, tuple)):
        return sequence
    return list(sequence)[:int(count)]


def take(count: int | str):
    """
        Keep only the first ``count`` elements of a list. ``count`` is either a number or a quoted dotted
         path to the field holding the number, resolved against the element enclosing the expression
         (the message root at top level), e.g. ``take('705.Crv.0.ActPt')`` for SunSpec active points.
         If the referenced field is absent the list is passed through unchanged.
    """
    if isinstance(count, str):
        n = scoped_path(count)
    else:
        n = c.naive(count)
    return c.call_func(_take, c.this, n)


def _pairs(sequence, start, count, x, y):
    points = []
    for i in range(count):
        xv = _resolve_path(sequence, str(start + 2 * i))
        yv = _resolve_path(sequence, str(start + 2 * i + 1))
        if xv is None and yv is None:
            continue
        points.append({x: MISSING if xv is None else xv, y: MISSING if yv is None else yv})
    return points


def pairs(start: int, count: int, x: str = 'x', y: str = 'y'):
    """
        Turn a flat, position-indexed array of alternating X and Y values (e.g., the IEEE 1815.2 curve
         point analog outputs, X1 Y1 X2 Y2 ...) into a list of ``{x: ..., y: ...}`` points. ``start`` is
         the index of the first X value and ``count`` the maximum number of points. Points with neither
         value present are skipped. Combine with ``take`` to honour a "number of points" field.
    """
    return c.call_func(_pairs, c.this, start, count, x, y)


def _count(value):
    return len(value) if isinstance(value, (list, tuple, dict)) else MISSING


def count():
    """Number of elements in a list (or keys in a group), e.g. SunSpec ``ActPt`` from 2030.5 ``CurveData``."""
    return c.call_func(_count, c.this)


def as_list():
    """Wrap a single value in a one-element list, e.g. to emit one curve into a SunSpec ``Crv`` array."""
    return c.call_func(lambda value: [value], c.this)


def const(value):
    """Replace the value with a literal, e.g. ``transform[DERCurve, opModVoltVar](const(2))`` emits 2 only when
    the source is present."""
    return c.naive(value)


def param(name: str):
    """
        A per-resource parameter bound when the chain is compiled, e.g. ``transform(param('mrid'))`` for the
         device mRID an OpenFMB header carries. The values come from the ``parameters`` of the resource being
         resolved (``TransformParser(context=...)``); an unbound name yields MISSING and the field is dropped.
    """
    return c.naive(parse_context.get().get(name, MISSING))


def uuid4():
    """A fresh random UUID string on every message, e.g. for an OpenFMB ``messageInfo.identifiedObject.mRID``."""
    return c.call_func(lambda: str(uuid.uuid4()))


def _timestamp():
    now = time.time()
    whole = int(now)
    return {'seconds': whole, 'nanoseconds': int(round((now - whole) * 1e9))}


def timestamp():
    """The current time as an OpenFMB ``Timestamp`` group, ``{'seconds': ..., 'nanoseconds': ...}``."""
    return c.call_func(_timestamp)


def _device_values(message):
    if isinstance(message, (list, tuple)) and len(message) == 2 and isinstance(message[0], dict):
        return message[0]
    return message


def device_values():
    """
        The point values of a platform driver ``devices/.../all`` message, which arrives as ``[values, meta]``
         (the modular driver) or as the bare ``{point: value}`` dict; either shape yields the dict. Meant for
         a pattern's root ``"#"`` entry so the rest of the pattern reads flat points. Has an inverse (identity).
    """
    conv = c.call_func(_device_values, c.this)
    conv.inverse = c.this.pipe(c.this)
    return conv


def _immediate(entries, path):
    if not isinstance(entries, (list, tuple)):
        return MISSING
    now = time.time()
    active, started = MISSING, None
    for entry in entries:
        start = entry.get('startTime') if isinstance(entry, dict) else None
        seconds = start.get('seconds') if isinstance(start, dict) else start
        if isinstance(seconds, str):
            try:
                seconds = float(seconds)                   # protobuf JSON carries int64 as a string
            except ValueError:
                continue
        if seconds in (None, 0):
            seconds = now                                  # no start time: in effect now
        if isinstance(seconds, (int, float)) and seconds <= now and (started is None or seconds >= started):
            active, started = entry, seconds
    value = active
    for segment in path:
        value = value.get(segment, MISSING) if isinstance(value, dict) else MISSING
        if value is MISSING:
            break
    return value


def immediate(*path: str):
    """
        From a list of schedule entries (OpenFMB ``crvPts`` with ``startTime`` and ``control``), the entry in
         effect now: the one with the latest ``startTime.seconds`` already reached, an entry without a start time
         counting as starting now. The quoted ``path`` segments are then followed inside that entry; no entry in
         effect, or a missing field, yields MISSING. This is the inverse of the ``as_list()`` convention that
         emits the hub's top-level logical nodes as the "immediate" schedule entry, so a schedule round trip
         through the hub carries its active entry twice: once immediate, once scheduled.
    """
    conv = c.call_func(_immediate, c.this, tuple(path))
    conv.fidelity = 1.0                                        # picking the active entry loses nothing
    return conv


def _equals(value, expected):
    return MISSING if value is None else value == expected


def equals(expected):
    """
        The value becomes the flag "equal to ``expected``": an enumeration code turns into a boolean, e.g. SunSpec's
         excitation ``Ext`` (0 over-excited, 1 under-excited) into 61850's ``PFExtSet`` (TRUE over-excited) with
         ``equals(0)``.
    """
    return c.call_func(_equals, c.this, expected)


def _select(value, if_true, if_false):
    return MISSING if value is None else (if_true if _flag(value) else if_false)


def select(if_true, if_false):
    """
        A flag becomes one of two values: ``select(0, 1)`` encodes TRUE as 0 and FALSE as 1, the inverse of
         ``equals(0)``.
    """
    return c.call_func(_select, c.this, if_true, if_false)


def _compare(value, expected, op):
    return MISSING if value is None else op(value, expected)


def not_equals(expected):
    """The value becomes the flag "not equal to ``expected``" (alias ``ne``)."""
    return c.call_func(_compare, c.this, expected, lambda v, e: v != e)


def less_than(limit):
    """The value becomes the flag "less than ``limit``" (alias ``lt``)."""
    return c.call_func(_compare, c.this, limit, lambda v, e: v < e)


def less_or_equal(limit):
    """The value becomes the flag "less than or equal to ``limit``" (alias ``le``)."""
    return c.call_func(_compare, c.this, limit, lambda v, e: v <= e)


def greater_than(limit):
    """The value becomes the flag "greater than ``limit``" (alias ``gt``)."""
    return c.call_func(_compare, c.this, limit, lambda v, e: v > e)


def greater_or_equal(limit):
    """The value becomes the flag "greater than or equal to ``limit``" (alias ``ge``)."""
    return c.call_func(_compare, c.this, limit, lambda v, e: v >= e)


def in_range(low, high):
    """The value becomes the flag "between ``low`` and ``high``, inclusive"."""
    return c.call_func(_compare, c.this, (low, high), lambda v, e: e[0] <= v <= e[1])


def subtract(subtrahend: float | int):
    """Subtract a constant; the inverse adds it."""
    conv = c.this - subtrahend
    conv.inverse = c.this + subtrahend
    return conv


def divide(divisor: float | int):
    """Divide by a constant; the inverse multiplies. A zero divisor is refused when the expression is compiled."""
    divisor = _numeric(divisor)
    if divisor == 0:
        raise ValueError('divide(0): the divisor must not be zero.')
    conv = c.this / divisor
    conv.inverse = c.this * divisor
    return conv


def negate():
    """Change the sign; its own inverse."""
    conv = c.this * -1
    conv.inverse = c.this * -1
    return conv


def absolute():
    """The absolute value. No inverse: the sign is gone."""
    return c.call_func(abs, c.this)


def power(exponent: float | int):
    """Raise to a constant power; the inverse takes the matching root."""
    exponent = _numeric(exponent)
    if exponent == 0:
        raise ValueError('power(0): the exponent must not be zero.')
    conv = c.this ** exponent
    conv.inverse = c.this ** (1 / exponent)
    return conv


def root(degree: float | int):
    """Take a constant root (``root(2)`` is the square root); the inverse raises to that power."""
    degree = _numeric(degree)
    if degree == 0:
        raise ValueError('root(0): the degree must not be zero.')
    conv = c.this ** (1 / degree)
    conv.inverse = c.this ** degree
    return conv


def log(base: float | int = math.e):
    """The logarithm to ``base`` (natural when omitted); the inverse raises the base to the value."""
    base = _numeric(base)
    conv = c.call_func(math.log, c.this, base)
    conv.inverse = c.naive(base) ** c.this
    return conv


def modulo(divisor: float | int):
    """The remainder of dividing by a constant. No inverse: the quotient is gone."""
    return c.this % _numeric(divisor)


def _round(places: int = 0):
    """Round to ``places`` decimal places (an int when ``places`` is 0). Lossy, so the inverse is the identity."""
    places = int(places)
    conv = c.call_func(round, c.this, places) if places else c.call_func(lambda v: int(round(v)), c.this)
    conv.inverse = c.this.pipe(c.this)
    conv.result_type = int if places == 0 else float
    return conv


def floor():
    """Round down to an integer. Lossy, so the inverse is the identity."""
    conv = c.call_func(math.floor, c.this)
    conv.inverse = c.this.pipe(c.this)
    return conv


def ceil():
    """Round up to an integer. Lossy, so the inverse is the identity."""
    conv = c.call_func(math.ceil, c.this)
    conv.inverse = c.this.pipe(c.this)
    return conv


def clamp(low: float | int, high: float | int):
    """Limit the value to ``[low, high]``. Lossy, so the inverse is the identity."""
    low, high = _numeric(low), _numeric(high)
    conv = c.call_func(lambda v: min(max(v, low), high), c.this)
    conv.inverse = c.this.pipe(c.this)
    return conv


def bit(index: int):
    """The flag held by bit ``index`` of an integer value (bit 0 is the least significant)."""
    index = int(index)
    return c.call_func(lambda v: bool((int(v) >> index) & 1), c.this)


def bits(start: int, count: int):
    """The integer held by ``count`` bits from bit ``start``; the inverse places it back (other bits zero)."""
    start, count = int(start), int(count)
    mask_value = (1 << count) - 1
    conv = c.call_func(lambda v: (int(v) >> start) & mask_value, c.this)
    conv.inverse = c.call_func(lambda v: (int(v) & mask_value) << start, c.this)
    return conv


def mask(value: int):
    """The value AND a constant mask. No inverse: the masked bits are gone."""
    value = int(value)
    return c.call_func(lambda v: int(v) & value, c.this)


def shift(count: int):
    """Shift left by ``count`` bits (right when negative); the inverse shifts the other way."""
    count = int(count)
    left = (lambda v: int(v) << count) if count >= 0 else (lambda v: int(v) >> -count)
    right = (lambda v: int(v) >> count) if count >= 0 else (lambda v: int(v) << -count)
    conv = c.call_func(left, c.this)
    conv.inverse = c.call_func(right, c.this)
    return conv


def invert(width: int | None = None):
    """Flip every bit of a ``width``-bit unsigned value (two's complement NOT without a width); its own inverse."""
    if width is None:
        flip = lambda v: ~int(v)
    else:
        all_ones = (1 << int(width)) - 1
        flip = lambda v: int(v) ^ all_ones
    conv = c.call_func(flip, c.this)
    conv.inverse = c.call_func(flip, c.this)
    return conv


def _bool():
    """The value as a flag, read as ``cast_value(bool)`` reads it (``1``, ``'true'``, ``'yes'`` ... are true)."""
    return c.call_func(_convert_to_boolean, c.this)


def _not():
    """The opposite flag; its own inverse."""
    conv = c.call_func(lambda v: MISSING if v is None else not _flag(v), c.this)
    conv.inverse = c.call_func(lambda v: MISSING if v is None else not _flag(v), c.this)
    return conv


def _map(*pairs):
    """
        Translate an enumeration: ``map(0, 'OFF', 1, 'ON')`` reads 0 as ``'OFF'`` and 1 as ``'ON'``. An odd trailing
         argument is the default for codes not listed; without one an unlisted code yields MISSING. When every value is
         distinct and there is no default, the inverse translates back.
    """
    if len(pairs) < 2:
        raise TypeError('map() needs at least one code and its value.')
    default = pairs[-1] if len(pairs) % 2 else MISSING
    table = dict(zip(pairs[0::2], pairs[1::2]))
    conv = c.call_func(lambda v: MISSING if v is None else table.get(v, default), c.this)
    if default is MISSING and len(set(table.values())) == len(table):
        back = {v: k for k, v in table.items()}
        conv.inverse = c.call_func(lambda v: MISSING if v is None else back.get(v, MISSING), c.this)
    return conv


#: Functions whose names are Python keywords or builtins: defined above under another name, registered under this one.
RESERVED_NAMES = {'bool': _bool, 'not': _not, 'round': _round, 'map': _map}


def _whole_seconds(value):
    return MISSING if value is None else int(float(value) // 1)


def whole_seconds():
    """The integer seconds of a duration given in seconds (``0.05`` -> ``0``), for a protobuf ``seconds`` field."""
    return c.call_func(_whole_seconds, c.this)


def _nanoseconds(value):
    return MISSING if value is None else int(round((float(value) - float(value) // 1) * 1_000_000_000))


def nanoseconds():
    """The fractional part of a duration given in seconds, as integer nanoseconds (``0.05`` -> ``50000000``)."""
    return c.call_func(_nanoseconds, c.this)


def _from_clearing_time(value):
    if isinstance(value, dict):
        return float(value.get('seconds') or 0) + float(value.get('nanoseconds') or 0) / 1_000_000_000
    return MISSING if value is None else float(value)


def from_clearing_time():
    """A protobuf ``{seconds, nanoseconds}`` duration (OpenFMB ``ClearingTime``, ``ControlTimestamp``) as float seconds;
    the inverse of ``whole_seconds`` and ``nanoseconds`` together."""
    return c.call_func(_from_clearing_time, c.this)


def _when(value, actual, expected):
    return value if actual is not None and actual == expected else MISSING


def when(path: str, expected):
    """
        Pass the value through only when the field at ``path`` (a quoted dotted path resolved against the
         element enclosing the expression) equals ``expected``; otherwise the field is treated as missing.
    """
    return c.call_func(_when, c.this, scoped_path(path), expected)


def when_equal(path_a: str, path_b: str):
    """
        Pass the value through only when the two fields named by the quoted dotted paths (resolved against
         the element enclosing the expression) are present and equal. Used to accept an IEEE 1815.2 curve
         window only when its selector matches a mode's curve index, e.g. ``when_equal('AI.328', 'AI.297')``.
    """
    return c.call_func(_when, c.this, scoped_path(path_a), scoped_path(path_b))


def _unpairs(points, start, x, y):
    if not isinstance(points, (list, tuple)):
        return MISSING
    flat = {}
    for i, point in enumerate(points):
        if not isinstance(point, dict):
            continue
        flat[str(start + 2 * i)] = point.get(x, MISSING)
        flat[str(start + 2 * i + 1)] = point.get(y, MISSING)
    return flat


def unpairs(start: int, x: str = 'x', y: str = 'y'):
    """
        Inverse of ``pairs``: lay a list of ``{x: ..., y: ...}`` points out as a flat, position-indexed
         group of alternating X and Y values starting at index ``start`` (keys are strings). Meant for a
         spread entry (``"*points": "transform[...](unpairs(249, xvalue, yvalue))"``) so the values merge
         into the enclosing point group.
    """
    return c.call_func(_unpairs, c.this, start, x, y)


def _parse_specs(specs):
    """'field:template' strings -> [(field, template)]; the template holds one ``{}`` for the index."""
    parsed = []
    for spec in specs:
        field, _, template = str(spec).partition(':')
        if not template or '{}' not in template:
            raise ValueError(f"series specs are 'field:name_with_{{}}' strings, got {spec!r}")
        parsed.append((field, template))
    return parsed


def _series(mapping, count, start, specs):
    if not isinstance(mapping, dict):
        return MISSING
    items = []
    for i in range(start, start + count):
        item = {field: mapping.get(template.format(i), MISSING) for field, template in specs}
        if all(v is MISSING for v in item.values()):
            continue
        items.append(item)
    return items


def series(count: int, start: int, *specs: str):
    """
        Gather numbered flat points into a list of records, e.g. a driver's ``705_Crv1_Pt1_V``, ``705_Crv1_Pt1_Var``,
         ``705_Crv1_Pt2_V`` ... into ``[{V, Var}, ...]``: ``series(4, 1, 'V:705_Crv1_Pt{}_V', 'Var:705_Crv1_Pt{}_Var')``.
         ``count`` indices from ``start`` are tried; indices with none of the fields present are skipped.
    """
    parsed = _parse_specs(specs)
    conv = c.call_func(_series, c.this, count, start, parsed)
    # The concrete point names each record field comes from, so field maps can account for them exactly.
    conv.series_fields = {field: [template.format(i) for i in range(start, start + count)] for field, template in parsed}
    return conv


def _unseries(items, start, specs):
    if not isinstance(items, (list, tuple)):
        return MISSING
    flat = {}
    for i, item in enumerate(items, start):
        if isinstance(item, dict):
            for field, template in specs:
                flat[template.format(i)] = item.get(field, MISSING)
    return flat


def unseries(start: int, *specs: str):
    """
        Inverse of ``series``: lay a list of records out as numbered flat points, for a spread entry:
         ``"*705 points": "transform[705, Crv, 0, Pt](unseries(1, 'V:705_Crv1_Pt{}_V', 'Var:705_Crv1_Pt{}_Var'))"``.
    """
    return c.call_func(_unseries, c.this, start, _parse_specs(specs))


def _scale_record(item, factors):
    item = dict(item)
    for field, factor in factors:
        if isinstance(item.get(field), (int, float)) and not isinstance(item[field], bool):
            item[field] = _fix_decimals(item[field] * factor, item[field], factor)
    return item


def _scale_fields(items, factors):
    if isinstance(items, dict):
        return _scale_record(items, factors)
    if not isinstance(items, (list, tuple)):
        return MISSING
    return [_scale_record(item, factors) if isinstance(item, dict) else item for item in items]


def scale_fields(*specs: str):
    """
        Multiply named fields by constants, in every record of a list or in a single group:
         ``scale_fields('V:0.01', 'Var:0.01')``. Used with ``series``/``unseries`` when a curve's points carry a
         scale factor known at configuration time, or on a point table keyed by index. Has an inverse.
    """
    factors = []
    for spec in specs:
        field, _, factor = str(spec).partition(':')
        factors.append((field, _numeric(factor)))
    conv = c.call_func(_scale_fields, c.this, factors)
    conv.inverse = c.call_func(_scale_fields, c.this, [(f, 1 / v) for f, v in factors])
    return conv


def _prefix_keys(mapping, prefix):
    if not isinstance(mapping, dict):
        return MISSING
    return {f'{prefix}{k}': v for k, v in mapping.items()}


def _strip_prefix(mapping, prefix):
    if not isinstance(mapping, dict):
        return MISSING
    return {(k[len(prefix):] if str(k).startswith(prefix) else k): v for k, v in mapping.items()}


def prefix_keys(prefix: str):
    """
        Rename every key of a group by prefixing it, e.g. ``prefix_keys('AO_')`` turns a DNP3 point table
         ``{"244": 1}`` into the driver's flat ``{"AO_244": 1}``. Meant for a spread entry. Has an inverse.
    """
    conv = c.call_func(_prefix_keys, c.this, prefix)
    conv.inverse = c.call_func(_strip_prefix, c.this, prefix)
    return conv
