# icm-lib-transforms

Transform expressions for interoperability: a library of functions, a grammar for patterns that map one message
format onto another, compiled pipelines that run them, and transforms of single values with their inverses.

## Patterns

```python
from icm.transforms import TransformParser

parser = TransformParser()
chain = parser.build_transform_from_schema({'kW': 'transform[W](scale(0.001))'})
chain.execute({'W': 2500})        # {'kW': 2.5}
```

A pattern is a dict whose keys are output fields and whose values are expressions:

```
transform[<source path>](<function>(<args>), ...)
```

The bracketed path locates the value in the input message (omitted, the output key is used); the functions are
applied to it in order. Nested dicts produce nested output, a key ending in `[#]` repeats a group over a source list,
a root `"#"` entry rebinds the stage input, and `param('name')` reaches values bound when the chain is compiled. The
module docstring of `icm.transforms.parser` is the full reference. `TransformParser.field_map()` reports which source
fields a pattern reads and how much of each survives, which lets a caller choose between chains by how many of a
message's fields they retain.

## Value transforms

A value transform is what a driver applies to one point: one function call, or several wrapped in `transform(...)`.

```python
from icm.transforms import compile_value_transform

t = compile_value_transform('transform(scale(0.1), add(-40))')
t.forward(500)                     # 10.0
t.inverse(10.0)                    # 500.0
t.functions, t.result_type         # ('scale', 'add'), float

sf = compile_value_transform('sunssf(W_SF)')
sf.references                      # frozenset({'W_SF'})
sf.forward(1234, {'W_SF': -1})     # 123.4
sf.forward(1234, {})               # MISSING: the factor is not known, so no value is produced
```

The inverse exists when every function in the chain has one (`invertible`, `non_invertible`). Functions that read
other registers (`scale_reg`, `sunssf`, `sum_of` ...) take them from the mapping passed with each value, and
`references` names them so a caller can check they exist before the first use. What an absent register means is the
caller's choice at compile time: `missing_reference='hold'` (the default) yields `MISSING`, `'pass'` leaves the value
unchanged, which is the right reading inside a resource pattern. A bare comma-separated list is rejected; a CSV cell
holding one is too easy to misread.

## Functions

| Group | Functions |
|---|---|
| Arithmetic | `scale`, `scale_int`, `scale_decimal_int_signed`, `multiple`, `add`, `subtract`, `divide`, `negate`, `absolute`, `power`, `root`, `log`, `modulo`, `no_op`, `mod10k`, `mod10k64`, `mod10k48` |
| Rounding | `round`, `floor`, `ceil`, `clamp` |
| With another register | `scale_reg`, `scale_reg_pow_10`, `sunssf`, `add_reg`, `subtract_reg`, `multiply_reg`, `divide_reg` |
| Over several registers | `value_of`, `sum_of`, `mean_of`, `min_of`, `max_of`, `all_equal_of`, `all_of`, `any_of`; arguments `literal(...)`, `reg_name(...)` |
| Comparison | `equals` (`eq`), `not_equals` (`ne`), `less_than` (`lt`), `less_or_equal` (`le`), `greater_than` (`gt`), `greater_or_equal` (`ge`), `in_range`, `select`, `map` |
| Bits and booleans | `bit`, `bits`, `mask`, `shift`, `invert`, `bool`, `not` |
| Type and value | `cast_value`, `const`, `param`, `uuid4`, `timestamp`, `approx`, `invalid` (device sentinels such as `-32768` or `0xFFFF` yield no value), `finite` (NaN and infinities yield no value) |
| Time | `whole_seconds`, `nanoseconds`, `from_clearing_time` |
| Lists and structure | `take`, `mean`, `count`, `when`, `when_equal`, `device_values`, `pairs`, `unpairs`, `series`, `unseries`, `as_list`, `immediate`, `scale_fields`, `prefix_keys` |

Each function returns a convtools conversion and, where it has one, carries its `inverse`. `sunssf` is the SunSpec
scale factor: a signed base-10 exponent whose value -32768 means "not implemented" and counts as absent. The `_of`
functions define a value from other registers and ignore the point's own; numbers among their arguments are literals,
words are register names.

## Adding functions

Functions resolve through `icm.transforms.registry`. A package adds its own by declaring an entry point in the
`icm.transforms.functions` group whose object is a factory (registered under the entry point's name) or a mapping of
names to factories:

```toml
[tool.poetry.plugins."icm.transforms.functions"]
bacnet_enum = "my_driver.transforms:bacnet_enum"
```

or in code with `icm.transforms.register(name, factory, aliases=..., fidelity=..., result_type=...)`. A factory is
called with the literal arguments of the expression and returns a conversion; `inverse`, `references`, `result_type`
and `fidelity` may be set as attributes on it. A name registered twice with different factories is an error, so the
same registry file means the same thing on every platform.

Dependencies: convtools and pyparsing.
