"""Transforms of one value: what a driver applies to a point, as opposed to a pattern over a whole message.

A value transform is one function call (``scale(0.001)``) or a sequence wrapped in ``transform(...)``
(``transform(scale(0.1), add(-40))``), in the same grammar as a pattern's expressions but without a source path: the
value itself is the input. :func:`compile_value_transform` turns the text into a :class:`ValueTransform` with a
``forward`` callable, an ``inverse`` callable when every function in the chain has one, the names of the other
registers the chain reads (``references``), the type it yields and its fidelity.

Functions that read other registers (``scale_reg``, ``sunssf``, ``sum_of`` ...) resolve them in the ``values`` mapping
the caller passes with each value, by plain name: ``transform.forward(raw, values={'W_SF': -1})``. What happens when a
register is not in the mapping is the caller's choice at compile time: ``missing_reference='hold'`` (the default here)
yields :data:`MISSING`, so the caller publishes no value; ``'pass'`` leaves the value unchanged, which is the right
reading inside a resource pattern, where an absent factor usually means it was already applied.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from convtools import conversion as c
from pyparsing import ParseException

from . import functions, registry
from .functions import MISSING, ROOT_LABEL
from .parser import TransformParser, _Expr

#: A hint for the error on a bare comma-separated list, which a CSV cell makes too easy to misread.
_WRAP_HINT = 'several functions must be wrapped: transform(f1(...), f2(...))'


class NotInvertible(ValueError):
    """The transform has a function without an inverse, so it cannot be applied backwards (a write)."""


@dataclass(frozen=True)
class ValueTransform:
    """A compiled transform of one value. ``forward`` and ``inverse`` take the value and, for functions that read other
    registers, a mapping of register names to their current values; both yield :data:`MISSING` when the result is
    undefined (an input of None, a held-out missing reference, a function that produced nothing)."""
    text: str
    functions: tuple[str, ...]            #: the function names in order, primary names (``eq`` reports as ``equals``)
    references: frozenset[str]            #: other registers the chain reads
    result_type: type | None              #: the type the chain yields; None: the input's type
    fidelity: float                       #: how much of the value's information survives, 1.0 exact
    missing_reference: str                #: ``'hold'`` or ``'pass'``
    non_invertible: tuple[str, ...]       #: the functions in the chain that have no inverse, in order
    _forward: Callable[[dict], Any]
    _inverse: Callable[[dict], Any] | None
    defines_value: bool = False           #: the chain starts with a function over several registers (``sum_of`` ...):
                                          #: it defines the value from them and the point's own value is not an input

    @property
    def invertible(self) -> bool:
        return self._inverse is not None

    @property
    def identity(self) -> bool:
        return not self.functions

    def forward(self, value, values: Mapping | None = None):
        """The engineering value for raw ``value``."""
        return self._forward({'value': value, 'values': values if values is not None else {}})

    def inverse(self, value, values: Mapping | None = None):
        """The raw value for engineering ``value``; raises :class:`NotInvertible` when a function has no inverse."""
        if self._inverse is None:
            raise NotInvertible(f'{self.text!r} cannot be inverted: {self.non_invertible[0]!r} has no inverse.')
        return self._inverse({'value': value, 'values': values if values is not None else {}})


def compile_value_transform(text: str, *, missing_reference: str = 'hold', context: Mapping | None = None) -> ValueTransform:
    """Compile ``text`` (one function call, or ``transform(...)`` around several; empty for the identity).

    ``missing_reference`` is what a function reading an absent register does: ``'hold'`` yields MISSING, ``'pass'``
    leaves the value unchanged. ``context`` binds ``param('name')`` values. Results are cached by text and options."""
    if missing_reference not in ('hold', 'pass'):
        raise ValueError(f"missing_reference must be 'hold' or 'pass', not {missing_reference!r}.")
    token = functions.missing_reference.set(missing_reference)      # read by the register functions as they are built
    try:
        return _compile(text.strip(), missing_reference, tuple(sorted((context or {}).items())))
    finally:
        functions.missing_reference.reset(token)


@lru_cache(maxsize=4096)
def _compile(text: str, missing_reference: str, context_items: tuple) -> ValueTransform:
    parser = TransformParser(context=dict(context_items))
    expr = _parse(parser, text)
    convs = list(expr.convs)
    for conv in convs:
        if conv is None:
            raise ValueError(f'{text!r}: unknown transform function (known: {", ".join(registry.names(aliases=False))}).')
        if isinstance(conv, (functions._Literal, functions._RegName)):
            raise ValueError(f'{text!r}: {conv!r} marks an argument of a function over several registers; it is not a function.')
    names = tuple(getattr(conv, 'transform_call', ('?',))[0] for conv in convs)
    references = frozenset().union(*(getattr(conv, 'references', frozenset()) for conv in convs))
    result_type = next((t for t in reversed([registry.result_type_of(conv, n) for conv, n in zip(convs, names)]) if t), None)
    fidelity = 1.0
    for conv, name in zip(convs, names):
        fidelity *= registry.fidelity_of(conv, name)
    inverses = [getattr(conv, 'inverse', None) for conv in convs]
    non_invertible = tuple(name for name, inv in zip(names, inverses) if inv is None)
    forward = _pipeline(convs)
    inverse = _pipeline(list(reversed(inverses))) if not non_invertible else None
    defines_value = bool(convs) and bool(getattr(convs[0], 'defines_value', False))
    return ValueTransform(text, names, references, result_type, fidelity, missing_reference, non_invertible, forward, inverse,
                          defines_value=defines_value)


def _parse(parser: TransformParser, text: str) -> _Expr:
    """One function call or ``transform(...)``; empty text is the identity."""
    if not text:
        return _Expr((), [], ('value',), 0)
    try:
        expr = parser._parse_expression(text, ('value',), 0, default_keys=())
    except ParseException as e:
        hint = f' ({_WRAP_HINT})' if _looks_like_bare_list(text) else ''
        raise ValueError(f'{text!r} is not a value transform: {e.msg} at column {e.column}{hint}.') from None
    if expr.segments:
        raise ValueError(f'{text!r}: a value transform takes no source path; the value itself is the input.')
    return expr


def _looks_like_bare_list(text: str) -> bool:
    depth = 0
    for ch in text:
        depth += (ch == '(') - (ch == ')')
        if ch == ',' and depth == 0:
            return True
    return False


def _pipeline(convs: list) -> Callable[[dict], Any]:
    """``{'value': v, 'values': mapping}`` -> the chain applied to ``v``, with the mapping bound as the element the
    register references resolve against. A None input or a MISSING intermediate stops the chain."""
    source = c.item('value')
    if convs and getattr(convs[0], 'defines_value', False):
        # A function over several registers defines the value from them; the point's own value, None included, is
        # not an input it waits for.
        applied = TransformParser._apply_functions(source.pipe(convs[0]), convs[1:])
    else:
        applied = TransformParser._apply_functions(source, convs) if convs else source
    # A function that produced None (a division by zero, a lookup that found nothing) produced no value.
    undefined_to_missing = c.if_(c.this.is_(None), c.naive(MISSING), c.this) if convs else c.this
    return c.this.pipe(applied.pipe(undefined_to_missing), label_input={ROOT_LABEL: c.item('values')}).gen_converter()
