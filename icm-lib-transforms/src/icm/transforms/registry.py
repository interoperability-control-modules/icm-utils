"""Where transform expressions find their functions.

Every function an expression may call is registered here by name: the library's own functions at import, functions
from other installed packages through the ``icm.transforms.functions`` entry-point group, and anything an application
adds with :func:`register`. Each entry carries what the expression language needs to know about the function besides
how to call it: how much of a value's information survives it (``fidelity``, 1.0 exact, 0.0 gone) and the Python type
it yields when that differs from its input (``result_type``). Both may instead be set as attributes on the conversion
a factory returns, which wins over the entry.

A package adds functions by declaring an entry point whose object is either a factory (registered under the entry
point's name) or a mapping of names to factories::

    [tool.poetry.plugins."icm.transforms.functions"]
    bacnet_enum = "my_driver.transforms:bacnet_enum"

A name registered twice with different factories is an error: the same registry file must mean the same thing on
every platform.
"""
from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable, Iterable, Mapping

from . import functions as _functions

_log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = 'icm.transforms.functions'
#: Fidelity assumed for a function that declares none and has no inverse.
DEFAULT_LOSSY = 0.9


class RegistrationError(ValueError):
    """A name is already registered with a different factory."""


@dataclass(frozen=True)
class Entry:
    name: str
    factory: Callable[..., Any]
    fidelity: float | None = None          #: None: 1.0 when the conversion has an inverse, else DEFAULT_LOSSY
    result_type: type | None = None        #: the type the function yields; None: the input's type
    aliases: tuple[str, ...] = ()
    package: str = 'icm.transforms'


_REGISTRY: dict[str, Entry] = {}
_entry_points_loaded = False


def register(name: str, factory: Callable[..., Any], *, aliases: Iterable[str] = (), fidelity: float | None = None,
             result_type: type | None = None, package: str | None = None) -> Entry:
    """Register ``factory`` as the function ``name`` (and each alias). Registering the same factory again is a no-op;
    a different factory under a taken name raises :class:`RegistrationError`."""
    if not callable(factory):
        raise TypeError(f'{name}: a transform function factory must be callable, got {type(factory).__name__}.')
    fidelity = fidelity if fidelity is not None else getattr(factory, 'fidelity', None)
    result_type = result_type if result_type is not None else getattr(factory, 'result_type', None)
    aliases = tuple(aliases) or tuple(getattr(factory, 'aliases', ()))
    entry = Entry(name, factory, None if fidelity is None else float(fidelity), result_type, aliases,
                  package or getattr(factory, '__module__', '?').split('.')[0])
    for alias in (name, *aliases):
        taken = _REGISTRY.get(alias)
        if taken is not None and taken.factory is not factory:
            raise RegistrationError(f'transform function {alias!r} is already registered by {taken.package} '
                                    f'({taken.factory.__module__}.{taken.factory.__qualname__}); '
                                    f'{entry.package} cannot register it as well.')
        _REGISTRY[alias] = entry
    return entry


def get(name: str) -> Entry | None:
    """The entry for ``name`` (or an alias), loading entry points on first use. None when unknown."""
    load_entry_points()
    return _REGISTRY.get(name)


def names(*, aliases: bool = True) -> list[str]:
    """Every registered name, sorted; without ``aliases`` only each function's primary name."""
    load_entry_points()
    return sorted(n for n, e in _REGISTRY.items() if aliases or n == e.name)


def entries() -> list[Entry]:
    """Every registered function once, sorted by name."""
    load_entry_points()
    return sorted({e.name: e for e in _REGISTRY.values()}.values(), key=lambda e: e.name)


def fidelity_of(conv, name: str) -> float:
    """How much of a value's information survives the conversion ``conv`` built by ``name``: the conversion's own
    ``fidelity`` attribute, else the entry's, else 1.0 when it has an inverse and DEFAULT_LOSSY otherwise."""
    explicit = getattr(conv, 'fidelity', None)
    if explicit is not None:
        return float(explicit)
    entry = _REGISTRY.get(name)
    if entry is not None and entry.fidelity is not None:
        return entry.fidelity
    return 1.0 if getattr(conv, 'inverse', None) is not None else DEFAULT_LOSSY


def result_type_of(conv, name: str) -> type | None:
    """The type ``conv`` yields: the conversion's ``result_type`` attribute, else the entry's, else None (unchanged)."""
    explicit = getattr(conv, 'result_type', None)
    if explicit is not None:
        return explicit
    entry = _REGISTRY.get(name)
    return entry.result_type if entry is not None else None


def load_entry_points(force: bool = False) -> None:
    """Register the functions other installed packages declare in the ``icm.transforms.functions`` group."""
    global _entry_points_loaded
    if _entry_points_loaded and not force:
        return
    _entry_points_loaded = True
    for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
        try:
            obj = ep.load()
        except Exception as e:                                   # a broken plugin must not take the registry down
            _log.warning(f'transform functions entry point {ep.name} ({ep.value}) failed to load: {e!r}')
            continue
        package = ep.dist.name if ep.dist is not None else ep.value.split(':')[0].split('.')[0]
        if isinstance(obj, Mapping):
            for name, factory in obj.items():
                register(name, factory, package=package)
        else:
            register(ep.name, obj, package=package)


# ---- The library's own functions ----------------------------------------------------------------------------------

#: What the expression language knows about each built-in beyond its factory: (fidelity, result type, aliases).
#: Fidelity: how much of a value's information survives (1.0 exact, 0.0 gone). Functions absent from this table count
#: as exact when they define an ``inverse`` and DEFAULT_LOSSY otherwise.
_BUILTIN_METADATA: dict[str, tuple[float | None, type | None, tuple[str, ...]]] = {
    # arithmetic and scaling
    'scale': (1.0, float, ()), 'scale_int': (0.9, int, ()), 'scale_decimal_int_signed': (1.0, float, ()),
    'multiple': (1.0, None, ()), 'add': (1.0, None, ()), 'subtract': (1.0, None, ()), 'divide': (1.0, float, ()),
    'negate': (1.0, None, ()), 'absolute': (0.9, None, ()), 'power': (1.0, None, ()), 'root': (1.0, float, ()),
    'log': (1.0, float, ()), 'modulo': (0.5, None, ()), 'no_op': (1.0, None, ()),
    'invalid': (1.0, None, ()), 'finite': (1.0, None, ()),
    'scale_reg': (1.0, float, ()), 'scale_reg_pow_10': (1.0, float, ()), 'sunssf': (1.0, float, ()),
    'add_reg': (1.0, None, ()), 'subtract_reg': (1.0, None, ()), 'multiply_reg': (1.0, None, ()), 'divide_reg': (1.0, float, ()),
    'mod10k': (1.0, int, ()), 'mod10k64': (1.0, int, ()), 'mod10k48': (1.0, int, ()),
    # rounding
    'round': (0.9, None, ()), 'floor': (0.9, int, ()), 'ceil': (0.9, int, ()), 'clamp': (0.9, None, ()),
    # bits and booleans
    'bit': (0.3, bool, ()), 'bits': (0.9, int, ()), 'mask': (0.9, int, ()), 'shift': (1.0, int, ()), 'invert': (1.0, int, ()),
    'bool': (0.9, bool, ()), 'not': (1.0, bool, ()),
    # comparison
    'equals': (0.9, bool, ('eq',)), 'not_equals': (0.9, bool, ('ne',)), 'less_than': (0.9, bool, ('lt',)),
    'less_or_equal': (0.9, bool, ('le',)), 'greater_than': (0.9, bool, ('gt',)), 'greater_or_equal': (0.9, bool, ('ge',)),
    'in_range': (0.9, bool, ()), 'select': (1.0, None, ()), 'map': (1.0, None, ()),
    # over several registers (the point's own value is not carried)
    'value_of': (0.0, None, ()), 'sum_of': (0.0, None, ()), 'mean_of': (0.0, float, ()), 'min_of': (0.0, None, ()), 'max_of': (0.0, None, ()),
    'all_equal_of': (0.0, bool, ()), 'all_of': (0.0, bool, ()), 'any_of': (0.0, bool, ()),
    # type and value
    'cast_value': (0.9, None, ()), 'const': (0.0, None, ()), 'param': (0.0, None, ()), 'uuid4': (0.0, str, ()),
    'timestamp': (0.0, None, ()), 'literal': (None, None, ()), 'reg_name': (None, None, ()),
    # time
    'whole_seconds': (0.9, int, ()), 'nanoseconds': (0.9, int, ()), 'from_clearing_time': (1.0, float, ()),
    # lists and structure
    'take': (0.9, None, ()), 'mean': (0.5, float, ()), 'count': (0.3, int, ()), 'when': (0.7, None, ()),
    'when_equal': (0.7, None, ()), 'device_values': (1.0, None, ()), 'pairs': (1.0, list, ()), 'unpairs': (1.0, dict, ()),
    'series': (1.0, list, ()), 'unseries': (1.0, dict, ()), 'as_list': (1.0, list, ()), 'immediate': (1.0, None, ()),
    'scale_fields': (1.0, None, ()), 'prefix_keys': (1.0, None, ()), 'approx': (None, None, ()),
}

#: Public callables of the functions module that are helpers for other functions, not functions of the language.
_NOT_FUNCTIONS = {'scope', 'scoped_path'}


def _register_builtins() -> None:
    for name, obj in vars(_functions).items():
        if name.startswith('_') or name in _NOT_FUNCTIONS or not callable(obj) or isinstance(obj, type):
            continue
        if inspect.getmodule(obj) is not _functions:
            continue
        fidelity, result_type, aliases = _BUILTIN_METADATA.get(name, (None, None, ()))
        register(name, obj, aliases=aliases, fidelity=fidelity, result_type=result_type, package='icm.transforms')
    # Names that are Python keywords or builtins are defined under another name and registered under their own.
    for name, obj in _functions.RESERVED_NAMES.items():
        fidelity, result_type, aliases = _BUILTIN_METADATA.get(name, (None, None, ()))
        register(name, obj, aliases=aliases, fidelity=fidelity, result_type=result_type, package='icm.transforms')
    missing = sorted(set(_BUILTIN_METADATA) - set(_REGISTRY))
    if missing:
        raise RuntimeError(f'metadata for unknown built-in transform functions: {missing}')


_register_builtins()
