"""Assembling one message out of several arrivals, by a consumer's :class:`~icm.resources.definitions.DeliveryPolicy`.

A resource's values arrive as device publications (``[values, meta]`` or ``{point: value}``), often a few points at a
time: a ``/multi`` push carries what changed, a slow-polled group arrives on its own cycle, a composed resource has
several publication topics. The transform chain is stateless, so something has to decide when the fields it needs are
together. That is the :class:`Assembler`: it keeps the last value and time of every field it has seen, and for each
arrival (``offer``) or passage of time (``due``) says whether to emit and with what.

It is pure: time comes in as an argument (or from ``clock``), nothing sleeps, nothing spawns. ``next_due`` tells a
runner when ``due`` should next be called; ``ResourceData`` provides the runner.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .definitions import DeliveryPolicy
from .resource import message_values, same_value

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Emit:
    """One assembled message: the values to hand the chain and how each required field got there."""
    values: dict[str, Any]
    fresh: frozenset[str]          #: fields carried by the arrival that caused the emit
    merged: frozenset[str]         #: required fields filled from earlier arrivals
    missing: frozenset[str]        #: required fields with no usable value
    when: float
    reason: str                    #: ``arrival``, ``complete``, ``max_wait``, ``interval`` or ``trigger``

    def as_header(self) -> dict[str, Any]:
        """The emit's bookkeeping as a JSON-friendly dict for a message header."""
        return {'reason': self.reason, 'fresh': sorted(self.fresh), 'merged': sorted(self.merged), 'missing': sorted(self.missing)}


@dataclass
class _Known:
    value: Any
    when: float


class Assembler:
    """State and policy between a subscription and a transform chain. ``required`` is the set of fields the chain
    reads (from the parser's field map); it may be empty, in which case every policy works on the fields that come."""

    def __init__(self, policy: DeliveryPolicy | None = None, required: Iterable[str] = (), *, clock: Callable[[], float] = time.monotonic):
        self.policy = policy or DeliveryPolicy()
        self.required = frozenset(str(f) for f in required)
        self.clock = clock
        self._known: dict[str, _Known] = {}
        self._arrived_since_emit: set[str] = set()
        self._first_arrival: float | None = None
        self._last_emit: float | None = clock() if self.policy.kind == 'interval' else None

    # ---- arrivals and time -------------------------------------------------------------------------------------
    def offer(self, message, when: float | None = None) -> Emit | None:
        """Take an arrival. Returns an emit when the policy says so, else None."""
        now = self.clock() if when is None else when
        points = message_values(message)
        if points is None:
            points = {'value': message}
        keys = frozenset(points)
        changed = {k for k, v in points.items() if k not in self._known or not same_value(self._known[k].value, v)}
        for k, v in points.items():
            self._known[k] = _Known(v, now)
        self._arrived_since_emit |= keys
        if self._first_arrival is None:
            self._first_arrival = now
        kind = self.policy.kind
        if kind == 'each':
            return self._emit(points, keys, now, 'arrival', merge=False)
        if kind == 'complete':
            if self.required <= self._arrived_since_emit:
                return self._emit(points, keys, now, 'complete', merge=True)
            return None
        if kind == 'merge':
            return self._emit(points, keys, now, 'arrival', merge=True)
        if kind == 'trigger':
            if changed & set(self.policy.fields):
                return self._emit(points, keys, now, 'trigger', merge=True)
            return None
        return None                         # interval: arrivals only feed the state

    def due(self, when: float | None = None) -> Emit | None:
        """Take the passage of time: a ``complete`` policy's ``max_wait`` running out, or an ``interval`` elapsing."""
        now = self.clock() if when is None else when
        kind = self.policy.kind
        if kind == 'complete' and self.policy.max_wait and self._first_arrival is not None \
                and now - self._first_arrival >= self.policy.max_wait and self._arrived_since_emit:
            return self._emit({}, frozenset(self._arrived_since_emit), now, 'max_wait', merge=True)
        if kind == 'interval' and (self._last_emit is None or now - self._last_emit >= self.policy.interval):
            known = self.known(now, every=True)
            fresh = frozenset(self._arrived_since_emit)
            emit = Emit(known, fresh, frozenset(k for k in known if k not in fresh), self.required - set(known), now, 'interval')
            self._after_emit(now)
            return emit
        return None

    def next_due(self, when: float | None = None) -> float | None:
        """When ``due`` should next be called, or None when the policy needs no clock."""
        now = self.clock() if when is None else when
        kind = self.policy.kind
        if kind == 'complete' and self.policy.max_wait and self._first_arrival is not None:
            return self._first_arrival + self.policy.max_wait
        if kind == 'interval':
            return (self._last_emit if self._last_emit is not None else now) + self.policy.interval
        return None

    # ---- state ------------------------------------------------------------------------------------------------------
    def known(self, when: float | None = None, *, every: bool = False) -> dict[str, Any]:
        """The last values that are not stale: of the required fields, or of every field seen with ``every``."""
        now = self.clock() if when is None else when
        stale_after = self.policy.stale_after
        return {k: known.value for k, known in self._known.items()
                if (every or k in self.required) and (stale_after is None or now - known.when <= stale_after)}

    def _emit(self, points: dict, keys: frozenset, now: float, reason: str, *, merge: bool) -> Emit:
        if merge:
            values = {**self.known(now), **points}
            merged = frozenset(k for k in values if k not in keys and k in self.required)
        else:
            values, merged = dict(points), frozenset()
        missing = self.required - set(values)
        self._after_emit(now)
        return Emit(values, keys, merged, missing, now, reason)

    def _after_emit(self, now: float) -> None:
        self._arrived_since_emit.clear()
        self._first_arrival = None
        self._last_emit = now


class Ticker:
    """Calls ``assembler.due()`` when its ``next_due`` comes, on a gevent greenlet when gevent is present (the agents
    run on it) and on a daemon thread otherwise; emits go to ``on_due``."""

    def __init__(self, assembler, on_due: Callable):
        self.assembler, self.on_due, self._stopped = assembler, on_due, False
        try:
            import gevent
            self._sleep, self._spawn = gevent.sleep, gevent.spawn
        except ImportError:                                         # pragma: no cover - the agents always have gevent
            import threading
            self._sleep = time.sleep
            self._spawn = lambda fn: threading.Thread(target=fn, daemon=True).start()

    def start(self):
        self._spawn(self._run)

    def cancel(self):
        self._stopped = True

    def _run(self):
        while not self._stopped:
            due = self.assembler.next_due()
            if due is None:
                self._stopped = True
                break
            delay = due - self.assembler.clock()
            if delay > 0:
                self._sleep(min(delay, 1.0))
                continue
            emit = self.assembler.due()
            if emit is not None:
                try:
                    self.on_due(emit)
                except Exception as e:                              # a consumer's failure must not stop the clock
                    _log.warning(f'assembled delivery failed: {e!r}')
            else:
                self._sleep(0.05)

