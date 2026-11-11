"""What an agent may declare about a resource in its own configuration, and how it wants messages delivered.

A resource entry in an agent's configuration either names a UAI for the presentation service to resolve
(``ResourceData.lookup``) or carries a full definition of its own (``ResourceData.from_definition``). The definitions
here are the second kind: a :class:`LocalCanonical` says where a device publishes and in which format, a
:class:`LocalAlias` adds the chain that turns those publications into another format and the codec they leave in. Both
compile into the same dict shape ``resolve`` returns, so everything after construction is identical.

A :class:`DeliveryPolicy` says when a consumer wants a message out of several arrivals (see ``assembler``).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CodecSpec(BaseModel):
    """The wire encoding a resource's messages leave in: JSON (the default) or protobuf with the message name the
    format declares and, when the codec depends on it, the standard's edition (an OpenFMB release)."""
    model_config = ConfigDict(extra='forbid')
    encoding: Literal['json', 'protobuf'] = 'json'
    proto: str | None = None
    edition: str | None = None

    @model_validator(mode='after')
    def _protobuf_needs_a_message(self):
        if self.encoding == 'protobuf' and not self.proto:
            raise ValueError('a protobuf codec needs the protobuf message name (proto).')
        return self

    def as_resolved(self) -> dict | None:
        """The ``codec`` dict ``resolve`` would return; None for plain JSON without a message name."""
        if self.encoding == 'json' and not self.proto:
            return None
        return {k: v for k, v in (('encoding', self.encoding), ('proto', self.proto), ('edition', self.edition)) if v is not None}


class LocalCanonical(BaseModel):
    """A canonical resource declared by the agent that uses it: a device's publications in the device's own format.
    The chain is the identity; only subscription, assembly and the codec are involved."""
    model_config = ConfigDict(extra='forbid')
    kind: Literal['canonical'] = 'canonical'
    uai: list[str]
    data_format: str
    publication_topic: str | None = None
    #: Further topics the same resource arrives on (composition: a resource assembled from several publications).
    publication_topics: list[str] = Field(default_factory=list)
    rpc_topic: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    codec: CodecSpec = Field(default_factory=CodecSpec)

    def as_resolved(self) -> dict:
        topics = [t for t in [self.publication_topic, *self.publication_topics] if t]
        resolved = {'data_format': self.data_format, 'owner': 'local', 'publication_topic': topics[0] if topics else None,
                    'publication_topics': topics, 'rpc_topic': self.rpc_topic, 'parameters': dict(self.parameters),
                    'encoding': self.codec.encoding, 'edition': self.codec.edition, 'canonical_uai': list(self.uai),
                    'alias_uais': [], 'transform': [], 'path': [self.data_format], 'retention': 1.0}
        if (codec := self.codec.as_resolved()) is not None:
            resolved['codec'] = codec
        return resolved


class LocalAlias(BaseModel):
    """An alias declared by the agent that uses it: a canonical resource's publications presented in another format
    through a stated chain. The chain is the stage list the service's mapping files hold and ``resolve`` returns; it
    is run as given (no path finding, no scoring, no per-field-set selection; agents that need those stay on
    ``lookup``). ``write_transform`` is the chain for writing the alias's format back to the device, when the alias is
    written through; it is not derived from ``transform``."""
    model_config = ConfigDict(extra='forbid')
    kind: Literal['alias'] = 'alias'
    uai: list[str]
    canonical_uai: list[str] = Field(default_factory=list)
    data_format: str                     #: the canonical's (source) format
    target_format: str                   #: the format this alias presents
    publication_topic: str | None = None
    publication_topics: list[str] = Field(default_factory=list)
    rpc_topic: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    codec: CodecSpec = Field(default_factory=CodecSpec)
    transform: list[dict[str, Any]] = Field(default_factory=list)
    write_transform: list[dict[str, Any]] | None = None

    @model_validator(mode='after')
    def _a_source_is_needed(self):
        if not self.publication_topic and not self.publication_topics:
            raise ValueError('an alias needs the publication topic(s) of the canonical resource it presents.')
        return self

    def as_resolved(self) -> dict:
        topics = [t for t in [self.publication_topic, *self.publication_topics] if t]
        resolved = {'data_format': self.data_format, 'owner': 'local', 'publication_topic': topics[0], 'publication_topics': topics,
                    'rpc_topic': self.rpc_topic, 'parameters': dict(self.parameters), 'encoding': self.codec.encoding,
                    'edition': self.codec.edition, 'canonical_uai': list(self.canonical_uai), 'alias_uais': [list(self.uai)],
                    'target_format': self.target_format, 'transform': list(self.transform),
                    'path': [self.data_format, self.target_format] if self.transform else [self.data_format], 'retention': 1.0}
        if self.write_transform is not None:
            resolved['write_transform'] = list(self.write_transform)
        if (codec := self.codec.as_resolved()) is not None:
            resolved['codec'] = codec
        return resolved


LocalDefinition = LocalCanonical | LocalAlias


def parse_definition(data: dict | LocalCanonical | LocalAlias) -> LocalCanonical | LocalAlias:
    """A definition from configuration: ``kind`` (``canonical`` or ``alias``) picks the model."""
    if isinstance(data, (LocalCanonical, LocalAlias)):
        return data
    kind = str(data.get('kind') or ('alias' if data.get('target_format') or data.get('transform') else 'canonical')).lower()
    if kind in ('canonical', 'canon'):
        return LocalCanonical(**{**data, 'kind': 'canonical'})
    if kind in ('alias', 'aliased'):
        return LocalAlias(**{**data, 'kind': 'alias'})
    raise ValueError(f'a resource definition is canonical or alias, not {kind!r}.')


class DeliveryPolicy(BaseModel):
    """When a consumer wants a message out of the arrivals on a resource's topics (see ``assembler``):

    - ``each``: every arrival as it comes, with the fields it carries (the default).
    - ``complete``: once every required field has arrived since the last emit; after ``max_wait`` seconds from the
      first arrival, with what is present.
    - ``merge``: every arrival, merged over the last known values of the other required fields; values older than
      ``stale_after`` seconds are not merged.
    - ``interval``: every ``interval`` seconds from the last known values, whether or not anything arrived.
    - ``trigger``: when one of ``fields`` changes value, merged like ``merge``.
    """
    model_config = ConfigDict(extra='forbid')
    kind: Literal['each', 'complete', 'merge', 'interval', 'trigger'] = 'each'
    max_wait: float | None = Field(default=None, gt=0)
    stale_after: float | None = Field(default=None, gt=0)
    interval: float | None = Field(default=None, gt=0)
    fields: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def _options_match_the_kind(self):
        if self.kind == 'interval' and not self.interval:
            raise ValueError("an 'interval' policy needs interval (seconds).")
        if self.kind == 'trigger' and not self.fields:
            raise ValueError("a 'trigger' policy needs the fields whose change triggers an emit.")
        if self.max_wait is not None and self.kind != 'complete':
            raise ValueError("max_wait applies to the 'complete' policy only.")
        if self.interval is not None and self.kind != 'interval':
            raise ValueError("interval applies to the 'interval' policy only.")
        return self
