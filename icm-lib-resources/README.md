# icm-lib-resources

Resources for interoperability agents: the models a resource is declared with, the clients that read and write one,
and the codecs that turn a transform's output into a wire payload.

- `icm.resources.resource`: `Resource`, `CanonicalResource` and `AliasedResource` (pydantic); `ResourceData`, which
  subscribes to a resource's publications and hands each one over transformed and encoded; `ResourceWriter`, which
  tracks what a target device is known to hold so a writer sends only what changed.
- `icm.resources.codecs`: `encoder_for` and `decoder_for` a codec declaration (`json`, or `protobuf` with an OpenFMB
  message name and edition).
- `icm.resources.openfmb.bus`: the headers OpenFMB messages travel with on the platform bus (content type, protobuf
  message name, origin, remote).
- `icm.resources.openfmb.descriptors`: the OpenFMB protobuf descriptors of each edition (1.0, 2.0, 2.1, 2.2), each in
  its own descriptor pool, so participants on different releases can be served from one process.

```python
from icm.resources import ResourceData

resource = ResourceData.lookup(agent, 'site/inverter/ess')     # resolved through the presentation service
resource.subscribe(lambda peer, sender, bus, topic, headers, message: ...)
```

A resource may also come from the agent's own configuration, with no service involved (`icm.resources.definitions`):
a `LocalCanonical` names where a device publishes and in which format, a `LocalAlias` adds the chain (the same stage
list the service's mapping files hold) and the codec. `ResourceData.from_definition(agent, topic, definition)` builds the
same object `lookup` does; `origin` says which constructor made it, and `reload()` re-resolves or re-reads and recompiles,
moving an active subscription when the publication topics change.

`subscribe` takes an optional `DeliveryPolicy` saying when a message is wanted out of several arrivals: `each` (every
arrival, the default), `complete` (once every field the chain reads has arrived, or after `max_wait`), `merge` (every
arrival merged over the last known values, `stale_after` dropping old ones), `interval` (every N seconds from the last
known values) or `trigger` (when a named field changes). The `Assembler` behind it is pure and clock-driven, and each
delivery's headers carry what was fresh, merged or missing (`icm-assembly`) and the definition's origin (`icm-origin`).

## Extras

- `openfmb`: the protobuf runtime, for encoding and decoding OpenFMB profiles.
- `volttron`: `volttron-core`, for `ResourceData` and `ResourceWriter`, which speak VIP pubsub and RPC. The models,
  codecs and headers need no platform.

Depends on `icm-lib-transforms` and pydantic.
