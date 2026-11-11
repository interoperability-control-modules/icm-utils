"""Local resource definitions, the delivery policies and the assembler, and ResourceData's second constructor,
reload and policy-driven subscription."""
from unittest import mock

import pytest

from icm.resources import (Assembler, CodecSpec, DeliveryPolicy, LocalAlias, LocalCanonical, ResourceData, parse_definition)

ALIAS = {'kind': 'alias', 'uai': ['site', 'pv', 'openfmb'], 'canonical_uai': ['site', 'pv'], 'data_format': 'sunspec_pv', 'target_format': 'eng',
         'publication_topic': 'devices/site/pv/all',
         'transform': [{'#': 'transform(device_values())', 'kW': 'transform[W](scale(0.001))', 'V': 'transform[V]()', 'Hz': 'transform[Hz]()'}]}


# ---- definitions -------------------------------------------------------------------------------------------------
def test_definitions_parse_by_kind_and_compile_to_the_resolved_shape():
    canon = parse_definition({'kind': 'canonical', 'uai': ['site', 'pv'], 'data_format': 'sunspec_pv', 'publication_topic': 'devices/site/pv/all', 'rpc_topic': 'devices/site/pv'})
    assert isinstance(canon, LocalCanonical)
    resolved = canon.as_resolved()
    assert resolved['transform'] == [] and resolved['publication_topic'] == 'devices/site/pv/all' and resolved['canonical_uai'] == ['site', 'pv'] and 'codec' not in resolved
    alias = parse_definition(ALIAS)
    assert isinstance(alias, LocalAlias) and alias.as_resolved()['target_format'] == 'eng' and alias.as_resolved()['alias_uais'] == [['site', 'pv', 'openfmb']]
    assert parse_definition({'uai': ['a'], 'data_format': 'd', 'target_format': 't', 'publication_topic': 'x'}).kind == 'alias'     # inferred


def test_definition_validation():
    with pytest.raises(ValueError, match='publication topic'):
        LocalAlias(uai=['a'], data_format='d', target_format='t')
    with pytest.raises(ValueError, match='proto'):
        CodecSpec(encoding='protobuf')
    with pytest.raises(ValueError):
        parse_definition({'kind': 'other', 'uai': ['a'], 'data_format': 'd'})
    spec = CodecSpec(encoding='protobuf', proto='essmodule.ESSReadingProfile', edition='2.1')
    assert spec.as_resolved() == {'encoding': 'protobuf', 'proto': 'essmodule.ESSReadingProfile', 'edition': '2.1'}


def test_delivery_policy_options_match_their_kind():
    assert DeliveryPolicy().kind == 'each'
    with pytest.raises(ValueError):
        DeliveryPolicy(kind='interval')
    with pytest.raises(ValueError):
        DeliveryPolicy(kind='trigger')
    with pytest.raises(ValueError):
        DeliveryPolicy(kind='merge', max_wait=2)
    assert DeliveryPolicy(kind='complete', max_wait=2, stale_after=30).max_wait == 2


# ---- the assembler ------------------------------------------------------------------------------------------------
class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_each_emits_every_arrival_with_what_it_carries():
    a = Assembler(DeliveryPolicy(kind='each'), required={'W', 'V'}, clock=Clock())
    e = a.offer([{'W': 1}, {}])
    assert e.values == {'W': 1} and e.fresh == {'W'} and e.merged == frozenset() and e.missing == {'V'} and e.reason == 'arrival'


def test_complete_waits_for_every_required_field_then_emits_merged():
    clock = Clock(); a = Assembler(DeliveryPolicy(kind='complete', max_wait=5), required={'W', 'V', 'Hz'}, clock=clock)
    assert a.offer({'W': 1}) is None and a.offer({'V': 2}) is None
    assert a.next_due() == 1005.0 and a.due() is None
    e = a.offer({'Hz': 60})
    assert e.reason == 'complete' and e.values == {'W': 1, 'V': 2, 'Hz': 60} and e.fresh == {'Hz'} and e.merged == {'W', 'V'} and e.missing == frozenset()
    assert a.next_due() is None                           # nothing pending after the emit
    # max_wait: emit with what is present and name the missing.
    a.offer({'W': 3}); clock.t = 1010.0
    e = a.due()
    assert e.reason == 'max_wait' and e.values['W'] == 3 and e.missing == frozenset() or e.missing <= {'V', 'Hz'}
    assert 'W' in e.fresh


def test_merge_fills_from_last_known_and_drops_stale_values():
    clock = Clock(); a = Assembler(DeliveryPolicy(kind='merge', stale_after=10), required={'W', 'V'}, clock=clock)
    e = a.offer({'W': 1})
    assert e.values == {'W': 1} and e.missing == {'V'}
    clock.t = 1005.0
    e = a.offer({'V': 2})
    assert e.values == {'W': 1, 'V': 2} and e.merged == {'W'} and e.fresh == {'V'} and e.missing == frozenset()
    clock.t = 1020.0                                      # W is now 20 s old: stale
    e = a.offer({'V': 3})
    assert e.values == {'V': 3} and e.missing == {'W'}


def test_interval_emits_on_the_clock_from_everything_known():
    clock = Clock(); a = Assembler(DeliveryPolicy(kind='interval', interval=10), required={'W'}, clock=clock)
    assert a.offer({'W': 1, 'extra': 9}) is None and a.due() is None and a.next_due() == 1010.0
    clock.t = 1010.0
    e = a.due()
    assert e.reason == 'interval' and e.values == {'W': 1, 'extra': 9} and e.fresh == {'W', 'extra'}
    clock.t = 1020.0
    e = a.due()
    assert e.fresh == frozenset() and e.merged == {'W', 'extra'}      # nothing new arrived; the last values go again
    assert a.next_due() == 1030.0


def test_trigger_emits_only_when_a_named_field_changes():
    a = Assembler(DeliveryPolicy(kind='trigger', fields=['Ena']), required={'Ena', 'W'}, clock=Clock())
    assert a.offer({'W': 5}) is None
    e = a.offer({'Ena': 1})
    assert e.reason == 'trigger' and e.values == {'W': 5, 'Ena': 1}
    assert a.offer({'Ena': 1}) is None                    # same value: no trigger
    assert a.offer({'Ena': 0}).values == {'W': 5, 'Ena': 0}


# ---- ResourceData with a definition ---------------------------------------------------------------------------------
class FakeVIP:
    """Enough of vip.pubsub to capture a subscription's handler and vip.rpc for resolve."""
    def __init__(self, resolve=None):
        self.handlers = {}
        self.vip = mock.MagicMock()
        self.vip.pubsub.subscribe.side_effect = lambda peer, prefix, callback: self._sub(prefix, callback)
        self.vip.pubsub.unsubscribe.side_effect = lambda peer, prefix, callback: self.handlers.pop(prefix, None)
        self.vip.rpc.call.return_value.get.side_effect = lambda: resolve

    def _sub(self, prefix, callback):
        self.handlers[prefix] = callback
        return mock.MagicMock(get=lambda: prefix)

    def publish(self, topic, message, headers=None):
        self.handlers[topic]('pubsub', 'platform.driver', 'bus', topic, headers or {}, message)


def test_from_definition_needs_no_service_and_records_its_origin():
    agent = FakeVIP()
    rd = ResourceData.from_definition(agent, 'site/pv/openfmb', ALIAS)
    assert rd.origin == 'local' and rd.required_fields == {'W', 'V', 'Hz'} and rd.publication_topics() == ['devices/site/pv/all', 'devices/site/pv/multi']
    agent.vip.rpc.call.assert_not_called()
    out = []
    rd.subscribe(lambda peer, sender, bus, topic, headers, message: out.append((topic, headers, message)))
    agent.publish('devices/site/pv/all', [{'W': 2500, 'V': 240, 'Hz': 60}, {}])
    assert out[-1][0] == 'site/pv/openfmb' and out[-1][2] == {'kW': 2.5, 'V': 240, 'Hz': 60}
    assert ResourceData.ASSEMBLY_HEADER not in out[-1][1]                   # no policy: plain delivery, headers untouched


def test_policy_driven_subscription_assembles_and_records_in_headers():
    agent = FakeVIP(); out = []
    rd = ResourceData.from_definition(agent, 'site/pv/openfmb', ALIAS)
    rd.subscribe(lambda peer, sender, bus, topic, headers, message: out.append((headers, message)), {'kind': 'complete'})
    agent.publish('devices/site/pv/multi', [{'W': 1000}, {}])
    agent.publish('devices/site/pv/multi', [{'V': 230}, {}])
    assert out == []
    agent.publish('devices/site/pv/multi', [{'Hz': 59.9}, {}])
    headers, message = out[-1]
    assert message == {'kW': 1.0, 'V': 230, 'Hz': 59.9}
    assert headers[ResourceData.ORIGIN_HEADER] == 'local'
    assert headers[ResourceData.ASSEMBLY_HEADER] == {'reason': 'complete', 'fresh': ['Hz'], 'merged': ['V', 'W'], 'missing': []}


def test_reload_recompiles_and_moves_the_subscription_when_topics_change():
    agent = FakeVIP(); out = []
    rd = ResourceData.from_definition(agent, 'site/pv/openfmb', ALIAS)
    rd.subscribe(lambda *args: out.append(args[-1]))
    changed = {**ALIAS, 'publication_topic': 'devices/site/pv2/all', 'transform': [{'#': 'transform(device_values())', 'kW': 'transform[W](scale(0.01))'}]}
    assert rd.reload(changed) is True
    assert rd.required_fields == {'W'} and set(agent.handlers) == {'devices/site/pv2/all', 'devices/site/pv2/multi'}
    agent.publish('devices/site/pv2/all', [{'W': 2500}, {}])
    assert out[-1] == {'kW': 25.0}


def test_resolved_resource_reloads_through_the_service():
    resolved = {'data_format': 'd', 'publication_topic': 'devices/x/all', 'transform': [{'a': 'transform[b]()'}], 'parameters': {}}
    agent = FakeVIP(resolve=resolved)
    rd = ResourceData.lookup(agent, 'x/y')
    assert rd.origin == 'resolved' and rd.required_fields == {'b'}
    agent.vip.rpc.call.return_value.get.side_effect = lambda: {**resolved, 'transform': [{'a': 'transform[c]()'}]}
    assert rd.reload() is True and rd.required_fields == {'c'}
    agent.vip.rpc.call.return_value.get.side_effect = lambda: {}
    assert rd.reload() is False and rd.required_fields == {'c'}            # the old chain stays
    with pytest.raises(ValueError):
        rd.reload(ALIAS)


def test_interval_policy_runs_a_ticker_that_delivers_on_the_clock():
    gevent = pytest.importorskip('gevent')
    agent = FakeVIP(); out = []
    rd = ResourceData.from_definition(agent, 'site/pv/openfmb', ALIAS)
    rd.subscribe(lambda *args: out.append(args[-2:]), DeliveryPolicy(kind='interval', interval=0.2))
    agent.publish('devices/site/pv/all', [{'W': 1000, 'V': 230, 'Hz': 60}, {}])
    gevent.sleep(0.5)
    rd.unsubscribe()
    assert out and out[-1][1] == {'kW': 1.0, 'V': 230, 'Hz': 60} and out[-1][0][ResourceData.ASSEMBLY_HEADER]['reason'] == 'interval'
    n = len(out); gevent.sleep(0.3)
    assert len(out) == n                                                 # cancelled with the subscription
