"""Resources: the models a resource is declared with, the clients that read and write one, and the wire codecs.

``resource`` holds the ``Resource`` models and the ``ResourceData`` and ``ResourceWriter`` clients. ``codecs`` turns
transform output into a wire payload and back (JSON, or protobuf for OpenFMB). ``openfmb`` carries the OpenFMB
protobuf descriptors per edition and the message headers OpenFMB traffic travels with.
"""
from .assembler import Assembler, Emit, Ticker
from .definitions import CodecSpec, DeliveryPolicy, LocalAlias, LocalCanonical, parse_definition
from .resource import ALL, MULTI, AliasedResource, CanonicalResource, Resource, ResourceData, ResourceWriter

__all__ = ['ALL', 'MULTI', 'AliasedResource', 'Assembler', 'CanonicalResource', 'CodecSpec', 'DeliveryPolicy', 'Emit', 'LocalAlias',
           'LocalCanonical', 'Resource', 'ResourceData', 'ResourceWriter', 'Ticker', 'parse_definition']
