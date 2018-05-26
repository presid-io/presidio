# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: health.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='health.proto',
  package='message_types',
  syntax='proto3',
  serialized_pb=_b('\n\x0chealth.proto\x12\rmessage_types\"%\n\x12HealthCheckRequest\x12\x0f\n\x07service\x18\x01 \x01(\t\"\x93\x01\n\x13HealthCheckResponse\x12@\n\x06status\x18\x01 \x01(\x0e\x32\x30.message_types.HealthCheckResponse.ServingStatus\":\n\rServingStatus\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x0b\n\x07SERVING\x10\x01\x12\x0f\n\x0bNOT_SERVING\x10\x02\x32X\n\x06Health\x12N\n\x05\x43heck\x12!.message_types.HealthCheckRequest\x1a\".message_types.HealthCheckResponseb\x06proto3')
)



_HEALTHCHECKRESPONSE_SERVINGSTATUS = _descriptor.EnumDescriptor(
  name='ServingStatus',
  full_name='message_types.HealthCheckResponse.ServingStatus',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SERVING', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NOT_SERVING', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=160,
  serialized_end=218,
)
_sym_db.RegisterEnumDescriptor(_HEALTHCHECKRESPONSE_SERVINGSTATUS)


_HEALTHCHECKREQUEST = _descriptor.Descriptor(
  name='HealthCheckRequest',
  full_name='message_types.HealthCheckRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='service', full_name='message_types.HealthCheckRequest.service', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=31,
  serialized_end=68,
)


_HEALTHCHECKRESPONSE = _descriptor.Descriptor(
  name='HealthCheckResponse',
  full_name='message_types.HealthCheckResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='message_types.HealthCheckResponse.status', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _HEALTHCHECKRESPONSE_SERVINGSTATUS,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=71,
  serialized_end=218,
)

_HEALTHCHECKRESPONSE.fields_by_name['status'].enum_type = _HEALTHCHECKRESPONSE_SERVINGSTATUS
_HEALTHCHECKRESPONSE_SERVINGSTATUS.containing_type = _HEALTHCHECKRESPONSE
DESCRIPTOR.message_types_by_name['HealthCheckRequest'] = _HEALTHCHECKREQUEST
DESCRIPTOR.message_types_by_name['HealthCheckResponse'] = _HEALTHCHECKRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

HealthCheckRequest = _reflection.GeneratedProtocolMessageType('HealthCheckRequest', (_message.Message,), dict(
  DESCRIPTOR = _HEALTHCHECKREQUEST,
  __module__ = 'health_pb2'
  # @@protoc_insertion_point(class_scope:message_types.HealthCheckRequest)
  ))
_sym_db.RegisterMessage(HealthCheckRequest)

HealthCheckResponse = _reflection.GeneratedProtocolMessageType('HealthCheckResponse', (_message.Message,), dict(
  DESCRIPTOR = _HEALTHCHECKRESPONSE,
  __module__ = 'health_pb2'
  # @@protoc_insertion_point(class_scope:message_types.HealthCheckResponse)
  ))
_sym_db.RegisterMessage(HealthCheckResponse)



_HEALTH = _descriptor.ServiceDescriptor(
  name='Health',
  full_name='message_types.Health',
  file=DESCRIPTOR,
  index=0,
  options=None,
  serialized_start=220,
  serialized_end=308,
  methods=[
  _descriptor.MethodDescriptor(
    name='Check',
    full_name='message_types.Health.Check',
    index=0,
    containing_service=None,
    input_type=_HEALTHCHECKREQUEST,
    output_type=_HEALTHCHECKRESPONSE,
    options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_HEALTH)

DESCRIPTOR.services_by_name['Health'] = _HEALTH

# @@protoc_insertion_point(module_scope)
