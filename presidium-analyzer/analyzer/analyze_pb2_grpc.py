# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import analyze_pb2 as analyze__pb2


class AnalyzeServiceStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.Apply = channel.unary_unary(
        '/types.AnalyzeService/Apply',
        request_serializer=analyze__pb2.AnalyzeRequest.SerializeToString,
        response_deserializer=analyze__pb2.Results.FromString,
        )


class AnalyzeServiceServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def Apply(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_AnalyzeServiceServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'Apply': grpc.unary_unary_rpc_method_handler(
          servicer.Apply,
          request_deserializer=analyze__pb2.AnalyzeRequest.FromString,
          response_serializer=analyze__pb2.Results.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'types.AnalyzeService', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))
