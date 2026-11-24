import argparse


_device = "cpu"


def get_device() -> str:
  return _device


def set_device(device: str) -> None:
  global _device
  _device = device


def _process_args(args):
  if isinstance(args, argparse.Namespace):
    return {str(key): args.__dict__[key] for key in args.__dict__.keys()}
  elif type(args) == dict:
    return args
  else:
    return {}
