from typing import Dict, List, Optional
from argparse import ArgumentParser

import scallopy
from scallopy.plugin.utils import get_device as get_device, set_device

class ScallopGPUPlugin(scallopy.Plugin):
  def __init__(self):
    super().__init__()

  def setup_argparse(self, parser: ArgumentParser):
    parser.add_argument("--cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--gpu", type=int, default=None, help="The ID of GPU to use")

  def configure(self, args: Dict = {}, unknown_args: List = []):
    self.configure_device(args.get("cuda", False), args.get("gpu"))

  def configure_device(self, use_cuda: bool = False, gpu: Optional[int] = None):
    if use_cuda:
      set_device(f"cuda:{gpu}" if gpu is not None else "cuda")
    else:
      set_device("cpu")
