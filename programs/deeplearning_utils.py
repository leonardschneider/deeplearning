from itertools import takewhile
import numpy as np
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError
from safetensors import deserialize
from io import BytesIO
import json
import math

nptypes = {
  "f32": np.float32,
}

def tensortype(fut_type):
  """
  Convert a futhark type to a tensor type
  """
  return fut_type.strip().upper()

def futharktype(tensor_type):
  """
  Convert a tensor type to a futhark type
  """
  return f"{tensor_type.lower():<4}"

# Parse a futhark model weights specifications
# Returns a dictionary with the following structure:
# Key: label
# Value: (type, list of dims)
def parse_spec(spec):
  bs = spec.tobytes()
  res = {}
  label = ""
  dims = []
  i = 0
  while i < len(bs):
    l = bytes(takewhile(lambda b: b != 0, bs[i:]))
    i += len(l)
    label = l.decode("utf-8")
    assert bs[i] == 0, f"Expected null byte, got {bs[i]}"
    i += 1
    assert bs[i:i+2] == b"b\x02", f"Expected b\\x02, got {bs[i:i+2]}"
    i += 2
    tpe = bs[i:i+4].decode()
    i += 4
    ndims = int.from_bytes(bs[i:i+1], byteorder='little')
    i += 1
    dims = []
    for j in range(ndims):
      dim = int.from_bytes(bs[i:i+8], byteorder='little')
      i += 8
      dims.append(dim)
    res[label] = {"tpe": tpe.strip(), "dims": dims}
  return res

# Get model weights from huggingface hub safe tensor
def load_weights_from_hf_hub(model, repo_id, filename, token):
  sf = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
  specs = parse_spec(model.from_futhark(model.specs()))
  res = bytearray()
  with open(sf, "rb") as f:
    weights = dict(deserialize(f.read()))
    for label, spec in specs.items():
      assert label in weights.keys(), f"Missing {label} in weights"
      assert weights[label]["dtype"].lower() == spec["tpe"], f"Expected dtype {spec['tpe']} from model spec, got {weights[label]["dtype"]} from safetensors"
      assert weights[label]["shape"] == spec["dims"], f"Expected shape {spec['dims']} from model spec, got {weights[label]["shape"]} from safetensors"
      #print(f"Loading {label} with shape {spec['dims']} and dtype {spec['tpe']}")
      #print(np.frombuffer(weights[label]["data"], dtype=nptypes[spec["tpe"]]).reshape(spec["dims"]))
      res += weights[label]["data"]
  return model.load_weights(np.frombuffer(bytes(res), dtype=np.uint8))

def weights_to_safetensors(model, ws):
  """
    Convert a futhark model weights to a safetensors format
    Returns a bytearray
  """
  header = {}
  start = 0
  specs = parse_spec(model.from_futhark(model.specs()))
  
  for label, spec in specs.items():
    dims = spec["dims"]
    tpe = spec["tpe"]
    size = math.prod(dims) * np.dtype(nptypes[tpe]).itemsize
    header[label] = {
      "dtype": tensortype(tpe),
      "shape": dims,
      "data_offsets": [start, start+size]
    }
    start += size
  header = json.dumps(header).encode("utf-8")
  header = len(header).to_bytes(8, byteorder='little') + header
  data = model.from_futhark(model.save_weights(ws)).tobytes()
  return header + data

def push_to_HF_hub(model, repo_id, futhark_model_file, ws, commit_message, token):
  """
    Push a futhark model to huggingface hub.
    Args:
      model: futhark model instance
      repo_id: huggingface repo id
      futhark_model_file: path to the futhark model file
      ws: weights of the futhark model
      commit_msg: commit message
      token: huggingface token
  """
  safetensor_weights = weights_to_safetensors(model, ws)
  api = HfApi()
  # Create repo if needed
  try:
    api.auth_check(repo_id=repo_id, token=token)
  except RepositoryNotFoundError:
    api.create_repo(
      repo_id=repo_id,
      repo_type="model",
      token=token,
    )
  # Upload weights
  api.upload_file(
      path_or_fileobj=BytesIO(safetensor_weights),
      path_in_repo="model.safetensors",
      repo_id=repo_id,
      repo_type="model",
      commit_message=commit_message,
      token=token,
  )
  # Upload model
  api.upload_file(
      path_or_fileobj=open(futhark_model_file, "rb"),
      path_in_repo=futhark_model_file,
      repo_id=repo_id,
      repo_type="model",
      commit_message=commit_message,
      token=token,
  )

