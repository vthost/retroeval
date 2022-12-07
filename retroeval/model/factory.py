import os
import json
import importlib
from argparse import Namespace

from retroeval.utils.config import DIR_WRAPPER_ARGS

# uncomment this for GLN
# try:
# 	import sys
# 	sys.argv += ['-f_atoms', './models/gln/cooked_schneider50k/atom_list.txt']
# 	# this one must be reloaded with f_atoms before loading any GLN gnn functionality, otherwise node feature dims are wrong
# 	import GLN.gln.mods.mol_gnn.mg_clib.mg_lib as mg_lib
# 	import GLN.gln.test.model_inference as model_inference
# 	from GLN.gln.test.model_inference import RetroGLN
# 	from examples.wrappers.gln2 import GLNWrapper
# except Exception as e:
# 	# print(e)
# 	print("GLN import could not be resolved")
# 	pass


# set checkpoint "" if you set your own args directly in  kwargs
def create_single_step_model(name, checkpoint="uspto-50k", **kwargs):
	file = f"{DIR_WRAPPER_ARGS}/{name}.json"
	if not os.path.exists(file):
		raise ValueError(f'Model arguments must be provided in directory {DIR_WRAPPER_ARGS}')

	with open(file) as f:
		data = json.load(f)

	if checkpoint:
		if "checkpoints" not in data:
			raise KeyError('JSON must contain `checkpoints` key')
		if "checkpoints" not in data:
			raise KeyError(f'JSON must contain `checkpoints`.`{checkpoint}` key')
		add_args = data["checkpoints"][checkpoint]
		for k, v in add_args.items():
			kwargs[k] = v
			# setattr(args, k, v)

	if "module" not in data:
		raise KeyError('JSON must contain `module` key')
	if "class" not in data:
		raise KeyError('JSON must contain `class` key')

	mod = importlib.import_module(data["module"])
	wrapper = getattr(mod, data["class"])

	return wrapper(Namespace(**kwargs))