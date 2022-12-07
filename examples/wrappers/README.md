## Single-Step Model Wrappers 


### Examples

For some models proposed in the literature, we provide example wrappers.
Please note that we gave our best to follow the original code in how the models are applied. If you spot any issues to be solved there, please let us know!

Examples are currently available for these models:

- Chemformer: https://github.com/MolecularAI/Chemformer.
- Graph2SMILES: https://github.com/coleygroup/Graph2SMILES. There is currently an import issue with this model. We are working on this.
- Graphretro: https://github.com/vsomnath/graphretro.  
- MLP: in [`examples/models`](../models)

 Note that we also provide the code for [GLN](https://github.com/Hanjun-Dai/GLN), yet, we were unable to create a reproducible environment setup. Please follow the instructions in the original repository if you want to run that model. 

### Add Your Own Wrapper

The interface is [`SingleStepModelWrapper`](../../retroeval/model/wrapper.py). This base class provides some very basic functionality. We recommend to have a short look at it, especially, the constructor lists some arguments you may use and should not overwrite: `self.args`, `self.model`, `self.topk`
You have to provide methods `load_model` and `_run`.
We use GLN as example because the wrapper is the most simple one:

````
from GLN.gln.test.model_inference import RetroGLN

class GLNWrapper(SingleStepModelWrapper):

    def __init__(self, args):
        super(GLNWrapper, self).__init__(args)
        // Initialize here all attributes you need for loading and running your model
        self.model_file = args.model_file
        ...

        // This is required, calls load_model
        super(GLNWrapper, self).__post_init__()

    def load_model(self):
        return RetroGLN(self.template_file, self.model_file)

    # return a dictionary containing this information, SORTED please (highest score first):
    # return {'reactants': list of reactant-smiles-strings, 'scores': list-of-float-scores}
    def _run(self, mol_smiles: str):
        results = self.model.run(mol_smiles, self.beam_size, self.topk)
        if results is None:
            return {'reactants': [], 'scores': []}
        results['scores'] = results['scores'].tolist()
        return results
````

Your code will be loaded automatically if you provide a JSON description in [`assets/wrapper-args`](../../assets/wrapper-args), which has to look as below.

````
{
  "module": "examples.wrappers.gln",
  "class": "GLNWrapper",
  "checkpoints": {
    "uspto-50k": {
      "model_file": "./models/gln/schneider50k.ckpt",
      "template_file": "./models/gln/",
      "data_name": "schneider50k"
    }
  }
}
````

The entries `module`, `class`, and `checkpoints` are fixed. The rest can be specified as needed.
In particlar, each checkpoint (here `uspto-50k`) lists the custom arguments for the constructor.

Finally, note that for running our scripts you may have to extend the python path as follows to point to your code or install it in the conda environment: 
<br/>`export PYTHONPATH=$PYTHONPATH:$LOCATION/GLN`
