# sencoder
A sentence autoencoder using an RSSM and self attention.

# Features
Currently can operate on the amazon nietzsche dataset to encode and
decode strings of variable length.

To train a model, you must first navigate to the `training_scripts` directory and add an entry in the `hyperranges.json` to search over. It comes preloaded with a learning rate search. You should additionally set the `hyperparameters.json` to your liking. See the `hyperparams_explained.md` file for a description of each parameter.

Once the jsons are set appropriately, run the following:

```
$ python3 main.py <path_to_hyperparams_json> <path_to_hyperranges_json>
```

Your model will be saved in a folder within a folder under the `exp_name` parameter set in the hyperparams.json. Note, setting the `exp_name` to "test" will override some of the parameters for testing.

# Setup
Open a bash session and type the following:

```
$ git clone https://github.com/grantsrb/sencoder
$ cd sencoder
$ pip3 install --user -e .
```

If you are using Anaconda, change the last line to:

```
$ pip install -e .
```
