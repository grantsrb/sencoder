This is a list of descriptions for each of the possible hyperparameters.

* `exp_name`: str
    * The name of the main experiment. Model folders will be saved within a folder of this name.
* `n_repeats`: int
    * The number of times to repeat any given hyperparameter set
* `save_every_epoch`: bool
    * A boolean determining if the model `state_dict` should be saved for every epoch, or only the most recent epoch.

* `model_type`: str
    * The string name of the main model class to be used for training. Currently the only option is "SeqAutoencoder"
* `rnn_type`: str
    * The string name of the rnn class to be used for the RSSM. Currently the only option is "GRU"
* `classifier_type`: str
    * The string name of the classifier class to be used for predicting word indices from recurrent states. Currently the only option is "SimpleClassifier"
* `train_in_eval`: bool
    * boolean denoting if the model should be trained in evaluation mode
* `attention`: bool
    * boolean denoting if self attention should be used in the encoding mechanism
* `enc_lossfxn`: str
    * The name of the loss function that should be used for the encoder. Currently only option is "CrossEntropyLoss"
* `dec_lossfxn`: str
    * The name of the loss function that should be used for the decoder. Currently only option is "CrossEntropyLoss"
* `dataset`: str
    * The name of the dataset that should be used for training. Currently only option is "Nietzsche",
* `shuffle`: bool
    * boolean determining if the order of samples with in a batch should be shuffled. This does not shuffle the sequence itself.

* `batch_size`: int
    * the number of samples to used in a single step of SGD
* `n_epochs`: int
    * the number of complete training loops through the data
* `optim_batches`: int
    * the number of batches to be included in a single step of SGD
* `wnorm`: bool
    * if true, model uses weight normalization where possible
* `bnorm`: bool
    * if true, model uses batch normalization where possible
* `bias`: bool
    * if true, model uses trainable bias parameters where possible
* `emb_size`: int
    * dimensionality of the token embeddings
* `h_size`: int
    * dimensionality of the deterministic recurrent state vector in the RSSM
* `s_size`: int
    * dimensionality of the stochastic recurrent state vector in the RSSM
* `seq_len`: int
    * the maximum length of the token sequences to be trained on
* `classifier_layers`: int
    * the number of layers in the classifier neural network.
* `dec_step_size`: int
    * the decoder will decode the encoding of every length divisble by `step_size` up to the `seq_len`. `seq_len` is always included.

* `lr`: float
    * the learning rate
* `dec_alpha`: float between 0 and 1
    * the portion of the loss from the decoder. The rest of the loss is atribued to the encoder.
* `l2`: float
    * the l2 weight penalty


