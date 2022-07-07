## Satellite Crop Classification
#### Supporting code for the master thesis "Satellite Crop Classification" written by _Abel de Wit_

This code is the support for the research performed in the master thesis "Satellite Crop Classification", written by _Abel de Wit_ in combination with [_Capgemini_](https://www.capgemini.com/service/digital-services/insights-data/).

In the research, different models and data-types are compared to benchmark the performance of the models.
The models are trained on the [PASTIS](https://github.com/VSainteuf/pastis-benchmark) dataset, wich is a satellite dataset of satellite images containing annotated images for 18 different crop types. 

In order to replicate the results of the research, the code is used in the following way:
* Install the dependencies using the command `pip install -r requirements.txt`
* Run the code using the command `python src/main.py`
* There are several options available to the user:
  * Data:
    * path (str): The path to the PASTIS Dataset
    * batch_size (int): Overwrite the batch size used (is also defined in `src/experiments/experiments.json`)
    * norm (bool): Whether to use normalise the images (default: `True`)
    * fold (int): The fold to train on, if None all folds will be used (default: `None`) 
    * num_workers: The number of workers used for the DataLoader (default: `0`)
    
  * Experiments:
    * begin (int): The index of the first experiment to run (default: `0`) 
    * end (int): The index of the last experiment to run (default: `None`) 
    * model_only (str): The name of the model to run (default: `''`) 
    
  * Pytorch Lightning trainer:
    * epochs (int): The number of epochs to train (default: `500`) 
    * accelerator (str): The name of the accelerator to use (default: `'cpu'`)
    * devices (int): The amount of devices available (in the case of GPU training)
    * fast_dev (bool/int): This is a flag to use to quickly debug the whole training process (default: `False`). See: 
    [Lightning Debugging](https://pytorch-lightning.readthedocs.io/en/stable/common/debugging.html#fast-dev-run)
    * overfit_batches (int): This is a flag to use to see whether the model is able to overfit on a small subset of the data (default: `0`). See: [Overfit Batches](https://pytorch-lightning.readthedocs.io/en/stable/common/debugging.html#make-model-overfit-on-subset-of-data)
    * log_every_n_steps (int): When overfitting batches, the logging interval should be shortened if you want to make use of the logs (default: `50`)
    
  * Deep learning options:
    * learning_rate: The learning rate used for the optimizer (default: `0.001`)
    * patience: The patience used for the early stopping (default: `10`)
    * min_delta: The minimum change in the validation loss before early stopping (default: `0.0`)
    * monitor: The metric to monitor for early stopping and learning rate decay (default: `'metrics/val/loss'`)
    * monitor_mode:  The mode to use for the early stopping and learning rate decay (default: `'min'`)

Example command when training UTAE on GPU:
`python src/main.py --path=./data/PASTIS --num_workers=8 --accelerator=gpu --devices=1 --model_only=UTAE

The folder structure is as follows:
* gfx: The folder containing the graphics used in the report
* notebooks: Notebooks used for data exploration and model evaluation
* src: The folder containing the code
  * backbones: The folder containing the models that are tested:
    * UNet
    * UTAE
    * Vit
  * data: This folder is not included on github as it contains the PASTIS dataset
  * experiments: The folder containing the code to create the `experiments.json` file
  * utilities: Several helper functions such as downloading and visualising the dataset.
