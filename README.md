## Satellite Crop Classification
#### Supporting code for the master thesis "Satellite Crop Classification" written by _Abel de Wit_

This code is the support for the research performed in the master thesis "Satellite Crop Classification", written by _Abel de Wit_ in combination with [_Capgemini_](https://www.capgemini.com/service/digital-services/insights-data/).

In the research, different models and data-types are compared to benchmark the performance of the models.
The models are trained on the [PASTIS](https://github.com/VSainteuf/pastis-benchmark) dataset, wich is a satellite dataset of satellite images containing annotated images for 18 different crop types. 

In order to replicate the results of the research, the code is used in the following way:
* Install the dependencies using the command `pip install -r requirements.txt`
* Run the code using the command `python main.py`
* The results are saved in the folder `results`

The folder structure is as follows:
* models: contains the different models used, and adapted to the dataset
* results: contains the results of the experiments
* data: contains the dataset (not included on GitHub because of its size)
* utilities: contains several helper functions used in the code
