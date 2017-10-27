# Stock-Market-Probabilities-Deep-Learning
Tries to predict if a stock will rise or fall with a certain percentage through giving probabilities of what events it thinks will happen.


## Requirements
* Python 3.*
* Tensorflow **1.3**
* tqdm
* Numpy
* Six


## Adjusting parameters
Before you create your dataset file and train your model it's adviseable that you adjust the parameters first. This you can do by editing the file ```config.py```. If you don't understand a parameter it's best to leave it alone.

When running a script you can always pass the parameters directly and override the ```config.py``` parameters. To get to know what the parameters are you can type ```python3 script_name.py -h```.


## Creating dataset file
As a default I use a csv file from Yahoo Finance of NASDAQ, you can see the format of the file and use the same format if you want to use your own dataset. 

Run ```python3 create_data_set.py```. This will create a file called ```data_set.pkl```. This script does everything that you'll need.


## Training
To train a model you firstly have to **create a dataset file**. After that it's as easy as running the script ```python3 train.py```


### Todos
* Evaluation script
* Fix a better dataset


### Other
Made by Oliver Edholm, 15 years old.
