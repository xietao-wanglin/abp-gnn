# Simulation of Active Brownian Particles with Graph Neural Networks

The project has been tested with Python 3.12.8.

## Using the project

### Install dependecies

`pip install -r requirements.txt`.

### Running a simulation

A sample simulation can be visualised using the launcher,

`python launcher.py`.

### Training a model

Firstly, generate a training and test dataset,

`python create_data.py`.

Then, train a model using the training script,

`python train.py`.
