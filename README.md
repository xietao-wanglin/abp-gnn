# Simulation of Active Brownian Particles with Graph Neural Networks

## About

Implementation of a Graph Neural Network for Active Brownian Particles. The repository also includes scripts to simulate the particles using numerical methods directly.

## Using the project

The project has been tested with Python 3.11.5, 3.12.9 on Linux using PyTorch 2.6.1, 2.7.0.

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
