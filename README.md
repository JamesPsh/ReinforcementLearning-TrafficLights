# ReinforcementTrafficLights

## Overview

**ReinforcementTrafficLights** is a reinforcement learning project aimed at optimizing traffic lights using the SUMO simulator as the environment. It utilizes the DPPO algorithm and TensorFlow.

## Project Tutorial Video
<a href="http://www.youtube.com/watch?feature=player_embedded&v=PynRgDwR3rs
" target="_blank"><img src="http://img.youtube.com/vi/PynRgDwR3rs/0.jpg" 
alt="Traffic Signal Control with Reinforcement Learning: A Real Project Tutorial" width="240" height="180" border="10" /></a>


## Installation

Before running the project, please install the following dependencies:
~~~
pip install ray  
pip install pandas 
pip install sumolib
pip install tensorflow
~~~

## SUMO (Simulation of Urban MObility)

[here](https://www.eclipse.org/sumo/).

## Usage

### Training

To train the model, use the following command:
~~~
python main.py --name [name_of_your_experiment] --mode train
~~~

### Testing

To test the model, use the following command:
~~~
python main.py --name [name_of_your_experiment] --mode test
~~~
