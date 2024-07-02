# Project Continuous Control

## Project Details
This project is part of the Udacity Deep Reinforcement Learning nanodegree. The goal of this project is to solve the Reacher environment. 
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The environment is considered solved if a reward of +100 is obtain for 30 consecutive episodes.

## Getting Started
To set up your python environment to run the code in this repository, follow the instructions below.

1. **Install required dependencies.** You can install all required dependencies using the provided `setup.py` and `requirements.txt` file.

    ```bash
    !pip -q install .
    ```

2. **Download the Unity Environment.** Select the environment that matches your operating system from one of the links below:
    Version 2: Twenty (20) Agents
    - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

3. **Unzip the environment file and place it in the root directory of this repository.**

## Instructions
To run the code, follow the Getting started instructions, git clone this repository and go to the folder repository. Then just type:

python Test.py