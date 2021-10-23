# Report for Continuous Control - Deep RL NanoDegree P2

The goal of this project is to create an agent that can be trained to maintain a double-jointed arm on a target location for as many time steps as possible.

The environment is solved once the agent has accumulated an average score of 30 over 100 episodes. An example of a trained agent can be seen below:

![trained agent](https://i.imgur.com/0JG7ud8.gif)

## 1. Implementation
I used the `DDPG` algorithm, with a `uniformly sampled experience replay buffer`.