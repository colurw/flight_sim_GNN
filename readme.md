# Flight_Sim_GNN

Flight_Sim_GNN uses reinforcement learning (RL) techniques to produce 
combat capable 'pilot' agents.  It integrates Pytorch neural networks with a 
physics-based flight simulator, and an optimisation algorithm inspired by 
Darwinian evolution.

## Reinforcement Learning

Neural networks typically learn in a supervised manner, with labelled training
data.  In our case this is not available due to the autonomous nature of the agent,
and the large number of discrete actions it takes in the training environment.  This
means that deep learning methods such as classical back-propagation through a neural network are ruled out.  
Because of this we are compelled to consider reinforcement learning techniques, where labelled training data are replaced by an overall fitness score that provides the feedback required for model weight tuning.  This fitness score is awarded to each agent based on their combat performance.

Based on work by John Holland, Genetic Algorithms (GAs) try to mimic the fundamental 
processes of natural selection: fitness, crossover, and mutation.  The unit of 
reproduction normally used by a GA is the genome - basically a hash function that 
can be easily crossed or mutated before being expanded out to generate the phenotype 
of whatever object is being optimised.

This method of producing the next generation usually offers the advantage of being 
able to search across the entire parameter-space relatively effectively, but the
reductive nature of hashing seems at odds with the finely-tuned innards of a neural 
network.  Flight_Sim_GNN instead considers all the weights and biases of the network as 
its genome, and uses a less aggressive averaging function when crossing parent 
networks, rather than the biomimetic 'random cut-and-paste' approach traditionally 
implemented.

Genetic algorithms have some disadvantages compared to typical machine learning 
optimisation techniques.  Like real evolution, there is no guiding hand at work.
Aiming to maximise fitness by competing against peers results in a random walk 
through the solution-space.  Mutated children infrequently outcompete their 
parents, and it can take many generations before an improvement is seen, resulting 
in a much slower optimisation process than gradient descent, during supervised learning, for example.

## The Genetic Algorithm

Training_loop_elite.py contains the optimisation loop that creates new generations 
of agents based on a selection of the fittest members from the previous generation.  

They, along with the previous generation's fittest, must compete in a handicapped 
elite contest (_i.e._ against the all-time highest scoring model). A points
handicap is given to earlier generations to discourage stagnation during the later stages 
when the opponent becomes more capable, and hence more difficult to score against.  

The best pilots in each generation are selected by calculating a weighted fitness 
score from the various flight metrics recorded during each match.
A summary of each generation's progress is then written to flight_scores.txt, and
any model that usurps the current highest score is saved.

## The Neural Network

The GeneticNeuralNetwork() class defines a Pytorch deep network.  Motion data from 
its own and the enemy plane are passed to the input layer, whilst the output layer is 
a categorical classifier mapped to the plane's controls.  The network is not trained 
through usual gradient descent methods, but by randomly tweaking its parameters 
until improvements are noticed.

These random changes are enacted by the mutate() and speciate() methods - the former
applies small random adjustments to every weight and bias of the network, whereas the 
latter applies a larger random adjustment to only a small percentage of the same 
parameters.  The speciate() method aims to explore more of the solution-space faster and also to
avoid the optimisation becoming stuck in local maxima of the fitness manifold.

When called, both these methods randomise, within a limited range, the size of their
effect to be more capable of both fine-tuning and efficient searching, as
is necessary during different stages of the training epoch.

## The Flight Simulator

Environment.py contains the Plane() class, which holds its current position and 
motion parameters as attributes, and updates these each processor cycle according 
to its physics model.  

An opponent Plane() instance can be designated as a target, so that each instance 
is aware of its competitor's location and motion parameters.  Each instance can manoeuvre to gain an advantage, or to fire its cannon.  Various performance 
metrics are tallied, such as length of flight, how close a shot string was to 
the enemy plane, hits taken, and making full use of the flight envelope.  

Plane() instances also have a basic autopilot option, to provide a long-lived 
moving target in the earliest stages of training, before being replaced with
an instance of GeneticNeuralNetwork().

## The Viewer

Combat_viewer.py uses Matplotlib to visualise combat between two instances of the
current fittest pilot, or between those from earlier in the high score lineage.

## Results

Draw_training_graphs.py uses the winner of each generation's flight data and fitness
scores to produce graphs that track the evolution of fitness in the population during training. 

## Adversarial Contest Mode

Training_loop_adversarial.py works similarly to the elite version, except that two separate
populations are trained simultaneously, with each member of a new generation competing 
against the highest scoring member from the other population.  

This delivers a significant reduction in the training time needed to converge upon a 
stable solution, _and_ with an order-of-magnitude increase in the fitness score at that point.  

![image](https://github.com/colurw/flight_sim_GNN/assets/66322644/66d0bb6b-ec7b-4eef-9f63-23d65cda377a)

![image](https://github.com/colurw/flight_sim_GNN/assets/66322644/6d9c6909-cb41-440a-a351-a38652a61f2d)

![image](https://github.com/colurw/flight_sim_GNN/assets/66322644/290fd42a-ad25-4457-8e9d-30b3324926dc)

![image](https://github.com/colurw/flight_sim_GNN/assets/66322644/5f9c7f9c-133c-449e-bf6d-b815ca6c99af)




