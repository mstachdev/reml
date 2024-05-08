## Learning to Learn by Gradient Descent as a Discrete-Time Finite-Horizon Markov Decision Process (2024)

## Abstract
Meta-learning, often referred to as "learning to learn," is a paradigm within machine learning that focuses on developing models that generalize knowledge across tasks. Such a model learns across tasks as it learns within tasks, enabling the model to adapt quickly to new, unseen tasks. A model for a new task can be generated with near-optimal parameters in no or very few training steps. Meta-learning in machine learning is desirable in cases where there is limited data for a task or when a model needs to learn efficiently within few data samples.

This thesis investigates whether Meta-learning can be cast as a Reinforcement Learning (RL) problem for non-RL tasks. RL applied to Meta-learning has included hyperparameter search for neural architectures and meta-policies in RL task domains (Meta-Reinforcement Learning). However, RL has not been used to train models (parameter search) for non-RL tasks in the Meta-learning paradigm. 

The thesis poses the question: Can Meta-learning be formulated as a sequential task? And if so, can Meta-learning be a discrete-time finite-horizon Markov Decision Process (MDP) solved by deep RL? Reinforcement Meta-Learning (REML) is proposed to formulate the Meta-learning task as composing neural networks layer by layer for sampled tasks with a shared parameter space across tasks. REML is evaluated according to the protocol proposed by Finn et al (2018) as used for the Model-agnostic meta-learning (MAML) algorithm.

REML is first tested on regression tasks as a series of varying sinusoidal curves. Performance is evaluated by looking at the loss per step (convergence speed) for a model constructed by REML compared to a model trained from scratch. REML is also tested on a few-shot learning task where it is given 5 and 10 samples and tested after 0, 1, and 10 gradient steps. Future work includes testing REML on classification and reinforcement learning tasks, and expanding the action space of REML to decide hyperparameters in addition to parameters.

## Architecture
[View Thesis PDF](./reml-diagram.pdf)
