---
toc: true
url: SAnnealing
covercopy: © Karobben
priority: 10000
date: 2024-05-30 16:34:39
title: "Simulated Annealing (SA)"
ytitle: "Simulated Annealing (SA)"
description: ""
excerpt: "Simulated Annealing (SA) is a probabilistic technique used for finding an approximate solution to an optimization problem. It is particularly useful for problems where the search space is large and complex, and other methods might get stuck in local optima."
tags: [Math, Algorithm]
category: [AI, Math]
cover: "https://imgur.com/uWGjeSm.png"
thumbnail: "https://imgur.com/uWGjeSm.png"
---


Video tutorial: [Challenging Luck, 2021](https://www.youtube.com/watch?v=FyyVbuLZav8)

Practicing Python code: [challengingLuck: Using Annealing Algorithm to Solve the Sudo Challenge](https://github.com/challengingLuck/youtube/blob/master/sudoku/sudoku.py)

> Simulated Annealing (SA) is a probabilistic technique used for finding an approximate solution to an optimization problem. It is particularly useful for problems where the search space is large and complex, and other methods might get stuck in local optima. Here's a structured way to start learning about the Simulated Annealing algorithm:

$$ P(E, E', T) = \begin{cases} 
      1 & \text{if } E' < E \\\\
      \exp\left(\frac{-(E' - E)}{T}\right) & \text{if } E' \ge E 
   \end{cases} 
$$

This idea is similar to DNA annealing during PCR. During the temperature drop after the high-temperature mutation, the DNA gradually returns to the double strand with its reverse complement strand or the primer due to the decrease in entropy. Unlike DNA annealing, Simulated Annealing (SA) introduces random values to replace the original ones. After that, the "Energy" is calculated, and only when the new score is lower than the previous one is the new value accepted, continuing the iteration until the lowest value is found. It works like this:

Calculate the initial `E` → randomly mutate the value and calculate the new `E'` → if `E' ≤ E`, then accept the mutated element; otherwise, try another value → if `E'` meets the lowest `E`, stop; otherwise, continue until no smaller `E'` can be found.

As a result, you could expect that it would ==waste a lot of resources on exploration== and easily ==fall into local optima==.

An example of the SA apply in sudo challenge

![Simulated annealing in Sudo](https://imgur.com/k4jbsQK.gif)
![Simulated annealing in Sudo](https://imgur.com/uWGjeSm.png)
In this example, it actually ==failed== to get the result because it **fall into local optimal**. It is a very good example to show the capability and limitations of the SA.

## SA and Stochastic gradient descent

$$
w_{t+1} = w_t - \eta \nabla f(w_t; x_i)
$$

- $w$ is the parameter, weight matrix, for example. While $w_t$ is the old one and the $w_{t-1}$ is updated parameter.
- $\eta$ is the learning rate
- $x$ is the input

> **Stochastic Gradient Descent (SGD)** is an optimization algorithm used primarily for training machine learning models. It iteratively updates the model parameters by computing the gradient of the loss function using a randomly selected subset (mini-batch) of the training data, rather than the entire dataset. This randomness introduces variability in the updates, which helps escape local optima and speeds up convergence. The learning rate controls the step size of each update, determining how far the parameters move in the direction of the negative gradient. By continuously adjusting the parameters, SGD aims to minimize the loss function and improve the model's performance.

So, it is very similar to Stochastic gradient descent (SGD). But for SGD, there are a learning process from the data. SGD is primally based on the exploitation. But in SA, exploration has more contribution compared with SGD because it hugely relies on random generation first and evaluating later.

## SA and Genetic Algorithm (GA)

> What is GA?
> Genetic Algorithms (GA) are evolutionary algorithms inspired by the principles of natural selection and genetics. They work by evolving a population of candidate solutions through successive generations. Each generation undergoes selection, where the fittest individuals are chosen based on their performance. These selected individuals then undergo crossover (recombination) to produce offspring that combine their parents' characteristics. Additionally, mutation introduces random changes to some individuals to maintain genetic diversity within the population. This process of selection, crossover, and mutation allows GAs to explore a wide search space and balance between exploring new solutions and exploiting the best solutions found so far. This diversity helps GAs avoid getting trapped in local optima, making them effective for solving complex optimization problems, including those that are non-differentiable and non-convex.


From a personal point of view, GA is like an upgraded version of SA. SA works at the level of a single individual, while GA operates at the population level. Similar to SA, GA evaluates and selects the "fitness scores" of each individual. The next generation introduces many random mutations, just like SA. However, unlike SA, GA also includes "crossover" steps, which can help enrich the "better phenotypes".


| Feature                          | Simulated Annealing (SA)                               | Stochastic Gradient Descent (SGD)                   | Genetic Algorithm (GA)                                      |
|----------------------------------|-------------------------------------------------------|-----------------------------------------------------|-------------------------------------------------------------|
| **Approach**                     | Probabilistic, accepts worse solutions occasionally   | Deterministic, updates in the direction of the gradient | Evolutionary, uses selection, crossover, and mutation        |
| **Objective Function**           | Non-differentiable and non-convex functions           | Differentiable functions                            | Non-differentiable and non-convex functions                 |
| **Exploration vs. Exploitation** | Balances both, reduces acceptance of worse solutions over time | Primarily exploitation with some exploration via mini-batches | Balances both, uses population diversity to explore the search space |
| **Cooling Schedule / Learning Rate** | Uses a cooling schedule to reduce probability of accepting worse solutions | Uses a learning rate to control step size of updates | Uses selection pressure to favor better solutions and mutation rate to introduce diversity |
| **Population-Based**             | No                                                    | No                                                  | Yes, operates on a population of solutions                  |
| **Escape Local Optima**          | Yes, by accepting worse solutions with a probability  | Limited, may get stuck in local optima              | Yes, by maintaining a diverse population                    |
| **Gradient Requirement**         | No                                                    | Yes                                                 | No                                                          |
| **Applications**                 | Combinatorial and continuous optimization without gradients | Training machine learning models, especially in deep learning | Optimization problems, including scheduling, design, and artificial intelligence |
| **Natural Inspiration**          | Annealing in metallurgy                               | Gradient descent in calculus                        | Natural selection and genetics                               |
| **Operators**                    | Acceptance probability based on temperature           | Gradient-based updates                              | Selection, crossover (recombination), and mutation           |

> table from: ChatGPT4o








<style>
pre {
  background-color:#38393d;
  color: #5fd381;
}
</style>
