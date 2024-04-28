## Introduction
This project involves implementing various reinforcement learning strategies for the 
Pacman game, including Tabular Q-learning, Approximate Q-learning, neural network-guided 
Q-learning, and Deep Q-learning. Each strategy requires developing and testing algorithms to 
enhance the game's performance across different levels and modes.


## Learning Objectives and Tasks

Each reinforcement learning strategy implemented in this project aims to optimize the decision-making process in the Pacman game, improving the efficiency and intelligence of the agent:
##### Tabular Q-learning: 
Focuses on creating a simple look-up table to store and retrieve values for each state-action pair.  
##### Approximate Q-learning:
Uses weighted features to estimate Q-values, reducing the dimensionality and potentially increasing the learning speed.  
##### Neural Network-guided Q-learning: 
Integrates neural networks to approximate Q-values, providing a powerful way to handle complex scenarios with high state space.  
##### Deep Q-learning: 
Extends neural network capabilities to directly learn the optimal policies over raw game states, aiming for high performance in complex levels.  


## Design Patterns Used

To maintain a clean and manageable codebase, the project employs several design patterns:

Factory Pattern: Used to create objects without specifying the exact class of object that will be created. This is particularly useful for creating different learning strategies dynamically.  
Repository Pattern: Facilitates a separation between the data layer and the business logic, ensuring that the game's state management is decoupled from the learning algorithms.  
Strategy Pattern: Enables the dynamic switching between different learning algorithms during runtime, allowing the game to adapt its strategy based on performance.  
Observer Pattern: Implemented to efficiently update the game's state in response to player actions or game events, ensuring real-time responsiveness without tight coupling between the game's components.  


## Conclusion

By employing advanced reinforcement learning techniques and robust software design principles, this project not only enhances the gameplay experience but also serves as a practical application of theoretical AI concepts in a dynamic environment. The use of MVC and strategic design patterns ensures that the project remains extensible, maintainable, and scalable.
