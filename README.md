# Crypto currency RL-based trader

## Project plan

1. Find a working implementation of PPO for continuous action space in tf2 - <b>Done!</b>
2. Test the implementation with an OpenAI gym environment - <b>Done!</b>
3. Collect historical data for several cryptocurrencies from different exchanges using ccxt - <b>In progress</b>
4. Create an offline environment using the historical data to train an agent 
5. Train the agent in the resulting environment, check different hyperparameters
6. Improve the agent by adding more features, changing the reward function, and/or modifying the algorithm until it becomes profitable
7. Implement a script for online recommending an optimal trading action using the agent trained
8. Collect historical data from uniswap
9. Create an offline environment with uniswap data
10. Train the second agent using the resulting environment
11. Modify the parammeters until it becomes profitable
12. Use the resulting agent to predict an optimal action online 
