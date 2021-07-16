# Crypto-currency RL-based trading agents

## Project plan

1. Find a working implementation of PPO for continuous action space in tf2 - <b>Done!</b>
2. Test the implementation with an OpenAI gym environment - <b>Done!</b>
3. Add save/load mechanism to the PPO implementation - <b>In progress...</b> 
4. Collect historical data for several cryptocurrencies from different exchanges using ccxt - <b>In progress...</b>
5. Create an offline environment using the historical data to train an agent 
6. Train the agent in the resulting environment, check different hyperparameters
7. Improve the agent by adding more features, changing the reward function, and/or modifying the algorithm until it becomes profitable
8. Implement a script for online recommending an optimal trading action using the agent trained
9. Collect historical data from uniswap
10. Create an offline environment with uniswap data
11. Train the second agent using the resulting environment
12. Modify the parammeters until it becomes profitable
13. Use the resulting agent to predict an optimal action online 
