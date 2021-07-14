# State-of-art RL algorithms with tf2

## Install requirements:

```bash
pip3 install -r requirements.txt 
```

## Train an RL agent:

```bash
python3 train_agent.py -e <environment> -a <algorithm> 
```

## Check the training results:

Start tensorboard in the terminal:

```bash
tensorboard --logdir <logdir>
```

In a browser, go to 
```bash
http://localhost:6006 
```
check the graph and stats.
