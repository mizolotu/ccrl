# State-of-art RL algorithms with tf2

## Install

```bash
pip3 install -r requirements.txt 
```

## Train

```bash
python3 train_agent.py -e <environment> -a <algorithm> 
```

## Check results

Start tensorboard:

```bash
tensorboard --logdir <logdir>
```

In a browser, go to 
```bash
http://localhost:6006 
```
check the graph and stats.
