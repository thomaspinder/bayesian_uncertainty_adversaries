# Using Bayesian Deep Learning Approximations to Detect Adversaries in Computer Vision and Reinforcement Learning Tasks

## What Is This All About?
This code repository corresponds to my masters thesis which attempts to detect the presence of adversaries in common AI/ML taks using Bayesian CNNs. Bayesian CNNs are produced based upon the work of [Gal and Ghahramani](https://arxiv.org/pdf/1506.02142.pdf) who prove that by incorporating dropout layers, the neural network converges to a deep Gaussian process. Adversaries are created using the Fast Gradient Sign Method [(FGSM)](https://arxiv.org/pdf/1412.6572.pdf), with varying values of ![equation](http://latex.codecogs.com/gif.latex?%5Cepsilon) being tested. 

Results are collected by establishing baselines and then crafting adversarial examples and observing the performance decrease. Using a BCNN, we then utilise the ability to extract uncertainty measures and test whether an increase in uncertainty can be used to detect the presence of an adversary.

## Tasks
### MNIST Digit Classification
A baseline in any vision experiment, we test the classifiers performance and resilience to adversaries when classifying digits from the MNIST database.

### Pneumonia Detection
A more _real world_ challenge would be to test for the presence of pneumonia in an individual based upon a chest x-ray image - [data here](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). 

<img src="other/imgs/pneumonia.png" alt="Pneumonia X-Ray" width="500px"/>

### FlappyBird DQN
We also train an agent using the DQN of [Mnih et. al.](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) to play flappybird, replacing the original CNN with a BCNN. Once trained, we inflict adversarial attacks on the agent while playing flappybird.

<img src="other/imgs/flappy_bird.png" alt="Flappy Bird" width="500px"/>

## Installing

1. Clone repository ```git clone https://github.com/thomaspinder/bayesian_uncertainty```

2. Create virual environment 
```
cd bayesian_uncertainty
pip install virtualenv
virtualenv env -p python3
source env/bin/activate
```

3. Install requirements
```
pip install -r requirements.txt
```

## Setup 
### Train Vision Networks
Takes approximately 15minutes on an Nvidia 1050ti GPU.
```
python experiments.py 'vision' -m 0
```

### Train FlappyBird Agent
Note, on an Nvidia 1050ti GPU this process took 2.5 days to complete.
```
python experiments.py 'rl'
```


## Running Experiments
### Computer Vision Tasks
1. Comparing CNNs against Bayesian CNNs
```
python experiments.py 'vision' -m 0
```

2. Calculating uncertainty values as MNIST digits are rotated through 180 degrees
```
python experiments.py 'vision' -m 0
```



### Reinforcement Learning Tasks