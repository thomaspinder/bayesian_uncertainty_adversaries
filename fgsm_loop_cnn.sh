#!/bin/bash
for i in 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99
do
    python -m src.experiments 'vision' -m 3 -f $i --model 'cnn'
done
