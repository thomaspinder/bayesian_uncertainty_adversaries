#!/bin/bash
for i in 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
do
    python -m src.vision.mc_dropout_keras -m 0 -a True -e $i
done
