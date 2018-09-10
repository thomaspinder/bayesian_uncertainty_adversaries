# Experiments - Rough
## MNIST Experiments
### Baseline
The MNIST dataset comprised of images of handwritten digits between 0 and 9 is a benchmark test within computer vision and accuracies of 97\% upwards is not uncommon. The entire dataset contains 60000 images of which the same 50000 images were used for training both a CNN and BCNN. For the CNN, a standard LeNet architecture was used which first consists of two convolutional layers which are followed by two linear layers. Each layer uses the Rectified Linear Units (ReLU) activation function and a 2D max-pooling layer. Between the final and penultimate layer there exists a single dropout layer with probability 0.5 with this layer being disabled at test time. The only divergence from this architecture in the BCNN is that every layer contains a dropout layer of probability 0.5 and this probability is kept on at test time.

Each network was trained for a total of twenty epochs using stochastic gradient descent with a learning rate initialised to 0.01 and momentum of 0.5. Finally, to compute the loss between the model's predictions and the ground truth during training, the negative log-likelihood loss across log-softmax probabilities is used as per __Check Concrete Dropout Imp to see if they too used it.__

Once both networks had finished training, the baseline accuracy of each was found using the entire 10000 test set. Across the entire test set, the CNN and BCNN reported accuracies of 98.34\% and 98.53\% respectively. We can visualise these results at a macro level through the set of heatmaps in Figure __Add MNIST Heatmap Fig. Here__. Within these heatmaps the results have been logarithmically scaled to better highlight the classes where the networks struggle. __Analyse heatmaps once numbers are overlaid__.

### MNIST adversaries
Using FGSM, we are now in a position to investigate the effect that adversaries have on each of our networks. It is evident from Figure __ref eps vs. acc plot__ that both networks are immune to epsilon at lower values before the effects of an increasing epsilon quickly become devastating for the classifier's accuracy. We can get some initial intuition into the effects of epsilon by fitting a simple linear regression (SLR) model to fit accuracy against epsilon, using a known intercept of 0.94 (the average of the two classifier's accuracies). This makes the model \footnote{Note a slight deviation from common notation as typically $\epsilon$ is used to model the model's error. As $\epsilon$ here denotes thr $l_{\infty}$ bound of FGSM, $\tau$ is used for the model's eror.}

\begin{align}

\bm{I}\cdot \bm{\hat{y}}-0.94 = \beta_0 \cdot \bm{\epsilon} + \tau.

\end{align}

From this model we obtain a coefficient of -0.26 on $\epsilon$, indicating that there a unit (+0.1) increase in $\epsilon@£ will cause a 2.5 drop in the either network's predictive power. It should be noted that this linear regressor is not a perfect fit, with a heavy skew being present in residuals, however, it does allow us to gauge at a high level the effect of epsilon on the predictive accuracy of a classifier.

As can be seen in Figure __ref eps vs. acc plot__, the BCNN is consistently more severely impinged when compared to the CNN.

# Bayesian BEST Test
We now assign prior distributions to the parameters of our likelihood; $\mu$ and $\sigma$. \citeauthor{Kruschke2013BayesianTest.} recommend using very vague priors so as not to overly influence the final posterior. As per these recommendations, a normal distribution is selected as the prior over $\mu$, with mean equal to the combined sample's mean and standard deviation equal to the sample's combined standard deviation, multiplied by 1000 to ensure a the probability density isn't too concentrated in a single area. For the likelihood's $\sigma$ parameter, a exponential prior distribution is used with $lambda=0.1$ to ensure a non-negligible amount of probability density occurs in the distribution's tail.

With a likelihood defined and a prior distribution specified over each of the likelihood's parameters we can proceed to fit our posterior distribution. Multiplying our likelihood and priors together will yield a posterior, however, to know the density in its full form we must adopt an MCMC approach. Using the Metropolis-Hastings algorithm from Section __add MH-Sampler Section__, we can attempt to sample from this posterior. All we need supply to the MH-Sampler is some reasonable starting values for our four parameters __Add in we're using a slightly different MH Sampler__.

\begin{align*}
\mu_{adv}=0.01 \quad & \quad \sigma_{adv}=5\\
\mu_{ori}=0.01 \quad & \quad \sigma_{ori}=5.
\end{align*}


We run the sampler for 10000 iterations, taking a final burn-in of 1000 samples. From Figure __Trace Plot Ref__ it can be seen that our Markov chain is __ergodic????__, exploring a large portion of the parameter space with little evidence of any sticking points. This convergence allows us to now infer the difference between the uncertainty estimates arising from adversaries compared to unperturbed images. __Add Posterior Inference Here__.

# Xray Data
\subsection{Real Word Dataset}
While establishing baselines within a community through datasets such as the MNIST dataset is important, these datasets are not particularly representative of real-world datasets. Further to this, there is the possibility that over time we will, as a community, only develop models good at predicting handwritten digits, therefore not developing \textit{true} computer vision models. To extend this work to a real world domain, we test both networks on an image dataset representing 5,863 chest x-rays \footnote{Dataset: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/home}. Unlike the MNIST task, this is a binary classification as to whether the observation has pneumonia or not.

__Insert Image of Pneumonia and non-pneumonia xray__

__Add MNIST Image somewhere in report__

The nature of this task is fundamentally the same as classifying MNIST digits, however, we are now trying to extract image based details at a much finer granularity. To successfully accomplish this task we must therefore adapt our model slightly in order for the task's increased complexity to be managed. We can do this by adding an additional convoluting layer into our network to enhance the model's power for learning features of the image. Remaining parameters are kept constant and loss and cost functions are unchanged.

We train both a CNN and BCNN on __No. TRAINING IMAGES__ training images, with the CNN and BCNN achieving accuracies of 92.24\% and 93.17\%, respectively. As with the MNIST dataset, both network's achieve comparably high accuracies, the BCNN slightly outdoing the CNN's performance. Due to the task now being a binary classification task, we can easily probe deeper into each classifiers performance by examining precision and recall metrics. We can formally define these metrics as

\begin{align*}
\text{Precision} = \frac{\text{True Positive}}{\text{True Positive} + \text{False Positive}} \quad \text{Recall} = \frac{\text{True Positive}}{\text{True Positive} + \text{False Negative}}.
\end{align*}

Intuitively we can think of precision as identifying the number of positive classifications that had a positive ground truth value. In the current pneumonia example this corresponds with diagnosing someone with pneumonia, based upon an xray image, and the person did in fact have pneumonia. Recall measures the amount of actual positives that were correctly identified as positive. Again, tying this back to the contextual case of pneumonia, recall would measure the number of patients suffering from pneumonia who were found to have the disease based upon the xray image. Both of these metrics are useful in practice and often an improvement in one corresponds to a decrease in the other. A good model should not only have high accuracy, but also reasonable precision and recall values.

__Insert both confusion mats__

We can visualise the precision and recall of our pneumonia CNN and BCNN through the confusion matrix in Figure __insert confusion matrix ref__. From this, we can calculate that the CNN classifier has a recall of 77.9 and precision 92.2. Similarly, the BCNN attains a recall of 79.5 and 94.3 for precision. Both models favour precision over recall which in the medical domain is usually preferable as it means we are catching a majority of the cases where an individual is suffering from a given ailment. As with accuracy, the BCNN outperforms the CNN in both recall and precision, reinforcing the notion that on unperturbed images, a BCNN offers greater predictive power.

# Adversaries in Real World Images
We'll now examine how a BCNN performs on a real world dataset that undergoes adversarial attacks to investigate to what degree the high benchmark set by the BCNN on unperturbed images can be retained across adversarial perturbed images.





__Add as a contribution we have provided an extensive empirical Bayesian analysis into the effect of uncertainty in detecting adversaries__.
