Welcome to Adversary Detection Using Bayesian Approximations's documentation!
=============================================================================

Note
-----------------------------------------------------
This is still a working project and documentation is in it's skeleton form and therefore may be missing code coverage or code may be very loosely documented in places.

Adversary
-----------------------------------------------------

While libraries such as Cleverhans and ART are able to simulate the effects of adversarial attacks, the abstractiveness removes the ability to visualise the noise being used to perturb an image. For this reason, FGSM was implemented in PyTorch below.

.. toctree::
   :maxdepth: 2

   adversary

Computer Vision Models
-----------------------------------------------------

For details and documentation surrounding computer vision based adversarial testing and Bayesian based classification:

.. toctree::
   :maxdepth: 2

   vision

Reinforcement Learning
-----------------------------------------------------
Alternatively, a DQN has been trained to play FlappyBird with the effects of adversarial attacks upon the trained policy tested:

.. toctree::
   :maxdepth: 2

   rl
