.. adversary:

Adversary
=====================================================
At present, the adversary has just a single pertubation method - FGSM. This works by computing the sign function of the derivative of network's cost function over the original image.

.. figure:: noise.png
    :width: 500px
    :align: center
    :height: 200px
    :alt: alternate text
    :figclass: align-center

.. automodule:: adversary
  :members: