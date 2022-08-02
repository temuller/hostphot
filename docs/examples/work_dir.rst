.. _work_dir:

Working Directory
=================

HostPhot relies on the same working directory accross different modules and functions. This is set to ``images`` (under your current directory):

.. code:: python

	import hostphot

	print('Hostphot version:', hostphot.__version__)
	print('Working directory:', hostphot.__workdir__)

.. code:: python
	
	Hostphot version: 2.0.0
	Working directory: images
