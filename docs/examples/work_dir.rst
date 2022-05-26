.. _work_dir:

Working Directory
=================

HostPhot relies on the same working directory accross different modules and functions. By default, this is set to ``images`` (under your current directory):

.. code:: python

	import hostphot

	print('Hostphot version:', hostphot.__version__)
	print('Working directory:', hostphot.__workdir__)

.. code:: python
	
	Hostphot version: 2.0.0
	Working directory: images

If you want to change the working directory, this can be done with the following function:

.. code:: python

	hostphot.choose_workdir('/home/user/project/images')
	print('Working directory:', hostphot.__workdir__)

.. code:: python
	
	Working directory: /home/user/project/images

Note that this needs to be done every time you run a script/notebook as the default will always be ``images``.
