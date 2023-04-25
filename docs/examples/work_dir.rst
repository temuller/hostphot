.. _work_dir:

Working Directory
=================

HostPhot relies on the same working directory accross different modules and functions. This is set to ``images`` by default (under your current directory):

.. code:: python

	import hostphot

	print('Hostphot version:', hostphot.__version__)
	print('Working directory:', hostphot.workdir)

.. code:: python
	
	Hostphot version: 2.5.0
	Working directory: images
	
	
If the user wants to change the HostPhot working directory, a ``.env`` file must be created under the current directory and must have the following content:


.. code::

	workdir = "path/to/new/workdir"
	
where ``path/to/new/workdir`` will be the new working directory for HostPhot:

.. code:: python

	print('Working directory:', hostphot.workdir)

.. code:: python
	
	Working directory: path/to/new/workdir
