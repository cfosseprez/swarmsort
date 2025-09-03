Installation
============

Requirements
------------

SwarmSort requires Python 3.9 or later and supports Linux, Windows, and macOS.

Install from PyPI
-----------------

The easiest way to install SwarmSort is via pip:

.. code-block:: bash

   pip install swarmsort

with GPU embedding support:

.. code-block:: bash

   pip install swarmsort[gpu]


Install from Source
-------------------

For the latest development version:

.. code-block:: bash

   git clone https://github.com/cfosseprez/swarmsort.git
   cd swarmsort
   pip install -e .

Development Installation
------------------------

If you want to contribute to SwarmSort:

.. code-block:: bash

   git clone https://github.com/cfosseprez/swarmsort.git
   cd swarmsort
   
   # Using Poetry (recommended)
   poetry install --with dev
   
   # Or using pip
   pip install -e ".[dev]"

Optional Dependencies
---------------------

**GPU Acceleration**

For GPU support, install CuPy according to your CUDA version:

.. code-block:: bash

   # For CUDA 11.x
   pip install cupy-cuda11x
   
   # For CUDA 12.x
   pip install cupy-cuda12x

**Visualization**

For visualization capabilities:

.. code-block:: bash

   pip install matplotlib

Verify Installation
-------------------

Check that SwarmSort is correctly installed:

.. code-block:: python

   import swarmsort
   print(swarmsort.__version__)
   
   # Check GPU availability
   from swarmsort import is_gpu_available
   print(f"GPU available: {is_gpu_available()}")