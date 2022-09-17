# MD

For the time being, I just wrote a simple molecular dynamics code for CPU as well as equivalent GPU kernels in Numba to speed it up. Molecular Dynamics is one of the areas that significantly benefit from specialized hardware (not that GPU is specialized hardware.)

First pip install the requirement file and then you should be able to run the notebook.

The outputs are generated in the LAMMPS dump file format. You can open them and view the dynamics with opensource software such as OVITO.

In this particular example I am achieving upto 10X speedup!

Note: Part of the original python code was stolen from someone on the internet but I don't remember who to give credit to.
