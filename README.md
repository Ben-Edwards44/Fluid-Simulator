# Fluid Simulator

A particle based fluid simulator written in python.

## About

This project was heavily inspired by [this video](https://www.youtube.com/watch?v=rSKMYc1CQHE) by Sebastian Lague.
This fluid simulator works in a very similar way, and uses similar techniques to, the one showcased in the video.

This simulator uses smoothed particle hydrodynamics (SPH) technique, which is described very well in [this paper](http://www.ligum.umontreal.ca/Clavet-2005-PVFS/pvfs.pdf).

In an attempt to increase performance and allow the simulation of a larger number of particles (as Python is notoriously slow), I have made use of the [Numba Python library](https://numba.pydata.org/) and the GPU acceleration it provides.

## Requirements

- A supported GPU (listed [here](https://numba.readthedocs.io/en/stable/cuda/overview.html#supported-gpus)).
- Python 3.11 (slightly eariler or later versions of Python, such as 3.9 or 3.10 may also work).
- The [Numpy](https://numpy.org/), [Numba](https://numba.pydata.org/) and [Pygame](https://www.pygame.org/news) libraries.
- A library made by me called [Pygame UI Toolkit](https://github.com/Ben-Edwards44/pygame-ui-toolkit).

## Possible improvements

- Improve performance by implementing a bitonic sort that can run on the GPU.
- Optimise thread and block sizes.
- Viscoelasiticty.
- More accurate surface tension.
- Interactions with objects.
- Finish the README.md file.