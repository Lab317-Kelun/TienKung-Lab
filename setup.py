# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from distutils.core import setup

from setuptools import find_packages

# PyTorch with CUDA 12.4
# Note: Install PyTorch with CUDA 12.4 using:
# pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

setup(
    name="LeggedLab",
    packages=find_packages(),
    version="0.1.0",
    install_requires=[
        "IsaacLab",
        "torch>=2.7",
        "torchvision>=0.21.0",
        "pynput",
        "mujoco==3.3.2",
        "mujoco-python-viewer",
        "matplotlib",
    ],
    dependency_links=[
        "https://download.pytorch.org/whl/cu128",
    ],
)
