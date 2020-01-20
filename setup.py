# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import setuptools, platform

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires=[
    'warmup-scheduler @ git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git@v0.2',
    'pystopwatch2 @ git+https://github.com/ildoonet/pystopwatch2.git',
    'hyperopt', #  @ git+https://github.com/hyperopt/hyperopt.git
    'tensorwatch @ git+https://github.com/microsoft/tensorwatch.git',
    'pretrainedmodels', 'tqdm', 'sklearn', 'matplotlib', 'psutil',
    'requests',
    'gorilla', 'pyyaml', 'overrides', 'runstats'
]

if platform.system()!='Windows':
    install_requires.append('ray')

setuptools.setup(
    name="archai",
    version="0.3.0",
    author="Shital Shah, Debadeepta Dey,",
    author_email="shitals@microsoft.com, dedey@microsoft.com",
    description="Research plateform for Network Architecture Search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sytelus/archai",
    packages=setuptools.find_packages(),
	license='MIT',
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    include_package_data=True,
    install_requires=install_requires
)
