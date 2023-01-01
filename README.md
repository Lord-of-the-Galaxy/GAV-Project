# GAV-Project
Course project for [E0 271: Graphics and Visualization](https://www.csa.iisc.ac.in/~vijayn/courses/Graphics/index.html). Based on [Graying the Black Box: Understanding DQNs](https://arxiv.org/abs/1602.02658).

![Screenshot of tool](SAMDP_VIS_TOOL.png?raw=true)

## Requirements and Installation

It is recommended to create two separate environments using [mamba](https://mamba.readthedocs.io/)/[conda](https://docs.conda.io/en/latest/) for the data generation and visualization. In the data generation environment, you must install [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) with Atari support (with `pip`, the following command may be sufficient: `pip install git+https://github.com/DLR-RM/stable-baselines3#egg=stable-baselines3[extra]`), along with [tqdm](https://tqdm.github.io/) and [rich](https://rich.readthedocs.io/). In the visualization environment, you must install [RAPIDS](https://rapids.ai/index.html) [cuML](https://github.com/rapidsai/cuml) along with [matplotlib](https://matplotlib.org/) and [scikit-learn](https://scikit-learn.org/stable/index.html).  

Then clone the repository, and create a `data` and a `models` folder. Download the trained `DQN` models for any of the (available) Atari games (or use your own, but with the same naming scheme) from [Stable Baselines3 on Hugging Face](https://huggingface.co/sb3) and place them in the `models` folder (without modifying the names).

## Usage

First, generate the data using the data generation environment by running `python generate_data.py GAME [GAME ...]` where `GAME` is any game for which you have downloaded the model; please run `python generate_data.py --help` for more information. To generate and visualize the tSNE plot and SAMDP model, switch to the visualization environment. First ensure that you have generated data for all the four games listed at the top of `control_buttons.py` (or modify those variables to make it so). Next, ensure that the `SEED`s listed in `control_buttons.py` match the actual seeds for the generated data.

Now you can run `python VIS_TOOL.py` to run the visualization tool. The `F/W` and `B/W` buttons step through a single episode (jumping to a new episode at the points where a new episode begins). Clicking on any point in the tSNE plot selects and displays that state instead. Click on the buttons for any of the four games to load the data for that game and generate a (fresh) tSNE plot for the same. You may change the perplexity values with the slider. Click on SAMDP to generate an SAMDP model and display it on top of the tSNE plot. Clicking on the Manual button gets rid of the SAMDP, and must be clicked before trying to generate a new SAMDP model for the same tSNE plot. Note that clicking the SAMDP button runs a grid search over the two parameters (number of clusters and window size) and thus takes some time to run.
