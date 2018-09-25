# ClusterWork

A framework to easily deploy experiments on an computing cluster with mpi. 
**ClusterWork** is based on the [Python Experiment Suite](https://github.com/rueckstiess/expsuite) by Thomas Rückstiess and uses the [job_stream](https://wwoods.github.io/job_stream/) package to distribute the work.

The **PlotWork** extension for [IPython](https://ipython.org/) allows for easy visualization of of experiment results during run time or after the experiment has finished using [Jupyter](https://jupyter.org/).

## Installation

0. Creating a virtual environment with [virtualenv](https://virtualenv.pypa.io/en/stable/) or [conda](https://conda.io/miniconda.html) is recommended. For the installation this virtual environment has to be activated first.
1. Install required packages
    1. [job_stream](https://wwoods.github.io/job_stream/) requires [boost](http://www.boost.org/) (filesystem, mpi, python, regex, serialization, system, thread) and mpi (e.g., [OpenMPI](http://www.open-mpi.org/))
       ```sh
       sudo apt-get install libboost-dev libopenmpi-dev
       ```
       In case, you have created a conda environment, both can also be installed in the environment as
       ```bash
       conda install boost mpi4py
       ```
       Currently, you need to manually fix a bug in a boost header, see [here](https://github.com/wwoods/job_stream/issues/1). 
    2. **ClusterWork** requires the Python packages PyYAML, job_stream and pandas
       ```sh
       pip install PyYAML job_stream pandas
       ```
       However, they should be automatically installed when installying **ClusterWork**
2. Clone this repository and install it
```sh
git clone https://github.com/gregorgebhardt/cluster_work cluster_work
cd cluster_work
pip install .
```

## Get your code on the computing cluster
Running your code on the computing cluster is now a very simple task. 
Currently, this requires the following three steps:

1. Write a Python class that inherits from **ClusterWork** and implements at least the methods `reset(self, config: dict, rep: int)` and `iterate(self, config: dict, rep: int, n: int)`.
2. Write a simple YAML-file to configure your experiment. 
3. Adopt a shell script that starts the experiment on your cluster.

### Subclassing ClusterWork

```Python
from cluster_work import ClusterWork

class MyExperiment(ClusterWork):
    # ...

    def reset(self, config=None, rep=0):
        # run code that sets up your experiment for each repetition here
        pass

    def iterate(self, config=None, rep=0, n=0):
        # run your experiment for iteration n
        # return results as a dictionary, for each key there will be one column in a results table.
        pass


# to run the experiments, you simply call run on your derived class
if __name__ == '__main__':
    MyExperiment.run()
```

#### Restarting your experiments

**ClusterWork** also implement a restart functionality. Since your results are stored after each iteration, your experiment can be restarted if its execution was interrupted for some reason. To obtain this functionality, you need to implement in addition at least the method `restore_state(self, config: dict, rep: int, n: int)`. Additionally, the method `save_state(self, config: dict, rep: int, n: int)` can be implemented to store additional information the needs to be loaded in the `restore_state` method. Finally, a flag `_restore_supported` must be set to `True`.

```Python
class MyExperiment(ClusterWork):
    _restore_supported = True

    # ...

    def save_state(self, config: dict, rep: int, n: int):
        # save all the necessary information for restoring the state later
        pass

    def restore_state(self, config: dict, rep: int, n: int):
        # load or reconstruct the state at repetition rep and iteration n
        pass
```

#### Default parameters

The parameters for the experiment can be defined in an YAML-file that is passed as an command-line argument. Inside the derived class, we can define default parameters as a dictionary in the `_default_params` field:

```Python
class MyExperiment(ClusterWork):
    # ...

    _default_params = {
        # ...
        'num_episodes': 100,
        'num_test_episodes': 10,
        'num_eval_episodes': 10,
        'num_steps': 30,
        'num_observations': 30,

        'optimizer_options': {
            'maxiter': 100
        },
        # ...
    }

    # ...
```

### The Configuration YAML

To configure the execution of the experiment, we need to write a small YAML-file. The YAML file consists several documents which are separated by a line of `---`. Optionally, the first document can be made a default by setting the key `name` to `"DEFAULT"`. This default document will then form the basis for all following experiment documents. Besides the optional default document, each document represents an experiment. However, experiments can be expanded by the _list_ __or__ _grid_ feature, which is explained below.

The required keys for each experiment are `name`, `repetitions`, `iterations`, and `path`. The parameters found below the key `params` overwrite the default parameters defined in the experiment class. Since the `config` dictionary that is passed to the methods of the **ClusterWork** subclass is the full configuration generated from the YAML-file and the default parameters, additional keys can be used.

```
# default document denoted by the name "DEFAULT"
name: "DEFAULT"
repetitions: 20
iterations: 5
# this is the path where the results are stored,
# it can be different from the location of the experiment scripts
path: "path/to/experiment/folder"

params:
    num_episodes: 150
    optimizer_options:
        maxiter: 50
---
# 1. experiment
name: "more_test_episodes"
params:
    num_test_episodes: False
---
# 2. experiment
name: "more_steps"
params:
    num_steps: 50
```

#### The list feature

If the key `list` is given in an experiment document, the experiment will be expanded for each value in the given list. For example

```
# ...
---
name: "test_parameter_a"
params:
    b: 5
list:
    a: [5, 10, 20, 30]
```

creates four experiments, one for each value of `a`. It is also possible define multiple parameters below the `list` key:

```
# ...
---
name: "test_parameter_a_and_c"
params:
    b: 5
list:
    a: [5, 10, 20, 30]
    c: [1, 2, 3, 4, 5]
```

In this case the experiment is run for the four parameter combinations `{a: 5, c: 1}`, `{a: 10, c: 2}`, ..., `{a: 30, c: 4}`. Since the list for `a` is shorter than the list for `c`, the remaining values for `c` are ignored.

#### The grid feature

The `grid` feature is similar to the `list` feature, however instead of iterating over all lists jointly it spans a grid with all the values given. For example

```
# ...
---
name: "test_parameters_foo_and_bar"
params:
    a: 5
grid:
    foo: [5, 10, 20, 30]
    bar: [1, 2, 3, 4, 5]
```

would run an experiment for each combination of `foo` and `bar`, in this case 4x5=20 experiments. Note that with more parameters below the `grid` key the number of experiments explodes exponentially.

### Run the experiment

To run your experiment, you can simply execute your python script:

```
python YOUR_SCRIPT.py YOUR_CONFIG.yml [arguments]
```

The following arguments are available:

+ `-c, --cluster` runs the experiments using the job_stream scheduler. By default the experiments are executed sequentially in a loop.
+ `-d, --delete` delete old results before running your experiments.
+ `-o, --overwrite` overwrite results if configuration has changed.
+ `-e, --experiments [EXPERIMENTS]` chooses the experiments that should run, by default all experiments will run.
+ `-v, --verbose` shows more output
+ `-p, --progress` displays only the progress of running or finished experiments.
+ `-P, --full_progress` displays the detailed (i.e., of each repetition) progress of running or finished experiments.
+ `-l, --log_level [LEVEL]` displays the detailed (i.e., of each repetition) progress of running or finished experiments.

#### Hostfile for OpenMPI

Depending on the configuration of your cluster, `job_stream` might not know about the available computing units in the cluster. In this case it is possible to pass a hostfile that specifies hostnames and available cpus to job_stream before starting the experiment script.

```sh
job_stream --hostfile HOSTFILE -- python YOUR_SCRIPT.py YOUR_CONFIG.yml [arguments]
```

The hostfile should have the following form:

```text
host0
host1 cpus=2
host2 cpus=3
```

For a SLURM-based cluster, this hostfile can be created by

```
srun hostname > hostfile.$SLURM_JOB_ID
hostfileconv hostfile.$SLURM_JOB_ID
```

where `hostfileconv` is a tool provided by **ClusterWork** that makes sure the hostfile has the right format. In this case the `HOSTFILE` argument for job_stream would be `hostfile.$SLURM_JOB_ID.converted`. 