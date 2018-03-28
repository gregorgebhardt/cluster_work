"""
Idea:
implement magic that takes the path to the experiment configuration


plotting function:
- iteration plots
    * learner, config, repetition, iteration
- results plots
    * config, results

"""
from cluster_work import ClusterWork

from IPython.core.getipython import get_ipython
from IPython.display import display, clear_output
from IPython.core.magic import register_line_magic
from IPython.core.magic_arguments import magic_arguments, argument, parse_argstring
from ipywidgets import Accordion, FloatProgress, Box, HBox, Label, Layout, Output, Widget

from typing import Callable, Union
from matplotlib.figure import Figure
from pandas import DataFrame

import os


__iteration_plot_functions = {}
__results_plot_functions = {}

__experiment_class: ClusterWork = None
__experiment_config = None
__experiment_selectors = None
__experiments: dict = None


def register_iteration_plot_function(name: str):
    def register_iteration_plot_function_decorator(plot_function: Callable[[ClusterWork, dict, int, int],
                                                                           Union[Figure, Widget]]):
        global __iteration_plot_functions
        __iteration_plot_functions[name] = plot_function
        return plot_function
    return register_iteration_plot_function_decorator


def register_results_plot_function(name: str):
    def register_results_plot_function_decorator(plot_function: Callable[[DataFrame], Union[Figure, Widget]]):
        global __results_plot_functions
        __results_plot_functions[name] = plot_function
        return plot_function
    return register_results_plot_function_decorator


@register_line_magic
def set_experiment_class(line: str):
    global __experiment_class
    __experiment_class = get_ipython().ev(line)


@register_line_magic
def load_experiment(line: str):
    # TODO add tab completion for file
    # read line, split at white spaces load experiments with selectors
    splits = line.split()
    experiment_config = splits.pop(0)
    experiment_selectors = splits

    # check if experiment config exists and load experiments
    if not os.path.exists(experiment_config):
        raise Warning('path does not exist: {}'.format(experiment_config))
    else:
        global __experiments, __experiment_config, __experiment_selectors
        __experiment_config = experiment_config
        __experiment_selectors = experiment_selectors
        with open(experiment_config, 'r') as f:
            __experiments = __experiment_class.load_experiments(f, experiment_selectors)


@register_line_magic
@magic_arguments()
@argument('plotter_name', type=str, help='the name of the plotter function')
@argument('repetition', type=int, help='the repetition')
@argument('iteration', type=int, help='the iteration to plot')
@argument('filter', nargs='*', help='filter strings that are applied on the experiment names')
def plot_iteration(line: str):
    """call a registered plotter function for the given repetition and iteration"""
    args = parse_argstring(plot_iteration, line)

    if args.filter is not None:
        filtered_experiments = list(filter(lambda c: all([f in c['name'] for f in args.filter]), __experiments))
    else:
        filtered_experiments = __experiments

    items = []

    for exp in filtered_experiments:
            learner = __experiment_class.init_from_config(exp, args.repetition, args.iteration)
            fw = __iteration_plot_functions[args.plotter_name](learner, exp)
            clear_output()
            if isinstance(fw, Figure):
                out = Output()
                items.append(out)
                with out:
                    clear_output(wait=True)
                    display(fw)
            else:
                items.append(fw)

    accordion = Accordion(children=items)
    for i, exp in enumerate(filtered_experiments):
        accordion.set_title(i, exp['name'])
    return accordion

    # return items[0]


def __plot_iteration_completer(ipython, event):
    return __iteration_plot_functions.keys()


def __create_exp_progress_box(name, exp_progress, rep_progress, show_full_progress=False):
    exp_progress_layout = Layout(display='flex', flex_flow='column', align_items='stretch', width='100%')
    exp_progress_bar = HBox([FloatProgress(value=exp_progress, min=.0, max=1., bar_style='info'), Label(name)])

    if show_full_progress:
        rep_progress_layout = Layout(display='flex', flex_flow='column', align_items='stretch',
                                     align_self='flex-end', width='80%')

        items = [FloatProgress(value=p, min=.0, max=1., description=str(i)) for i, p in enumerate(rep_progress)]
        rep_progress_box = Box(children=items, layout=rep_progress_layout)

        return Box(children=[exp_progress_bar, rep_progress_box], layout=exp_progress_layout)
    else:
        return exp_progress_bar


@register_line_magic
def show_progress(line: str):
    show_full_progress = line == 'full'

    global __experiment_config, __experiment_selectors

    with open(__experiment_config, 'r') as f:
        total_progress, experiments_progress = ClusterWork.get_progress(f, __experiment_selectors)

    box_layout = Layout(display='flex', flex_flow='column', align_items='stretch', widht='100%')
    items = [__create_exp_progress_box(*progress, show_full_progress) for progress in experiments_progress]

    total_progress_bar = FloatProgress(value=total_progress, min=.0, max=1., description='Total', bar_style='success')

    return Box(children=items + [total_progress_bar], layout=box_layout)


def load_ipython_extension(ipython):
    global __iteration_plot_functions, __results_plot_functions
    # ipython.push(['__experiment_config', '__experiment_selectors'])

    ipython.set_hook('complete_command', __plot_iteration_completer, re_key='%plot_iteration')
    # ipython.set_hook('complete_command', lambda e: __results_plot_functions.keys(), re_key='%plot_results')


del register_line_magic
