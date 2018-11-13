"""
Idea:
implement magic that takes the path to the experiment configuration


plotting function:
- iteration plots
    * learner, config, repetition, iteration
- results plots
    * config, results

"""
import os
import warnings
from collections import namedtuple, OrderedDict
from functools import reduce
from itertools import cycle
from typing import Callable, List, Union

import matplotlib.pyplot as plt
from IPython.core.getipython import get_ipython
from IPython.core.magic import register_line_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from IPython.display import display, FileLink
from ipywidgets import Box, FloatProgress, HBox, Label, Layout, Output, Tab, VBox
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame, Series

from cluster_work import ClusterWork

__iteration_plot_functions = {}
__results_plot_functions = {}
__file_provider_functions = {}

__experiment_class: ClusterWork = None
__experiment_config = None
__experiment_selectors = None
__instantiated_experiments = None


def register_iteration_plot_function(name: str):
    def register_iteration_plot_function_decorator(plot_function: Callable[[ClusterWork, int, int, List], Figure]):
        global __iteration_plot_functions
        __iteration_plot_functions[name] = plot_function
        return plot_function
    return register_iteration_plot_function_decorator


def register_results_plot_function(name: str):
    def register_results_plot_function_decorator(plot_function: Callable[[str, DataFrame, plt.Axes], None]):
        global __results_plot_functions
        __results_plot_functions[name] = plot_function
        return plot_function
    return register_results_plot_function_decorator


DownloadFile = namedtuple('DownloadFile', ['path', 'file_name', 'link_text'])


def register_file_provider(name: str):
    def register_file_provider_decorator(file_provider_function: Callable[[ClusterWork, dict, list],
                                                                          Union[DownloadFile, List[DownloadFile]]]):
        global __file_provider_functions
        __file_provider_functions[name] = file_provider_function
        return file_provider_function
    return register_file_provider_decorator


@register_line_magic
def set_experiment_class(line: str):
    global __experiment_class
    __experiment_class = get_ipython().ev(line)


@register_line_magic
@magic_arguments()
@argument('config', type=str)
@argument('-e', '--experiments', nargs='*')
@argument('-f', '--filter', nargs='*', help='filter strings that are applied on the experiment names')
def load_experiment(line: str):
    # TODO add tab completion for file
    # read line, split at white spaces load experiments with selectors
    args = parse_argstring(load_experiment, line)
    # experiment_config = splits.pop(0)
    # experiment_selectors = splits

    # check if experiment config exists and load experiments
    if not os.path.exists(args.config):
        raise Warning('path does not exist: {}'.format(args.config))
    else:
        global __experiments, __experiment_config, __experiment_selectors
        __experiment_config = args.config
        __experiment_selectors = args.experiments

        with open(__experiment_config, 'r') as f:
            __experiments = __experiment_class.load_experiments(f, __experiment_selectors)

        if args.filter is not None:
            __experiments = list(filter(lambda c: all([_f in c['name'] for _f in args.filter]), __experiments))
        else:
            __experiments = __experiments

        get_ipython().user_ns['experiments'] = __experiments


@register_line_magic
@magic_arguments()
@argument('-r', '--repetition', type=int, help='the repetition', default=0)
@argument('-i', '--iteration', type=int, help='the iteration to plot', default=0)
def restore_experiment_state(line: str):
    args = parse_argstring(restore_experiment_state, line)
    global __instances, __instantiated_experiments
    __instances = list()
    __instantiated_experiments = list()

    with Output():
        for exp in get_ipython().user_ns['experiments']:
            exp_instance = __experiment_class.init_from_config(exp, args.repetition, args.iteration)
            if exp_instance:
                __instances.append(exp_instance)
                __instantiated_experiments.append(exp)

    get_ipython().user_ns['experiment_instances'] = __instances


@register_line_magic
@magic_arguments()
@argument('column')
def restore_best_experiment_state(line: str):
    args = parse_argstring(restore_best_experiment_state, line)
    global __instances, __instantiated_experiments, __experiments
    __instances = list()
    __instantiated_experiments = list()

    experiment_results = [ClusterWork.load_experiment_results(exp) for exp in __experiments]
    best_results_idx = []

    with Output():
        for exp, result in zip(__experiments, experiment_results):
            result_column = result[args.column]
            r, i = result_column.idxmax()
            best_results_idx.append((r, i))
            exp_instance = __experiment_class.init_from_config(exp, r, i)
            if exp_instance:
                __instances.append(exp_instance)
                __instantiated_experiments.append(exp)

    get_ipython().user_ns['best_results_idx'] = best_results_idx
    get_ipython().user_ns['experiment_instances'] = __instances


@register_line_magic
@magic_arguments()
@argument('plotter_name', type=str, help='the name of the plotter function')
@argument('--save_figures', action='store_true', help='store the figures to files')
@argument('--prefix', type=str, help='add a prefix to the filename', default='')
@argument('--format', type=str, help='format to store the figure in', default=None)
@argument('--tab_title', type=str, help="Choose columns from config for tab titles", default=None)
@argument('args', nargs='*', help='extra arguments passed to the filter function')
def plot_iteration(line: str):
    """call a registered plotter function for the given repetition and iteration"""
    args = parse_argstring(plot_iteration, line)

    items = []

    from ipywidgets.widgets.interaction import show_inline_matplotlib_plots

    global __instances, __instantiated_experiments
    for exp_instance, exp_config in zip(__instances, __instantiated_experiments):
        out = Output()
        items.append(out)
        with out:
            # clear_output(wait=True)
            figures = __iteration_plot_functions[args.plotter_name](exp_instance, args.args)
            show_inline_matplotlib_plots()
            if args.save_figures:
                if args.format is None:
                    args.format = plt.rcParams['savefig.format']
                os.makedirs('plots/{}'.format(exp_config['name']), exist_ok=True)
                for i, f in enumerate(figures):
                    filename = 'plots/{}/{}figure_{}.{}'.format(exp_config['name'], args.prefix, i, args.format)
                    if args.format == 'tikz':
                        try:
                            from matplotlib2tikz import save as tikz_save
                            with Output():
                                tikz_save(filename, figureheight='\\figureheight', figurewidth='\\figurewidth')
                        except ModuleNotFoundError:
                            warnings.warn('Saving figure as tikz requires the module matplotlib2tikz.')
                    else:
                        f.savefig(filename, format=args.format)

    if len(items) > 1:
        tabs = Tab(children=items)
        for i, exp in enumerate(__instantiated_experiments):
            if args.tab_title:
                if (args.tab_title[0] == args.tab_title[-1]) and args.tab_title.startswith(("'", '"')):
                    selectors = args.tab_title[1:-1]
                else:
                    selectors = args.tab_title
                selectors = selectors.split(' ')
                values = [reduce(lambda a, b: a[b], [exp['params'], *selector.split('.')]) for selector in
                          selectors]
                tabs.set_title(i, ' '.join(map(str, values)))
            else:
                tabs.set_title(i, '...' + exp['name'][-15:])
        display(tabs)
    elif len(items) == 1:
        return items[0]
    else:
        warnings.warn('No plots available for {} with args {}'.format(args.plotter_name, args.args))


def __plot_iteration_completer(_ipython, _event):
    return __iteration_plot_functions.keys()


@register_line_magic
@magic_arguments()
@argument('plotter_name', type=str, help='the name of the plotter function')
@argument('column', type=str, help='column of the results DataFrame to plot')
@argument('--save_figures', action='store_true', help='store the figures to files')
@argument('--prefix', type=str, help='add a prefix to the filename', default='')
@argument('--format', type=str, help='format to store the figure in', default=None)
@argument('-i', '--individual', action='store_true', help='plot each experiment in a single axes object')
def plot_results(line: str):
    args = parse_argstring(plot_results, line)

    # global __experiments, __results_plot_functions
    config_results = [(config, ClusterWork.load_experiment_results(config)) for config in __experiments]
    config_results = list(map(lambda t: (t[0], t[1][args.column]), filter(lambda t: t[1] is not None, config_results)))

    global __experiment_selectors
    for selector in __experiment_selectors:
        selected_config_results = list(filter(lambda t: t[0]['name'].startswith(selector), config_results))
        f = plt.figure()
        if args.individual:
            axes = f.subplots(len(selected_config_results), 1)
        else:
            axes = [f.subplots(1, 1)] * len(selected_config_results)

        for config_result, ax in zip(selected_config_results, axes):
            config, result = config_result
            # ax.set_xlim(0, config['iterations'])
            ax.set_title('Results')
            ax.set_xlabel('iterations')
            ax.set_ylabel(args.column)
            __results_plot_functions[args.plotter_name](config['name'], result, ax)

        if args.save_figures:
            if args.format is None:
                args.format = plt.rcParams['savefig.format']
            filename = 'plots/{}{}_figure.{}'.format(args.prefix, selector, args.format)
            if args.format == 'tikz':
                try:
                    from matplotlib2tikz import save as tikz_save
                    with Output():
                        tikz_save(filename, figureheight='\\figureheight', figurewidth='\\figurewidth')
                except ModuleNotFoundError:
                    warnings.warn('Saving figure as tikz requires the module matplotlib2tikz.')
            else:
                f.savefig(filename, format=args.format)


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


class DownloadFileLink(FileLink):
    html_link_str = "<a href='{link}' download={file_name}>{link_text}</a>"

    def __init__(self, path, file_name=None, link_text=None, *args, **kwargs):
        super(DownloadFileLink, self).__init__(path, *args, **kwargs)

        self.file_name = file_name or os.path.split(path)[1]
        self.link_text = link_text or self.file_name

    def _format_path(self):
        from html import escape
        fp = ''.join([self.url_prefix, escape(self.path)])
        return ''.join([self.result_html_prefix,
                        self.html_link_str.format(link=fp, file_name=self.file_name, link_text=self.link_text),
                        self.result_html_suffix])


@register_line_magic
@magic_arguments()
@argument('file_provider', type=str, help='name of the file provider function')
@argument('--tab_title', type=str, help="Choose columns from config for tab titles", default=None)
@argument('args', nargs='*', help='extra arguments passed to the filter function')
def provide_files(line: str):
    args = parse_argstring(provide_files, line)

    items = []

    ipy = get_ipython()
    url_prefix = os.path.relpath(os.getcwd(), ipy.starting_dir) + os.path.sep
    print(url_prefix)

    global __file_provider_functions
    global __instances, __instantiated_experiments
    for exp_instance, exp_config in zip(__instances, __instantiated_experiments):
        dfs = __file_provider_functions[args.file_provider](exp_instance, exp_config, args.args)
        if isinstance(dfs, DownloadFile):
            items.append(Output())
            with items[-1]:
                display(DownloadFileLink(dfs.path, file_name=dfs.file_name, link_text=dfs.link_text,
                                         url_prefix=url_prefix))
        else:
            items.append(VBox([DownloadFileLink(df.path, file_name=df.file_name, link_text=df.link_text,
                                                url_prefix=url_prefix) for df in dfs]))

    if len(items) > 1:
        tabs = Tab(children=items)
        for i, exp in enumerate(__instantiated_experiments):
            if args.tab_title:
                if (args.tab_title[0] == args.tab_title[-1]) and args.tab_title.startswith(("'", '"')):
                    selectors = args.tab_title[1:-1]
                else:
                    selectors = args.tab_title
                selectors = selectors.split(' ')
                values = [reduce(lambda a, b: a[b], [exp['params'], *selector.split('.')]) for selector in selectors]
                tabs.set_title(i, ' '.join(map(str, values)))
            else:
                tabs.set_title(i, '...' + exp['name'][-15:])
        display(tabs)
    elif len(items) == 1:
        return items[0]
    else:
        warnings.warn('No files loaded')


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


_line_styles = OrderedDict(
    [('solid', (0, ())),

     ('densely dotted', (0, (1, 1))),
     ('densely dashed', (0, (5, 1))),
     ('densely dashdotted', (0, (3, 1, 1, 1))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),

     ('dashed', (0, (5, 5))),
     ('dotted', (0, (1, 5))),
     ('dashdotted', (0, (3, 5, 1, 5))),
     ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),

     ('loosely dotted', (0, (1, 10))),
     ('loosely dashed', (0, (5, 10))),
     ('loosely dashdotted', (0, (3, 10, 1, 10))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ])

_line_style_cycles = dict()


def line_style_cycle(axes: Axes):
    if axes in _line_style_cycles:
        return _line_style_cycles[axes]
    else:
        _line_style_cycles[axes] = cycle(_line_styles)
        return _line_style_cycles[axes]


@register_results_plot_function('mean_2std')
def plot_mean_2std(name: str, results_df: Series, axes: Axes):
    mean = results_df.groupby(level=1).mean()
    std = results_df.groupby(level=1).std()

    ls_name = next(line_style_cycle(axes))
    ls_def = _line_styles[ls_name]

    axes.plot(mean.index, results_df.unstack(level=0), c='grey', ls=ls_def, alpha=.5)

    axes.fill_between(mean.index, mean - 2 * std, mean + 2 * std, alpha=.5)
    axes.plot(mean.index, mean, label=name, ls=ls_def)
    axes.legend()


del register_line_magic
