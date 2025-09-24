# __init__.py
from .param_utils import get_params_space_and_org, make_param_dict_unique
from .clip_utils import my_clip, smart_clip
from .seed_utils import set_seed
from .batch_utils import get_max_batch_size_for_cuda
from .time_utils import TimeRecorder, time_start, log_time_delta
from .result_utils import (
    get_valid_params_mode_data, find_top_param_dict_list, find_top_param_dict_list_pareto,
    calc_statistics, find_top_param_dict_list_by_statistics, find_top_param_dict_list_pareto_by_statistics,
    find_result_by_param_dict, calc_improve_percent1, calc_improve_percent2, calc_improve_percent_statistics,
    calc_better_draw_percent, calc_improve_percent_in_better_and_worse, calc_improve_percent_in_hard_medium_easy
)
from .plotting_utils import many_plot, plot_hpo_progress, plot_metric_hist_comparison
from .exit_utils import atexit_handler