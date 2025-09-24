

from utils.param_utils import get_params_space_and_org


def ablation_substitute(kwargs, ablation):
    params_space, origin_param_dict = get_params_space_and_org()
    if ablation == 'Trimmer' or ablation == 'Context':
        kwargs['trimmer_seq_len'] = origin_param_dict['trimmer_seq_len']
    elif ablation == 'Sampler' or ablation == 'Context':
        kwargs['sampler_factor'] = origin_param_dict['sampler_factor']
    elif ablation == 'Aligner' or ablation == 'Context':
        kwargs['aligner_mode'] = origin_param_dict['aligner_mode']
        kwargs['aligner_method'] = origin_param_dict['aligner_method']

    elif ablation == 'Normalizer' or ablation == 'Range':
        kwargs['normalizer_method'] = origin_param_dict['normalizer_method']
        kwargs['normalizer_mode'] = origin_param_dict['normalizer_mode']
        kwargs['normalizer_ratio'] = origin_param_dict['normalizer_ratio']
    elif ablation == 'Warper' or ablation == 'Range':
        kwargs['warper_method'] = origin_param_dict['warper_method']
    elif ablation == 'Differentiator' or ablation == 'Range':
        kwargs['differentiator_n'] = origin_param_dict['differentiator_n']

    elif ablation == 'Inputer' or ablation == 'Anomaly':
        kwargs['inputer_detect_method'] = origin_param_dict['inputer_detect_method']
        kwargs['inputer_fill_method'] = origin_param_dict['inputer_fill_method']
    elif ablation == 'Denoiser' or ablation == 'Anomaly':
        kwargs['denoiser_method'] = origin_param_dict['denoiser_method']
    elif ablation == 'Clipper' or ablation == 'Anomaly':
        kwargs['clip_factor'] = origin_param_dict['clip_factor']

    elif ablation == 'Pipeline':
        kwargs['pipeline_name'] = origin_param_dict['pipeline_name']
    return kwargs