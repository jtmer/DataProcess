from transforms import *

class PipelineExecutor:
    def __init__(self, stages: list, config: dict, model=None, dataset=None):
        self.stages = stages
        self.config = config
        self.model = model
        self.dataset = dataset
        self.instances = {}

    def _instantiate(self, name, x):
        p = self.config
        if name == "Trimmer":
            return Trimmer(p["trimmer_seq_len"], p["pred_len"])
        elif name == "Sampler":
            return Sampler(p["sampler_factor"])
        elif name == "Inputer":
            return Inputer(p["inputer_detect_method"], p["inputer_fill_method"], p["original_data"])
        elif name == "Denoiser":
            return Denoiser(p["denoiser_method"])
        elif name == "Warper":
            return Warper(p["warper_method"], p["clip_factor"])
        elif name == "Decomposer":
            return Decomposer(p["decomposer_period"], p["decomposer_components"])
        elif name == "Differentiator":
            return Differentiator(p["differentiator_n"], p["clip_factor"])
        elif name == "Normalizer":
            scaler = self.dataset.get_scaler(p["normalizer_method"], p["target_column"])
            return Normalizer(p["normalizer_method"], p["normalizer_mode"], x, p["original_data"], scaler,
                              p["normalizer_ratio"], p["clip_factor"])
        elif name == "Aligner":
            return Aligner(p["aligner_mode"], p["aligner_method"],
                           data_patch_len=p["patch_len"], model_patch_len=self.model.patch_len)
        else:
            raise ValueError(f"Unknown stage: {name}")

    def preprocess(self, x):
        for name in self.stages:
            instance = self._instantiate(name, x)
            self.instances[name] = instance
            x = instance.pre_process(x)
        return x

    def postprocess(self, x):
        for name in reversed(self.stages):
            x = self.instances[name].post_process(x)
        return x
