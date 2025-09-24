class BaseTransform:
    def __init__(self, **kwargs):
        """用于初始化参数"""
        pass

    def pre_process(self, data):
        """对输入数据进行变换"""
        raise NotImplementedError

    def post_process(self, data):
        """对模型输出进行还原"""
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: dict):
        """从配置字典构造实例"""
        return cls(**config)
