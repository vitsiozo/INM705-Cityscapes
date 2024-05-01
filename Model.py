from torch import nn

from models.BaselineModel import BaselineModel
from models.UNetModel import UNetModel
from models.BaselineNoBatchNormModel import BaselineNoBatchNormModel
from models.UNetNoBatchNormModel import UNetNoBatchNormModel
from models.UNetTransformerModel import UNetTransformerModel
from models.UNetTransformerPretrainedModel import UNetTransformerPretrainedModel
from models.Swin2Model import Swin2Model
from models.Swin2BaseModel import Swin2BaseModel
from models.Swin2LargeModel import Swin2LargeModel
from models.NewBaselineModel import NewBaselineModel
from models.ResnetBaseline import *
from models.SimpleFCN import *

class Model:
    models = {
        'Baseline': SimpleFCN,
        'EnhancedUNet': UNetNoBatchNormModel,
        'EnhancedSwin': Swin2BaseModel,

        'BaselineNoBatchNorm': BaselineNoBatchNormModel,
        'UNet': UNetModel,
        'UNetNoBatchNorm': UNetNoBatchNormModel,
        'UNetTransformerPretrained': UNetTransformerPretrainedModel,
        'UNetTransformer': UNetTransformerModel,
        'Swin2': Swin2Model,
        'Swin2Base': Swin2BaseModel,
        'Swin2Large': Swin2LargeModel,
        'NewBaseline': NewBaselineModel,
        'ResnetBaseline': ResnetBaseline,
        'BetterResnetBaseline': ResnetBaseline,
        'SimpleFCN': SimpleFCN,
    }

    @classmethod
    def keys(cls) -> list[str]:
        return cls.models.keys()

    @classmethod
    def instanciate(cls, model_name: str, **kwargs) -> nn.Module:
        model = cls.models[model_name]
        model_args = {k: v for k, v in kwargs.items() if v is not None}
        return model(**model_args)
