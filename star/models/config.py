import json
from attrdict import AttrDict
from transformers.configuration_utils import PretrainedConfig

def load_config_from_json(json_path):
    with open(json_path, "r") as f:
        config_data = json.load(f)
    return config_data

class STARMultiModalConfig(PretrainedConfig):
    model_type = "STARMultiModal"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.pixel_encoder = AttrDict(kwargs.get("pixel_encoder", {}))
        self.pixel_adapter = AttrDict(kwargs.get("pixel_adapter", {}))
        self.pixel_output_head = AttrDict(kwargs.get("pixel_output_head", {}))
        self.language_model = AttrDict(kwargs.get("language_model", {}))
        self.stacked_ar = AttrDict(kwargs.get("stacked_ar", {}))
        self.pixel_decoder = AttrDict(kwargs.get("pixel_decoder", {}))
        