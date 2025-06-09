from .encoders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "type_r.editor.udifftext.modules.GeneralConditioner",
    "params": {"emb_models": []},
}
