# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import List, Optional, Union

from modelscope.models.base import Model
from modelscope.utils.config import ConfigDict, check_config
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, ThirdParty
from modelscope.utils.hub import read_config
from modelscope.utils.plugins import register_modelhub_repo, register_plugins_repo

from .nlp.llm_pipeline import LLMAdapterRegistry, ModelTypeHelper
from .util import is_official_hub_path
from .util import is_official_hub_path


def llm_first_checker(
    model: Union[str, List[str], Model, List[Model]], revision: Optional[str]
) -> Optional[str]:
    if isinstance(model, list):
        model = model[0]
    if not isinstance(model, str):
        model = model.model_dir
    model_type = ModelTypeHelper.get(
        model, revision, with_adapter=True, split="-", use_cache=True
    )
    if LLMAdapterRegistry.contains(model_type):
        return "llm"


def pipeline(
    task: str = None,
    model: Union[str, List[str], Model, List[Model]] = None,
    preprocessor=None,
    config_file: str = None,
    pipeline_name: str = None,
    framework: str = None,
    device: str = "gpu",
    model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
    ignore_file_pattern: List[str] = None,
    **kwargs,
) -> Pipeline:
    """Factory method to build an obj:`Pipeline`.


    Args:
        task (str): Task name defining which pipeline will be returned.
        model (str or List[str] or obj:`Model` or obj:list[`Model`]): (list of) model name or model object.
        preprocessor: preprocessor object.
        config_file (str, optional): path to config file.
        pipeline_name (str, optional): pipeline class name or alias name.
        framework (str, optional): framework type.
        model_revision: revision of model(s) if getting from model hub, for multiple models, expecting
        all models to have the same revision
        device (str, optional): whether to use gpu or cpu is used to do inference.
        ignore_file_pattern(`str` or `List`, *optional*, default to `None`):
            Any file pattern to be ignored in downloading, like exact file names or file extensions.

    Return:
        pipeline (obj:`Pipeline`): pipeline object for certain task.

    Examples:
        >>> # Using default model for a task
        >>> p = pipeline('image-classification')
        >>> # Using pipeline with a model name
        >>> p = pipeline('text-classification', model='damo/distilbert-base-uncased')
        >>> # Using pipeline with a model object
        >>> resnet = Model.from_pretrained('Resnet')
        >>> p = pipeline('image-classification', model=resnet)
        >>> # Using pipeline with a list of model names
        >>> p = pipeline('audio-kws', model=['damo/audio-tts', 'damo/auto-tts2'])
    """
    if task is None and pipeline_name is None:
        raise ValueError("task or pipeline_name is required")

    third_party = kwargs.get(ThirdParty.KEY)
    if third_party is not None:
        kwargs.pop(ThirdParty.KEY)
    pipeline_props = {"type": pipeline_name}
    if pipeline_name is None:
        # get default pipeline for this task
        if isinstance(model, str) or (
            isinstance(model, list) and isinstance(model[0], str)
        ):
            if is_official_hub_path(model, revision=model_revision):
                # read config file from hub and parse
                cfg = (
                    read_config(model, revision=model_revision)
                    if isinstance(model, str)
                    else read_config(model[0], revision=model_revision)
                )
                register_plugins_repo(cfg.safe_get("plugins"))
                register_modelhub_repo(model, cfg.get("allow_remote", False))
                pipeline_name = (
                    llm_first_checker(model, model_revision)
                    if kwargs.get("llm_first")
                    else None
                )
                if pipeline_name is not None:
                    pipeline_props = {"type": pipeline_name}
                else:
                    check_config(cfg)
                    pipeline_props = cfg.pipeline
        elif model is not None:
            # get pipeline info from Model object
            first_model = model[0] if isinstance(model, list) else model
            if not hasattr(first_model, "pipeline"):
                # model is instantiated by user, we should parse config again
                cfg = read_config(first_model.model_dir)
                check_config(cfg)
                first_model.pipeline = cfg.pipeline
            pipeline_props = first_model.pipeline
        else:
            pipeline_name, default_model_repo = get_default_pipeline_info(task)
            model = normalize_model_input(default_model_repo, model_revision)
            pipeline_props = {"type": pipeline_name}

    pipeline_props["model"] = model
    pipeline_props["device"] = device
    cfg = ConfigDict(pipeline_props)

    # support set llm_framework=None
    if pipeline_name == "llm" and kwargs.get("llm_framework", "") == "":
        kwargs["llm_framework"] = "vllm"
    clear_llm_info(kwargs)
    if kwargs:
        cfg.update(kwargs)

    if preprocessor is not None:
        cfg.preprocessor = preprocessor

    return build_pipeline(cfg, task_name=task)
