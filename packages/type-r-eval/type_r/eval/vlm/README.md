# Setup

1. initialize .envrc if it does not exist

```
cp .envrc.example .envrc
```

2. Set appropriate environment variables in `.envrc`

# Examples

## Voting

### Text quality

```
poetry run python3 -m evalbyvlm \
    evaluation=voting_text_quality \
    dataset.image_dir=[<INPUT_DIR_1>,<INPUT_DIR_2>] \
    output_path=<OUTPUT_PATH>
```

### Text-image matching
Prompts can be fed by two ways:

- from text file: specify `+dataset.prompt_dir=<PROMPT_DIR>`, where the directory contain text files where each file contain a prompt.
- from hfds file: `+dataset.prompt_hfds=<PROMPT_HFDS> +dataset.prompt_hfds_split=<PROMPT_HFDS_SPLIT> +dataset.prompt_hfds_dataset_name=<PROMPT_HFDS_DATASET_NAME>`

```
poetry run python3 -m evalbyvlm \
    evaluation=voting_text_image_matching \
    dataset.image_dir=[<INPUT_DIR_1>,<INPUT_DIR_2>] \
    +dataset.prompt_dir=<PROMPT_DIR> \
    output_path=<OUTPUT_PATH>
```

### Collecting results

```
poetry run python3 -m evalbyvlm.metric --type voting --csv_path <CSV_PATH>
```

## Rating

### Text quality
```
poetry run python3 -m evalbyvlm \
    evaluation=rating_text_quality \
    dataset.image_dir=<INPUT_DIR> \
    output_path=<OUTPUT_PATH>
```

### Text-image matching
```
poetry run python3 -m evalbyvlm \
    evaluation=rating_text_image_matching \ \
    dataset.image_dir=<INPUT_DIR> \
    +dataset.prompt_dir=<PROMPT_DIR> \
    output_path=<OUTPUT_PATH>
```
### Collecting results

```
poetry run python3 -m evalbyvlm.metric --type rating --csv_path <CSV_PATH>
```

# Additional options
- `vlm_name`: change base VLMs for evaluation. If you work on pairwise comparison, please specify the models that can support multiple input images. For full list of models, please refer to [here](https://github.com/open-compass/VLMEvalKit/tree/main?tab=readme-ov-file#-datasets-models-and-evaluation-results).

# Q&A
- why not using the official [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)?
because of very tight dependency requirements in [requirements.txt](https://github.com/open-compass/VLMEvalKit/blob/main/requirements.txt)