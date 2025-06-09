uv run pytest tests \
    --gpufunc \
    --masktextspotterv3_weightdir=${MASKTEXTSPOTTEV3_WEIGHT} \
    --craft_weightdir=${CRAFT_WEIGHT} \
    --hisam_weightdir=${HISAM_WEIGHT} \
    --deepsolo_weightdir=${DEEPSOLO_WEIGHT} \
    --modelscope_weightdir=${MODELSCOPE_WEIGHT} \
    --trocr_weightdir=${TROCR_WEIGHT} \
    --clovarecog_weightdir=${CLOVARECOG_WEIGHT} \
    --font_path=${FONT_PATH} \
    --anytext_path=${ANYTEXT_WEIGHT} \

