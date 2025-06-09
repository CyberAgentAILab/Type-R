MNT="data"
DATA_ROOT="/${MNT}/datasets/anytext"


MASKTEXTSPOTTEV3_WEIGHT="/${MNT}/work/MaskTextSpotterV3/output/mixtrain/trained_model.pth"
CRAFT_WEIGHT="/${MNT}/datasets/ocr/weight/"
HISAM_WEIGHT="/${MNT}/datasets/ocr/hisam_weight"
DEEPSOLO_WEIGHT="/${MNT}/datasets/ocr/solo/vitaev2-s_pretrain_synth-tt-mlt-13-15-textocr.pth"

MODELSCOPE_WEIGHT="/${MNT}/datasets/anytext/eval_model/cv_convnextTiny_ocr-recognition-general_damo"
TROCR_WEIGHT="/${MNT}/datasets/ocr/trocr-large-str"
CLOVARECOG_WEIGHT="/${MNT}/datasets/ocr/weight/"

pytest tests \
    --gpufunc \
    --masktextspotterv3_weightdir=${MASKTEXTSPOTTEV3_WEIGHT} \
    --craft_weightdir=${CRAFT_WEIGHT} \
    --hisam_weightdir=${HISAM_WEIGHT} \
    --deepsolo_weightdir=${DEEPSOLO_WEIGHT} \
    --modelscope_weightdir=${MODELSCOPE_WEIGHT} \
    --trocr_weightdir=${TROCR_WEIGHT} \
    --clovarecog_weightdir=${CLOVARECOG_WEIGHT} \
    --font_path=${DATA_ROOT}/arial_unicode_ms.ttf \
    --anytext_path=${DATA_ROOT}/anytext_v1.1.ckpt \
