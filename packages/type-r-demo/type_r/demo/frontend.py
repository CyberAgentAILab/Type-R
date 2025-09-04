import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit.logger import get_logger
from type_r.demo.download import download_resources
from type_r.util.loader import load_class_lazily

from type_r_app.launcher.layout_correction import get_word_mapping
from type_r_app.launcher.typo_correction import brushup_all, get_text_editing_inputs

logger = get_logger(__name__)


def initialize_session_state() -> None:
    """Initialize session state variables for the Type-R Model Server."""
    logger.info("Initializing session state variables.")
    for key in [
        "adjusted_image",
        "edited_image",
        "initial_image",
        "prompt",
        "word_mapping",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None

    if "t2i_model" not in st.session_state:
        with st.spinner("Initializing T2I models..."):
            st.session_state.t2i_model = load_class_lazily(
                "type_r.t2i.flux.Flux"
            )(cache_dir="resources/weight/t2i/flux")
    if "ocr_recognition" not in st.session_state:
        with st.spinner("Initializing OCR recognition model."):
            st.session_state.ocr_recognition = load_class_lazily(
                "type_r.ocr.modelscopeocr.ConvnextTinyOCR"
            )("resources/weight/ocr/modelscope")
    if "ocr_detection_kls" not in st.session_state:
        with st.spinner("Initializing OCR detection model."):
            st.session_state.ocr_detection_kls = load_class_lazily(
                "type_r.ocr.paddleocr.PaddleOCRDet"
            )
    if "text_eraser" not in st.session_state:
        with st.spinner("Initializing text eraser model."):
            st.session_state.text_eraser = load_class_lazily(
                "type_r.eraser.lama.LamaWrapper"
            )(
            weight_path="resources/weight/text_eraser/big-lama.pt",
            polygon_dilation=True,
            dilate_kernel_size=25,
            dilate_iteration=1,
        )
    if "adjuster" not in st.session_state:
        with st.spinner("Initializing adjuster model."):
            st.session_state.adjuster = load_class_lazily(
                "type_r.adjuster.default.DefaultAdjuster"
            )()
    if "text_editor" not in st.session_state:
        with st.spinner("Initializing text editor model."):
            st.session_state.text_editor = load_class_lazily(
                "type_r.editor.anytext.AnyTextWrapper"
            )(
                weight_path="resources/weight/text_editor/anytext.ckpt",
                font_path="resources/data/LiberationSans-Regular.ttf",
            )


def generate_initial_image_from_prompt() -> None:
    """Generate an initial prompt for the Type-R Model Server."""
    if st.session_state.prompt is None:
        logger.info("Prompt must be set before generating an image.")
        return

    try:
        logger.info(f"Generating image for prompt: {st.session_state.prompt}")
        with st.spinner("Generating an image..."):
            image = st.session_state.t2i_model.sample(st.session_state.prompt, 512, 512)
            st.session_state.initial_image = image
        st.image(image, caption="Generated Image", use_container_width=True)

    except Exception as e:
        logger.exception(f"Error generating image: {e}")
        st.error("Failed to generate image. Please try again.")


def identify_word_mapping() -> None:
    """Identify word mapping in the generated image."""
    if st.session_state.initial_image is None:
        logger.info("Image must be generated before identifying word mapping.")
        return

    try:
        with st.spinner("Identifying word mapping..."):
            image = np.array(st.session_state.initial_image).astype(dtype=np.uint8)
            # PaddleOCR cannot be used multiple times in a single session.
            # https://github.com/PaddlePaddle/PaddleOCR/discussions/12623
            ocr_detection = st.session_state.ocr_detection_kls()
            initial_word_mapping = get_word_mapping(
                st.session_state.prompt,
                image,
                ocr_detection,
                st.session_state.ocr_recognition,
                filter=True,
            )
        with st.chat_message("assistant"):
            # TODO: Display the summary of editing plans from word_mapping
            st.markdown("Word mapping identified successfully.")
            st.json(initial_word_mapping.model_dump(), expanded=False)

        with st.spinner("Adjusting text layout..."):
            aligned_image, word_mapping = st.session_state.adjuster(
                image=image,
                word_mapping=initial_word_mapping,
                text_eraser=st.session_state.text_eraser,
            )
            st.session_state.adjusted_image = aligned_image
            st.session_state.word_mapping = word_mapping

        with st.chat_message("assistant"):
            st.markdown("Word mapping adjusted successfully.")
            st.json(word_mapping.model_dump(), expanded=False)
            st.image(aligned_image, caption="Aligned Image", use_container_width=True)

    except Exception as e:
        logger.exception(f"Error identifying word mapping: {e}")
        st.error("Failed to identify word mapping. Please try again.")


def edit_text_in_image() -> None:
    """Edit text in the image using the text editor."""
    if st.session_state.adjusted_image is None or st.session_state.word_mapping is None:
        logger.info("Image and word mapping must be available for text editing.")
        return

    # TODO: Check if there is any text to edit. If not, skip this step.

    try:
        with st.spinner("Editing text in image..."):
            text_editing_inputs = get_text_editing_inputs(st.session_state.word_mapping)
            logger.info(f"Text editing inputs: {text_editing_inputs}")
            edited_image, _ = brushup_all(
                st.session_state.prompt,
                cv2.resize(st.session_state.adjusted_image, (512, 512)),
                st.session_state.text_editor,
                st.session_state.ocr_recognition,
                inpaint_text_inputs=text_editing_inputs,
            )
            image = Image.fromarray(edited_image.astype(np.uint8))
            st.session_state.edited_image = image

        st.image(image, caption="Edited Image", use_container_width=True)

    except Exception as e:
        logger.exception(f"Error editing text: {e}")
        st.error("Failed to edit text in image. Please try again.")


def build_ui():
    """Build the Streamlit UI for the Type-R Model Server."""
    st.title("Type-R Model Server")

    if not torch.cuda.is_available():
        st.info("CUDA is not available. Please use a machine with a GPU.")
        return

    download_resources("resources/")
    initialize_session_state()

    if prompt := st.chat_input("Enter your prompt to generate an image"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.prompt = prompt

    generate_initial_image_from_prompt()
    identify_word_mapping()
    edit_text_in_image()
