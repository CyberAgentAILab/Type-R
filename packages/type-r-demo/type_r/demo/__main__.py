from streamlit.logger import get_logger

from type_r.demo import frontend

logger = get_logger(__name__)


def main():
    """Main entry point for the Type-R Model Server."""
    frontend.build_ui()


if __name__ == "__main__":
    main()
