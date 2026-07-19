"""Root entry point for Streamlit Community Cloud.

Streamlit Cloud looks for `streamlit_app.py` at the repository root by default, so this
shim runs the real app in `app/streamlit_app.py` rather than duplicating it.

    streamlit run streamlit_app.py        # same as running app/streamlit_app.py

`__file__` is set to the real module path in the exec namespace. Without it the app
inherits this shim's `__file__`, and its `Path(__file__).parent.parent` walks one level
above the repository — which is how the hosted build ended up looking for the cohort
data in `/mount/src/data/cohort` instead of `/mount/src/<repo>/data/cohort`.
"""

from pathlib import Path

APP = Path(__file__).resolve().parent / "app" / "streamlit_app.py"

exec(
    compile(APP.read_text(), str(APP), "exec"),
    {"__file__": str(APP), "__name__": "__main__"},
)
