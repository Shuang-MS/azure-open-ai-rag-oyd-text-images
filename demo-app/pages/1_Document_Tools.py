import os
import subprocess
from pathlib import Path

import streamlit as st


# Resolve important paths relative to this file so the page works no matter where it is run from.
APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent
HELPER_SCRIPT = PROJECT_ROOT / "helper.sh"
SAMPLE_DOCS_DIR = PROJECT_ROOT / "sample-documents"

def run_helper(subcommand: str, extra_env: dict | None = None) -> subprocess.CompletedProcess[str]:
    """Invoke helper.sh with the requested subcommand and return the completed process."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        ["bash", str(HELPER_SCRIPT), subcommand],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


st.title("Document Upload & Indexer")

st.caption("Use the same helper.sh commands from the UI to push files to Blob Storage and run the Azure AI Search indexer.")

SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)

with st.form("upload_form", clear_on_submit=True):
    uploaded_file = st.file_uploader("Document to upload", type=["pdf"], accept_multiple_files=False)
    suggested_name = uploaded_file.name if uploaded_file else ""
    blob_name = st.text_input("Blob filename", value=suggested_name, placeholder="example.pdf")
    submit_upload = st.form_submit_button("Upload via helper.sh")

    if submit_upload:
        if uploaded_file is None:
            st.warning("Please choose a PDF document to upload.")
        else:
            safe_name = os.path.basename(blob_name.strip())
            if not safe_name:
                st.warning("Provide a valid filename for the document in Blob Storage.")
            else:
                destination = SAMPLE_DOCS_DIR / safe_name
                try:
                    destination.write_bytes(uploaded_file.getbuffer())
                except OSError as exc:
                    st.error(f"Failed to save file locally: {exc}")
                else:
                    with st.spinner("Uploading document to Blob Storage..."):
                        result = run_helper("upload-pdf", extra_env={"file_name": safe_name})

                    output_lines = []
                    if result.stdout:
                        output_lines.append(result.stdout.strip())
                    if result.stderr:
                        output_lines.append(result.stderr.strip())
                    combined_output = "\n".join(line for line in output_lines if line)

                    if result.returncode == 0:
                        st.success(f"Upload completed for {safe_name}.")
                    else:
                        st.error(f"Upload failed with exit code {result.returncode}.")

                    if combined_output:
                        st.code(combined_output, language="bash")

st.divider()

st.subheader("Run Azure AI Search Indexer")
if st.button("Trigger indexer run"):
    with st.spinner("Running indexer..."):
        result = run_helper("run-indexer")

    output_lines = []
    if result.stdout:
        output_lines.append(result.stdout.strip())
    if result.stderr:
        output_lines.append(result.stderr.strip())
    combined_output = "\n".join(line for line in output_lines if line)

    if result.returncode == 0:
        st.success("Indexer run triggered successfully.")
    else:
        st.error(f"Indexer run failed with exit code {result.returncode}.")

    if combined_output:
        st.code(combined_output, language="bash")

if SAMPLE_DOCS_DIR.exists():
    current_docs = sorted(p.name for p in SAMPLE_DOCS_DIR.glob("**/*") if p.is_file())
    if current_docs:
        st.caption("Current files in sample-documents:")
        st.write(", ".join(current_docs))
