import html
import json
import os
import re
import textwrap
from typing import Any, Dict, List

import dotenv
import streamlit as st
from openai import AzureOpenAI


st.set_page_config(page_title="Inline Image Chat", layout="wide")

dotenv.load_dotenv()

endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
api_key = os.environ.get("AZURE_OPENAI_KEY")
deployment = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
deployment_embedding = os.environ.get("AZURE_OPENAI_CHATGPT_EMBEDDING_DEPLOYMENT")

search_endpoint = os.environ.get("SEARCH_ENDPOINT")
search_query_type = os.environ.get("SEARCH_QUERY_TYPE")

DEFAULT_IMAGE_SEMANTIC_CONFIGURATION = "semanticconfig"
DEFAULT_IMAGE_QUERY_TYPE = "semantic"

raw_image_semantic = os.environ.get("IMAGE_SEARCH_SEMANTIC_CONFIGURATION")
if raw_image_semantic is not None:
    candidate_semantic = raw_image_semantic.strip()
    if candidate_semantic.lower() in {"", "none", "null"}:
        image_semantic_config = None
    else:
        image_semantic_config = candidate_semantic
else:
    image_semantic_config = DEFAULT_IMAGE_SEMANTIC_CONFIGURATION

raw_image_query_type = os.environ.get("IMAGE_SEARCH_QUERY_TYPE")
if raw_image_query_type:
    image_query_type = raw_image_query_type.strip() or DEFAULT_IMAGE_QUERY_TYPE
else:
    image_query_type = search_query_type or DEFAULT_IMAGE_QUERY_TYPE

if image_query_type and "semantic" in image_query_type.lower() and not image_semantic_config:
    image_query_type = "simple"
search_api_key = os.environ.get("SEARCH_API_KEY")

storage_account_name = os.environ.get("STORAGE_ACCOUNT_NAME")
blob_sas_token = os.environ.get("BLOB_SAS_TOKEN_CUSTOM")
index_name_custom = os.environ.get("INDEX_NAME_CUSTOM")

DEFAULT_SYSTEM_PROMPT = (
    "You are an AI assistant that helps people explore documents that include both text and images. When you respond, keep answers concise and cite the supporting pages."
)

CHAT_STATE_KEY = "inline_image_chat_state"


def open_avatar_image(file_path: str) -> bytes | None:
    try:
        with open(file_path, "rb") as avatar_file:
            return avatar_file.read()
    except OSError:
        return None


def ensure_global_state() -> None:
    if "avatar_user" not in st.session_state:
        st.session_state["avatar_user"] = open_avatar_image("avatars/avatar_user.png")
    if "avatar_ai" not in st.session_state:
        st.session_state["avatar_ai"] = open_avatar_image("avatars/avatar_ai.png")
    if "aoai_client" not in st.session_state:
        st.session_state["aoai_client"] = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )


def init_image_chat_state(system_prompt: str | None = None) -> None:
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    st.session_state[CHAT_STATE_KEY] = {
        "system_prompt": prompt,
        "messages": [{"role": "system", "content": prompt}],
        "history": [],
        "responses": [],
        "citations": []
    }


def get_state() -> Dict[str, Any]:
    if CHAT_STATE_KEY not in st.session_state:
        init_image_chat_state()
    state = st.session_state[CHAT_STATE_KEY]
    return state


def remove_resource_markers(text: str | None) -> str:
    if not text:
        return ""
    patterns = [
        re.compile(r"\s*[Rr]esource\s+[A-Za-z0-9+/=]+(?:;\d+)?", re.MULTILINE),
        re.compile(r"\s*[A-Za-z0-9+/=]{30,}(?:;\d+)?", re.MULTILINE),
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = pattern.sub(" ", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def generate_sas_token(path: str | None) -> str | None:
    storage_path = f"https://{storage_account_name}.blob.core.windows.net/{path.lstrip('/')}" if path else None
    if not path or not blob_sas_token:
        return storage_path
    if "?" in path:
        return storage_path
    return f"{storage_path}?{blob_sas_token}"


def maybe_parse_json(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return value
    if stripped[0] in "{[" and stripped[-1] in "}]" :
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


def coerce_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def extract_numeric_field(citation: Dict[str, Any], field_names: set[str]) -> int | None:
    stack: List[Any] = [citation]
    visited: set[int] = set()

    while stack:
        current = stack.pop()

        if isinstance(current, str):
            parsed = maybe_parse_json(current)
            if parsed is current:
                continue
            stack.append(parsed)
            continue

        if isinstance(current, (dict, list)):
            obj_id = id(current)
            if obj_id in visited:
                continue
            visited.add(obj_id)

        if isinstance(current, dict):
            for key, value in current.items():
                if key in field_names:
                    numeric = coerce_int(value)
                    if numeric is not None:
                        return numeric
                stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)

    return None


def extract_page_number(citation: Dict[str, Any]) -> int | None:
    candidates = {"pageNumber", "page_number", "page"}
    return extract_numeric_field(citation, candidates)

def extract_image_urls(citation: Dict[str, Any]) -> List[str]:
    keys_to_check = ["url"]
    urls: List[str] = []

    for key in keys_to_check:
        if key == "content_path" and not citation.get("image_document_id"):
            continue

        raw_value = citation.get(key)
        if not raw_value:
            continue

        values_to_process: List[Any]
        if isinstance(raw_value, list):
            values_to_process = raw_value
        elif isinstance(raw_value, str):
            parsed = maybe_parse_json(raw_value)
            if isinstance(parsed, list):
                values_to_process = parsed
            else:
                values_to_process = [raw_value]
        else:
            values_to_process = [raw_value]

        for candidate in values_to_process:
            if not isinstance(candidate, str):
                continue
            cleaned = candidate.strip()
            if not cleaned:
                continue
            tokenised = generate_sas_token(cleaned)
            if tokenised and tokenised not in urls:
                urls.append(tokenised)

    return urls


def build_citation(index: int, citation: Dict[str, Any]) -> Dict[str, Any]:
    citation_reference = f"[doc{index}]"
    snippet_source = (
        citation.get("content")
        or ""
    )
    snippet = remove_resource_markers(snippet_source)
    images = extract_image_urls(citation)
    page_number = extract_page_number(citation)
    raw_source_path = citation.get("filepath") or citation.get("url")
    source_url = generate_sas_token(raw_source_path) if raw_source_path else None

    title = (
        citation.get("document_title")
        or citation.get("title")
        or citation.get("filename")
        or ""
    )
    return {
        "citation_reference": citation_reference,
        "snippet": snippet,
        "images": images,
        "page_number": page_number,
        "source_path": raw_source_path,
        "source_url": source_url,
        "title": title,
        "has_images": bool(images),
    }

def llm_request(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    client = st.session_state["aoai_client"]
    search_parameters: Dict[str, Any] = {
        "endpoint": search_endpoint,
        "index_name": index_name_custom,
        "authentication": {"type": "api_key", "key": search_api_key},
        "query_type": image_query_type,
        "strictness": 3,
        "top_n_documents": 5,
        "embedding_dependency": {
            "type": "deployment_name",
            "deployment_name": deployment_embedding,
        },
        "fields_mapping": {
            "content_fields": ["content_text"],
            "content_fields_separator": "\n",
            "title_field": "document_title",
            "filepath_field": "content_path",
            "vector_fields": ["content_embedding"],
            "url_field": "content_path"
        },
    }

    if image_semantic_config:
        search_parameters["semantic_configuration"] = image_semantic_config
    elif image_query_type and "semantic" in image_query_type.lower():
        search_parameters["query_type"] = "simple"

    completion = client.chat.completions.create(
        model=deployment,
        messages=messages,
        max_tokens=800,
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        extra_body={
            "data_sources": [
                {
                    "type": "azure_search",
                    "parameters": search_parameters,
                }
            ]
        },
    )

    return json.loads(completion.to_json())


def process_completion(completion: Dict[str, Any]) -> tuple[str, str, List[Dict[str, Any]], Dict[str, Any]]:
    message = completion["choices"][0]["message"]
    raw_content = message.get("content", "")
    print("Raw content:", raw_content)
    content = remove_resource_markers(raw_content)

    citations: List[Dict[str, Any]] = []
    context = message.get("context", {})
    raw_citations: List[Dict[str, Any]] = context.get("citations", [])

    for idx, citation in enumerate(raw_citations, start=1):
        placeholder = f"[doc{idx}]"
        if placeholder not in raw_content:
            continue  # skip unused doc
        built = build_citation(idx, citation)
        citation_token = built["citation_reference"]
        if built["has_images"]:
            image_tags: List[str] = []
            for image_index, image_url in enumerate(built["images"], start=1):
                escaped_url = html.escape(image_url, quote=True)
                alt_text = html.escape(
                    f"Citation {idx} image {image_index}", quote=True
                )
                image_tags.append(
                    f"<div class='inline-citation-image'><img src=\"{escaped_url}\" alt=\"{alt_text}\" style=\"max-width: 100%; border-radius: 4px; margin: 0.5rem 0;\" /></div>"
                )
            inline_markup = "\n".join(image_tags)
            content = content.replace(citation_token, inline_markup, 1)
        content = content.replace(citation_token, "")
        citations.append(built)

    usage = completion.get("usage", {})
    return content, raw_content, citations, usage


def append_conversation(
    state: Dict[str, Any],
    prompt: str,
    display_answer: str,
    raw_answer: str,
    citations: List[Dict[str, Any]]
) -> None:
    state["history"].append(prompt)
    state["responses"].append(display_answer)
    state["citations"].append(citations)
    state["messages"].extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": raw_answer},
        ]
    )


def render_citations(
    citations: List[Dict[str, Any]]
) -> None:
    if not citations:
        return
    with st.expander("References"):
        for idx, citation in enumerate(citations):
            reference = citation["citation_reference"]
            title = citation.get("title") or citation.get("source_path") or reference
            source_url = citation.get("source_url")
            source_path = citation.get("source_path")
            page_number = citation.get("page_number")
            snippet = citation.get("snippet")

            header_parts: List[str] = [reference]
            if page_number is not None:
                header_parts.append(f"Page {page_number}")
            header_text = " • ".join(header_parts)
            st.markdown(f"**{header_text}**")

            if source_url:
                st.markdown(f"- Link: [{title}]({source_url})")
            elif source_path:
                st.markdown(f"- Source: {title} — `{source_path}`")
            else:
                st.markdown(f"- Source: {title}")

            if snippet:
                condensed = re.sub(r"\s+", " ", snippet.strip())
                preview = textwrap.shorten(condensed, width=360, placeholder="…")
                st.caption(f"Preview: {preview}")

            image_urls = citation.get("images") or []
            for image_index, image_url in enumerate(image_urls, start=1):
                st.image(image_url, caption=f"Image preview {image_index}", use_column_width=True)

            st.caption("Payload")
            st.json(citation, expanded=False)

            if idx < len(citations) - 1:
                st.divider()


ensure_global_state()
state = get_state()

with st.sidebar:
    st.subheader("Vision Chat Settings")
    st.caption(f"Using search index `{index_name_custom}`")

    new_prompt = st.text_area("System prompt", value=state["system_prompt"], height=120)
    if new_prompt.strip() and new_prompt != state["system_prompt"]:
        init_image_chat_state(new_prompt.strip())
        state = get_state()

    if st.button("Clear conversation"):
        init_image_chat_state(state["system_prompt"])
        state = get_state()

st.title("Document Vision Chat")
st.caption(
    "Ask anything about your indexed documents. Answers include inline images when available and detailed references in the expandable section."
)

response_container = st.container()

with st.form(key="image_chat_form", clear_on_submit=True):
    user_input = st.text_area("Type a question", key="image_chat_prompt", height=100)
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    prompt = user_input.strip()
    try:
        completion = llm_request(state["messages"] + [{"role": "user", "content": prompt}])
        display_answer, raw_answer, citations, _usage = process_completion(completion)
        append_conversation(state, prompt, display_answer, raw_answer, citations)
    except Exception as exc:
        st.error(f"Request failed: {exc}")

if state["responses"]:
    with response_container:
        for question, answer, citations in zip(
            state["history"],
            state["responses"],
            state["citations"]
        ):
            with st.chat_message("user", avatar=st.session_state.get("avatar_user")):
                st.write(question)
            with st.chat_message("assistant", avatar=st.session_state.get("avatar_ai")):
                st.markdown(
                    answer + "\n\n---\n\n<p style='text-align: right; font-size: 0.8em;'>AI-generated content may be incorrect.</p>",
                    unsafe_allow_html=True,
                )
                render_citations(citations)
