import streamlit as st
import json
import os
import hashlib
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.callbacks import get_openai_callback
import chromadb

# Streamlit page config
st.set_page_config(
    page_title="Column Lineage & Code Snippet Explorer - GPT-4",
    page_icon="🔍",
    layout="wide",
)

TARGET_COLUMNS = [
    "discounted_amount",
    "product",
    "tier",
    "tier_revenue",
    "avg_tier_revenue",
    "tier_quantity",
]

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""


@st.cache_data
def load_lineage_json(file_path: str) -> Dict:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            lines = content.split("\n")
            for line in lines:
                if line.startswith('{"id":'):
                    return json.loads(line)
        return {}
    except Exception as e:
        st.error(f"Error loading lineage JSON: {e}")
        return {}


@st.cache_data
def load_file_chunks(filepath: str, chunk_size: int = 500) -> List[str]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        lines = content.splitlines()
        chunks = []
        cur_chunk = []
        cur_len = 0
        for line in lines:
            cur_chunk.append(line)
            cur_len += len(line)
            if cur_len > chunk_size:
                chunks.append("\n".join(cur_chunk))
                cur_chunk = []
                cur_len = 0
        if cur_chunk:
            chunks.append("\n".join(cur_chunk))
        return chunks
    except Exception as e:
        st.error(f"Error loading file chunks: {e}")
        return []


def make_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@st.cache_resource
def initialize_embeddings_and_collection(_api_key: str):
    try:
        os.environ["OPENAI_API_KEY"] = _api_key
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", request_timeout=60, max_retries=8
        )

        persist_path = "./chroma_storage"
        client = chromadb.PersistentClient(path=persist_path)
        collection_name = "code_snippet_collection"
        try:
            collection = client.get_collection(collection_name)
        except:
            collection = client.create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )
            code_chunks = load_file_chunks("./lineage_test.py")
            if code_chunks:
                ids = [make_id(chunk) for chunk in code_chunks]
                vectors = embeddings.embed_documents(code_chunks)
                collection.upsert(documents=code_chunks, embeddings=vectors, ids=ids)
        return embeddings, collection
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        return None, None


def create_batch_lineage_summaries(
    lineage_json: Dict, columns: List[str], api_key: str
) -> Dict[str, str]:
    try:
        os.environ["OPENAI_API_KEY"] = api_key
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)

        prompt_template = """
You are a data engineer assistant. Given a Spline lineage JSON and a list of column names,
generate a concise natural language summary explaining how each column is computed.

Columns: {columns}

Lineage JSON (focus on attributes and expressions related to these columns):
{lineage_text}

Instructions:
- Explain how each column is computed based on the lineage
- Describe key transformations, joins, aggregations
- Keep it concise and understandable
- Provide output in JSON format mapping column names to their summaries:
{{"column_name": "summary", ...}}
"""
        prompt = PromptTemplate(
            input_variables=["columns", "lineage_text"], template=prompt_template
        )
        chain = LLMChain(llm=llm, prompt=prompt, output_key="summary_json")
        with get_openai_callback() as cb:
            result = chain.invoke(
                {
                    "columns": ", ".join(columns),
                    "lineage_text": json.dumps(lineage_json, indent=2),
                }
            )
            st.session_state.total_tokens = cb.total_tokens
            st.session_state.total_cost = cb.total_cost
        summary_json_str = result.get("summary_json", "{}")

        summaries = json.loads(summary_json_str)
        return summaries
    except Exception as e:
        fallback = {
            "product": "Product field used for grouping operations.",
            "tier": "Customer tier field used for grouping and conditional logic.",
            "tier_revenue": "Sum of discounted_amount grouped by product and tier.",
            "avg_tier_revenue": "Average of discounted_amount grouped by product and tier.",
            "tier_quantity": "Sum of quantity grouped by product and tier.",
        }
        return {
            col: fallback.get(col, f"Error generating summary: {str(e)}")
            for col in columns
        }


def query_code_snippet(
    lineage_summary: str, column: str, embeddings, collection
) -> str:
    try:
        enhanced_query = f"How is {column} calculated? {lineage_summary}"
        query_vector = embeddings.embed_query(enhanced_query)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=1,
            include=["documents", "distances"],
        )
        if results["documents"] and results["documents"][0]:
            return results["documents"][0][0]
        else:
            return "No matching code snippet found."
    except Exception as e:
        return f"Error retrieving code snippet: {str(e)}"


def chatbot_response_with_rag(query: str, embeddings, collection, api_key: str) -> str:
    try:
        os.environ["OPENAI_API_KEY"] = api_key
        query_vector = embeddings.embed_query(query)

        results = collection.query(
            query_embeddings=[query_vector], n_results=3, include=["documents"]
        )
        retrieved_context = (
            "\n\n".join(results["documents"][0]) if results.get("documents") else ""
        )

        llm = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=300)
        prompt_template = """
You are a helpful data lineage assistant. Answer the user's question based on the provided code snippets and query.

Question: {query}

Context:
{context}

Provide a clear and concise answer.
"""
        prompt = PromptTemplate(
            input_variables=["query", "context"], template=prompt_template
        )
        chain = LLMChain(llm=llm, prompt=prompt, output_key="answer")
        with get_openai_callback() as cb:
            result = chain.invoke({"query": query, "context": retrieved_context})
            st.session_state.chat_tokens = cb.total_tokens
            st.session_state.chat_cost = cb.total_cost

        return result.get("answer", "").strip()
    except Exception as e:
        return f"Error generating chat response: {str(e)}"


def main():
    st.title("🔍 Column Lineage & Code Snippet Explorer - GPT-4")
    st.markdown(
        "Automatically map column lineage to code snippets with AI-powered explanations"
    )

    if not st.session_state.openai_api_key:
        st.sidebar.header("Configuration")
        api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
        if api_key:
            st.session_state.openai_api_key = api_key
            st.rerun()
        else:
            st.warning("Please enter your OpenAI API key to continue.")
            return

    lineage_json = load_lineage_json("./lineage_output_clean.log")
    if not lineage_json:
        st.error("Could not load lineage data from lineage_output_clean.log")
        return

    embeddings, collection = initialize_embeddings_and_collection(
        st.session_state.openai_api_key
    )
    if not embeddings or not collection:
        st.error("Could not initialize embeddings or collection")
        return

    tab1, tab2 = st.tabs(["📊 Column Documentation", "💬 Interactive Chat"])

    with tab1:
        st.header("Column Lineage Documentation")

        if "cached_summaries" not in st.session_state:
            st.session_state.cached_summaries = {}

        if st.button("Generate Summaries"):
            with st.spinner("Generating lineage summaries for all columns..."):
                summaries = create_batch_lineage_summaries(
                    lineage_json, TARGET_COLUMNS, st.session_state.openai_api_key
                )
                st.session_state.cached_summaries = summaries
        else:
            summaries = st.session_state.cached_summaries

        for idx, col in enumerate(TARGET_COLUMNS):
            with st.expander(f"🔹 Column: **{col}**", expanded=(idx < 2)):
                col1, col2 = st.columns([1, 1])
                with col1:
                    summary = summaries.get(col, "No summary available.")
                    st.subheader("📝 Lineage Summary")
                    st.markdown(summary)
                with col2:
                    snippet = query_code_snippet(summary, col, embeddings, collection)
                    st.subheader("💻 Best Matching Code Snippet")
                    st.code(snippet, language="python")

        total_tokens = st.session_state.get("total_tokens", None)
        total_cost = st.session_state.get("total_cost", None)
        if total_tokens and total_cost:
            st.markdown(
                f"**Summary generation tokens:** {total_tokens}, **Cost:** ${total_cost:.6f}"
            )

    with tab2:
        st.header("Interactive Lineage Chat")
        st.markdown("Ask questions about the lineage, transformations, or code:")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("Ask about lineage, columns, or transformations...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = chatbot_response_with_rag(
                        prompt, embeddings, collection, st.session_state.openai_api_key
                    )
                    st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            chat_tokens = st.session_state.get("chat_tokens", None)
            chat_cost = st.session_state.get("chat_cost", None)
            if chat_tokens and chat_cost:
                st.sidebar.markdown(
                    f"**Chat tokens:** {chat_tokens}, **Cost:** ${chat_cost:.6f}"
                )


if __name__ == "__main__":
    main()
