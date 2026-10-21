import json
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Text Visualizer", layout="wide")

@st.cache_data
def load_data():
    with open("bertopic_data.json", "r") as f:
        return json.load(f)

data = load_data()

st.title("Text Corpus Exploration Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "Frequency",
    "Topics",
    "Hierarchy",
    "Embeddings"
])

# Tab 1: TF-IDF Frequency
with tab1:
    st.subheader("Corpus-Wide Term Frequency (Mean TF-IDF)")
    st.caption("Reveals salient terms across the corpus. Misleads by ignoring word order and context.")
    freq_key = "frequencies" if "frequencies" in data else "global_frequencies"
    f_freq = pd.DataFrame(data[freq_key])    
    fig_freq = px.bar(f_freq, x="score", y="word", orientation="h")
    fig_freq.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_freq, use_container_width=True)

# Tab 2: Topic Model (Enhanced View)
with tab2:
    st.subheader("BERTopic Model (c-TF-IDF)")
    st.caption("Reveals thematic clusters. Misleads by forcing documents into discrete buckets.")
    
    topics = data["topics"]
    
    # 1. Okunabilir ve zenginlestirilmis secim menusu
    def format_topic_label(t_id):
        raw_name = topics[t_id]["name"]
        clean_words = ", ".join(raw_name.split("_")[1:4]).title()
        count = topics[t_id].get("count", len(topics[t_id].get("words", [])))
        return f"Topic {t_id}: {clean_words}  ({count} docs)"

    selected_id = st.selectbox(
        "Select Topic to Inspect",
        options=list(topics.keys()),
        format_func=format_topic_label
    )
    
    selected_topic = topics[selected_id]
    words_df = pd.DataFrame(selected_topic["words"])
    
    # 2. Iki kolonlu duzen: Sol panel istatistikler, sag panel gorsel grafik
    col_stats, col_chart = st.columns([1, 2])
    
    with col_stats:
        doc_count = selected_topic.get("count", "N/A")
        st.metric(label="Total Documents in Cluster", value=doc_count)
        st.markdown("**Top Salient Keywords (c-TF-IDF)**")
        st.dataframe(
            words_df.rename(columns={"word": "Token", "weight": "Score"}),
            use_container_width=True,
            hide_index=True
        )
        
    with col_chart:
        # Renk skalali interaktif bar grafik
        fig_topic = px.bar(
            words_df,
            x="weight",
            y="word",
            orientation="h",
            color="weight",
            color_continuous_scale="Tealgrn",
            labels={"weight": "c-TF-IDF Weight", "word": "Token"},
            title=f"Vocabulary Importance for Topic {selected_id}"
        )
        fig_topic.update_layout(
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_topic, use_container_width=True)

# Tab 3: Hierarchical Topics
with tab3:
    st.subheader("Topic Merging Hierarchy")
    st.caption("Reveals taxonomic relationships. Misleads by enforcing a binary tree on overlapping concepts.")
    df_hier = pd.DataFrame(data["hierarchy"])
    fig_hier = px.scatter(df_hier, x="Distance", y="Parent_Name", size="Distance")
    st.plotly_chart(fig_hier, use_container_width=True)

# Tab 4: 2D Embedding Projection
with tab4:
    st.subheader("2D UMAP Projection")
    st.caption("Reveals continuous semantic space. Misleads through non-linear distance distortion.")
    df_proj = pd.DataFrame(data["projections"])
    fig_proj = px.scatter(df_proj, x="x", y="y", color="topic", hover_data=["snippet"])
    st.plotly_chart(fig_proj, use_container_width=True)