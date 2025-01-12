import streamlit as st
from create_embeddings import embedd_text, index

st.set_page_config(
    page_title="Kwoot",
)

def search_pinecone(query_text, namespace):

    query_embedding = embedd_text(query_text)
    search_result = index.query(vector=query_embedding, top_k=3, include_metadata=True, namespace=namespace)
    return search_result

# Streamlit app
st.title("Kwoot Search Engine")
st.write("Une très belle description.")

# Select box for namespace
book = st.selectbox(
    "Livre:",
    ("Le Temps de l\'innocence", "Les Suppliantes", "Les Septs contre Thèbes", "Traité théologico-politique"),
)
book_table = {"Le Temps de l\'innocence": "wharton", "Les Suppliantes": "suppliantes", "Les Septs contre Thèbes": "septs_contre_thebes", "Traité théologico-politique": "spinoza"}
book_storage_url = "https://github.com/Monnandre/Kwoot/blob/main/Texts/"
# Input field for the query
query_text = st.text_area("Rechercher:", placeholder="Etre ou ne pas être telle est la question")

# Button to trigger search
if st.button("Go"):
    if query_text.strip():  # Ensure the input is not empty
        with st.spinner("Att je compile..."):
            book_namespace = book_table[book]
            st.divider()
            results = search_pinecone(query_text, book_namespace)
            st.write("### Résultats:")
            
            if "matches" in results:
                for i, match in enumerate(results["matches"]):
                    col1, col2 = st.columns([2, 1])
                    col1.write(f"##### **{i+1}. Score:** {(match['score'] * 100):.01f}%")
                    col2.write(f"[Lien vers le texte]({book_storage_url}{book_namespace}.txt#L{match['metadata']['line_start']})")
                    st.write(f"###### {match['metadata']['text']}")
                    st.write("---")
                    
            else:
                st.write("No results found.")
    else:
        st.warning("Please enter a query before searching.")
