from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Documents
User_docs = [
    "Climate change is mainly caused by greenhouse gases released from burning fossil fuels and industrial activities.",
    "The effects of climate change include rising global temperatures, melting glaciers, and increasing sea levels.",
    "Climate change leads to extreme weather events such as heatwaves, floods, droughts, and stronger storms.",
    "Climate change disrupts ecosystems, damages agriculture, and increases risks to human health worldwide.",
    "Using renewable energy, planting trees, and reducing emissions are key solutions to slow climate change.",
    "Climate change causes rising temperatures, melting ice caps, and sea level rise, impacting life globally.",
    "Global warming results in extreme weather like droughts, floods, and heatwaves, which are major climate impacts.",
    "Reducing carbon emissions through clean energy and sustainability can help control climate change effects."
]

# Query
User_query = "What are the effects of climate change?"

# Generate embeddings
doc_embeddings = embeddings.embed_documents(User_docs)
query_embedding = embeddings.embed_query(User_query)

# Compute cosine similarity
similarity_scores = cosine_similarity(
    [query_embedding],
    doc_embeddings
)[0]

# Filter, sort, and select top 3
top_3_results = sorted(
    [(index, score) for index, score in enumerate(similarity_scores) if score >= 0.55],
    key=lambda x: x[1],
    reverse=True
)[:3]

# Output
print("........START....")
print("User Query:", User_query)
print("\nTop 3 Similar Results (score ≥ 0.55):\n")

if not top_3_results:
    print("No document matched the threshold.")
else:
    for rank, (doc_index, score) in enumerate(top_3_results, start=1):
        print(f"{rank}. Document:")
        print(f"   Text: {User_docs[doc_index]}")
        print(f"   Similarity Score: {round(score, 3)}\n")

print("........END....")
