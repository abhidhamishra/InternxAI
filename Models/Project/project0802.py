from langchain_huggingface import HuggingFaceEmbeddings
em = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

User_docs = [ "Climate change is mainly caused by greenhouse gases released from burning fossil fuels and industrial activities.", 
"The effects of climate change include rising global temperatures, melting glaciers, and increasing sea levels.",
 "Climate change leads to extreme weather events such as heatwaves, floods, droughts, and stronger storms.", 
 "Climate change disrupts ecosystems, damages agriculture, and increases risks to human health worldwide.", 
 "Using renewable energy, planting trees, and reducing emissions are key solutions to slow climate change.", 
 "Climate change causes rising temperatures, melting ice caps, and sea level rise, impacting life globally.",
  "Global warming results in extreme weather like droughts, floods, and heatwaves, which are major climate impacts.", 
  "Reducing carbon emissions through clean energy and sustainability can help control climate change effects." ]

User_query = "What are the effects of climate change?"

User_docs_embeddings = em.embed_documents(User_docs)
User_query_embedding = em.embed_query(User_query)

from sklearn.metrics.pairwise import cosine_similarity

similarity_scores = cosine_similarity([User_query_embedding], User_docs_embeddings)
results = list(enumerate(similarity_scores[0]))

sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

# ✅ Fix: slice sorted_results, not a tuple
top_results = sorted_results[:3]

THRESHOLD = 0.55

print("........START....")
print(f"User Query: {User_query}\n")

# ✅ Fix: filter each of the top results against the threshold
filtered_results = [(idx, score) for idx, score in top_results if score >= THRESHOLD]

if filtered_results:
    print(f"Top {len(filtered_results)} Similar Results (score ≥ {THRESHOLD}):\n")
    for i, (doc_index, score) in enumerate(filtered_results):
        print(f"{i+1}. Document:")
        print(f"   Text: {User_docs[doc_index]}")
        print(f"   Similarity Score: {round(score, 3)}\n")
else:
    print("NA")

print("........END....")