from langchain_huggingface import HuggingFaceEmbeddings

em = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 

user_docs = ["Taj Mahal is in Agra", "Eiffel Tower is in Paris", "Statue of Liberty is in New York", "Colosseum is in Rome", "Sydney Opera House is in Sydney"
]
user_query = "Where is Eiffel Tower located?"

user_docs_embeddings = em.embed_documents(user_docs)
user_query_embedding = em.embed_query(user_query)

#cosine similarity

from sklearn.metrics.pairwise import cosine_similarity

similarity_scores = cosine_similarity([user_query_embedding], user_docs_embeddings)

index, score = sorted(list(enumerate(similarity_scores[0])), key=lambda x: x[1], reverse=True)[0]


#most_similar_doc_index = similarity_scores.argmax()

#max_score, idx = 0, -1
#for index, score in lst{enumerate(similarity_scores[0])}:
#    if score > max_score:
#        max_score = score
#        idx = index


# print(user_docs[index])


# print(user_docs[most_similar_doc_index])

print("........START....")
print("User Query: ", user_query)
print("LLM Result", user_docs[index])
print("Similarity Score: ", score)
print("........END....")


