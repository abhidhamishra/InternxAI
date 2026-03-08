from langchain_huggingface import HuggingFaceEmbeddings

cm = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

result = cm.embed_query("Where is Eiffel Tower located?")

print(result)



