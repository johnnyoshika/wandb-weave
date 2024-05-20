from openai import OpenAI


def docs_to_embeddings(docs: list) -> list:
    openai = OpenAI()
    document_embeddings = []
    for doc in docs:
        response = (
            openai.embeddings.create(input=doc, model="text-embedding-3-small")
            .data[0]
            .embedding
        )
        document_embeddings.append(response)
    return document_embeddings
