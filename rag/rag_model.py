import weave
from weave import Model
from rag.get_most_relevant_documents import get_most_relevant_document
from openai import OpenAI


class RAGModel(Model):

    system_message: str
    model_name: str = "gpt-3.5-turbo-1106"

    @weave.op()
    # note: `question` will be used later to select data from our evaluation rows
    def predict(self, question: str) -> dict:
        context = get_most_relevant_document(question)
        client = OpenAI()
        query = f"""Use the following information to answer the subsequent question. If the answer cannot be found, write "I don't know."
        Context:
        \"\"\"
        {context}
        \"\"\"
        Question: {question}"""
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            response_format={"type": "text"},
        )
        answer = response.choices[0].message.content
        return {'answer': answer, 'context': context}
