from openai import OpenAI
import weave
import json


@weave.op()
async def context_precision_score(question, model_output):
    context_precision_prompt = """Given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as "1" if useful and "0" if not with json output. 
    Output in only valid JSON format.

    question: {question}
    context: {context}
    answer: {answer}
    verification: """

    client = OpenAI()

    prompt = context_precision_prompt.format(
        question=question,
        context=model_output['context'],
        answer=model_output['answer'],
    )

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    response_message = response.choices[0].message
    response = json.loads(response_message.content)
    return {
        "verdict": int(response["verdict"]) == 1,
    }
