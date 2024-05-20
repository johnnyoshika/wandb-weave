from rag.context_precision_score import context_precision_score
import weave
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from rag.rag_model import RAGModel  # noqa

weave.init('rag-qa')


model = RAGModel(
    system_message="You are an expert in finance and answer questions related to finance, financial services, and financial markets. When responding based on provided information, be sure to cite the source."
)

questions = [
    {"question": "What significant result was reported about Zealand Pharma's obesity trial?"},
    {"question": "How much did Berkshire Hathaway's cash levels increase in the fourth quarter?"},
    {"question": "What is the goal of Highmark Health's integration of Google Cloud and Epic Systems technology?"},
    {"question": "What were Rivian and Lucid's vehicle production forecasts for 2024?"},
    {"question": "Why was the Norwegian Dawn cruise ship denied access to Mauritius?"},
    {"question": "Which company achieved the first U.S. moon landing since 1972?"},
    {"question": "What issue did Intuitive Machines' lunar lander encounter upon landing on the moon?"}
]

# We define an Evaluation object and pass our example questions along with scoring functions
evaluation = weave.Evaluation(dataset=questions, scorers=[
                              context_precision_score])
asyncio.run(evaluation.evaluate(model))
