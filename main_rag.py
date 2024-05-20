import weave
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from rag.rag_model import RAGModel  # noqa

weave.init('rag-qa')


model = RAGModel(
    system_message="You are an expert in finance and answer questions related to finance, financial services, and financial markets. When responding based on provided information, be sure to cite the source."
)
model.predict(
    "What significant result was reported about Zealand Pharma's obesity trial?")
