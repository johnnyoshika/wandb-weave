import weave
from dotenv import load_dotenv
from extract_fruit import extract_fruit


# Load environment variables from .env file
load_dotenv()


weave.init('intro-example')
sentence = "There are many fruits that were found on the recently discovered planet Goocrux. There are neoskizzles that grow there, which are purple and taste like candy."
extract_fruit(sentence)
