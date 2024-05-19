import asyncio
import weave
from dotenv import load_dotenv
from extract_fruits_model import ExtractFruitsModel


# Load environment variables from .env file
load_dotenv()

weave.init('intro-example')

model = ExtractFruitsModel(model_name='gpt-3.5-turbo-1106',
                           prompt_template='Extract fields ("fruit": <str>, "color": <str>, "flavor": <str>) from the following text, as json: {sentence}')

sentence = "There are many fruits that were found on the recently discovered planet Goocrux. There are neoskizzles that grow there, which are purple and taste like candy."
print(asyncio.run(model.predict(sentence)))
