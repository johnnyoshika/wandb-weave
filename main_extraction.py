import asyncio
import weave
from weave.flow.scorer import MultiTaskBinaryClassificationF1
from dotenv import load_dotenv
from extraction.extract_fruits_model import ExtractFruitsModel
from extraction.extract_fruits_scoring import fruit_name_score


# Load environment variables from .env file
load_dotenv()

weave.init('intro-example')

model = ExtractFruitsModel(name='gpt3.5-turbo',
                           model_name='gpt-3.5-turbo-1106',
                           prompt_template='Extract fields ("fruit": <str>, "color": <str>, "flavor": <str>) from the following text, as json: {sentence}')

sentences = [
    "There are many fruits that were found on the recently discovered planet Goocrux. There are neoskizzles that grow there, which are purple and taste like candy.",
    "Pounits are a bright green color and are more savory than sweet.",
    "Finally, there are fruits called glowls, which have a very sour and bitter taste which is acidic and caustic, and a pale orange tinge to them."
]
labels = [
    {'fruit': 'neoskizzles', 'color': 'purple', 'flavor': 'candy'},
    {'fruit': 'pounits', 'color': 'bright green', 'flavor': 'savory'},
    {'fruit': 'glowls', 'color': 'pale orange', 'flavor': 'sour and bitter'}
]
examples = [
    {'id': '0', 'sentence': sentences[0], 'target': labels[0]},
    {'id': '1', 'sentence': sentences[1], 'target': labels[1]},
    {'id': '2', 'sentence': sentences[2], 'target': labels[2]}
]

evaluation = weave.Evaluation(
    name='fruit_eval',
    dataset=examples, scorers=[
        # Use default scoring function MultiTaskBinaryClassificationF1
        MultiTaskBinaryClassificationF1(
            class_names=["fruit", "color", "flavor"]),
        # Also use our own custom scoring function
        fruit_name_score
    ],
)
print(asyncio.run(evaluation.evaluate(model)))
