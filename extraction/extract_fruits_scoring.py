import weave

# Define a scoring function to compare our model predictions with a ground truth label.


@weave.op()
def fruit_name_score(target: dict, model_output: dict) -> dict:
    return {'correct': target['fruit'] == model_output['fruit']}
