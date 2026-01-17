"""
Problem: We have so approaches to solve a single problem. In this case we have to make a prediction system that can classify the support ticket into one category, and we have three three famous approaches. But which approach fits best?

Objective: To automatically classify support tickets into relevant categories by exploring zero-shot, few-shot, and fine-tuned language model approaches, and compare their effectiveness in producing accurate and reliable ticket tags
"""

from .zero_shot_classification import zero_shot_predict
from .few_shot_learning import few_shot_prompt
from .fine_tuning import predict_ticket

ticket = "I entered the correct mail address for rest password, but main isn't reciving."

zero_shot_prediction = zero_shot_predict(ticket)
few_shot_prediction = few_shot_prompt(ticket)
fine_tune_model_prediction = predict_ticket(ticket)

print({
    "zero_shot_prediction": zero_shot_prediction,
    "few_shot_prediction": few_shot_prediction,
    "fine_tuned_model_prediction": fine_tune_model_prediction
})

"""
Findings:
- The zero-shot approach achieved 2/5 accuracy without requiring any labeled training data.
- The few-shot approach improved performance to 3/5 accuracy by providing contextual examples.
- The fine-tuned approach achieved the highest accuracy (4/5) and proved to be the most stable and effective method.

"""