
# LMSYS - Chatbot Arena Human Preference Predictions

This competition challenges you to predict which responses users will prefer in a head-to-head battle between chatbots powered by large language models (LLMs). You'll be given a dataset of conversations from the [Chatbot Arena](https://chat.lmsys.org/), where different LLMs generate answers to user prompts. By developing a winning machine learning model, you'll help improve how chatbots interact with humans and ensure they better align with human preferences.


Large language models (LLMs) are rapidly entering our lives, but ensuring their responses resonate with users is critical for successful interaction. This competition presents a unique opportunity to tackle this challenge with real-world data and help us bridge the gap between LLM capability and human preference.

We utilized a large dataset collected from Chatbot Arena, where users chat with two anonymous LLMs and choose the answer they prefer. Your task in this competition is to predict which response a user will prefer in these head-to-head battles.

This challenge aligns with the concept of "reward models" or "preference models" in reinforcement learning from human feedback (RLHF). Previous research has identified limitations in directly prompting an existing LLM for preference predictions. 
These limitations often stem from biases such as :
- favoring responses presented first (position bias), 
- being overly verbose (verbosity bias), or 
- exhibiting self-promotion (self-enhancement bias).

We encourage you to explore various machine-learning techniques to build a model that can effectively predict user preferences. Your work will be instrumental in developing LLMs that can tailor responses to individual user preferences, ultimately leading to more user-friendly and widely accepted AI-powered conversation systems.

## 1 Solution

### Baseline


## Readings:

### [tweet](https://x.com/dk21/status/1826292289930674590)

Notes from LMSYS - Chatbot Arena Human Preference Predictions[@kaggle](https://x.com/kaggle)
competition top solutions (I didn’t participate). Goal is to predict which LLM response to a user prompt is better (as judged by Chat Arena users). Needs to be optimized to run for limited time on Kaggle (2xT4 GPUs). Tricks and findings: - Distillation from larger models (70B > 9B) for the 1st place team - Pseudo-labeling (similar to distillation except the signal is less granular) for many other top teams - Averaging LORA weights across 5-fold trained models (!) - With BF16 and optimizer with kahan summation support, it's possible to train 7B models using single A100 80G, 9B models requires two A100s. Found this great implementation by

[@benjamin_warner](https://x.com/benjamin_warner)

: [https://optimi.benjaminwarner.dev/kahan_summation/…](https://t.co/IRv2OAa8d2) - Many top teams started fine-tuning from reward models (vs. instruct models) - Adding to reading backlog - reward modeling approach, from the models used by many top teams: [https://rlhflow.github.io/posts/2024-05-29-multi-objective-reward-modeling/…](https://t.co/4wepu8BzaV) - Adding to reading backlog - tascj training code [https://github.com/tascj/kaggle-lmsys-chatbot-arena…](https://t.co/ytHXnEbLBm) - Many teams disabled the softcapping on Gemma2 without hurt to performance - Many teams struggled with finetuning Gemma2-27B There was a bit of drama with a leak revealed last moment, but seems to have been resolved well. Congrats to the participants and winners!

