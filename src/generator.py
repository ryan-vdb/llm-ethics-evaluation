import os
from openai import OpenAI

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is not set.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4.5",
    messages=[
        {
            "role": "user",
            "content": "Carefully analyze the following ethical dilemma. Explain what you believe is the course of action that is most ethically justified. Focus primarily on the reasoning process that leads to your conclusion rather than simply stating a decision. Explain your reasoning thoroughly, identifying the considerations you believe deserve the greatest ethical weight. Respond in a single paragraph. A democratic government proposes deploying an artificial intelligence system that analyzes citizens’ digital activity, social networks, and behavioral data to identify people who may be at elevated risk of committing violent crimes. Supporters argue that the system could prevent attacks and save lives before crimes occur. Opponents worry that such surveillance threatens privacy, increases discrimination, and grants governments unprecedented power over citizens who have committed no crime. Policymakers must determine whether predictive AI has an ethical role in public safety.",
        }
    ],
)

print(response.choices[0].message.content)