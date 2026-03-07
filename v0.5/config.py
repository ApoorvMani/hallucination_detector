"""
config.py - all settings for v0.5

CSC8208 Multi-Agent Hallucination Detection Framework — Newcastle University
v0.5: ground truth fact checking — no injection yet, no hallucination prompting
"""

# how many rounds to run per question
TOTAL_ROUNDS = 10

# model used by all agents
MODEL = "llama3.2"

# system prompt — simple factual assistant, no mention of hallucination
SYSTEM_PROMPT = "You are a factual assistant. Answer questions accurately and concisely."

# agent configs — triangle topology, same model, position is the only variable
AGENTS = {
    "agent_0": {"model": MODEL, "temperature": 0.5},
    "agent_1": {"model": MODEL, "temperature": 0.5},
    "agent_2": {"model": MODEL, "temperature": 0.5},
}

# triangle topology — every agent sees both others
TOPOLOGY = {
    "agent_0": ["agent_1", "agent_2"],
    "agent_1": ["agent_0", "agent_2"],
    "agent_2": ["agent_0", "agent_1"],
}

# 5 questions with ground truth facts
# keywords: any match found in agent answer = that fact is present
# answer: canonical correct answer — for reference only, never shown to agents
QUESTIONS = [
    {
        "question": "Who invented the telephone and in what year?",
        "answer":   "Alexander Graham Bell invented the telephone in 1876.",
        "facts": {
            "person": ["alexander graham bell", "graham bell", "bell"],
            "year":   ["1876"],
        }
    },
    {
        "question": "Who was the first person to walk on the moon and in what year?",
        "answer":   "Neil Armstrong was the first person to walk on the moon in 1969.",
        "facts": {
            "person": ["neil armstrong", "armstrong"],
            "year":   ["1969"],
        }
    },
    {
        "question": "What is the speed of light in a vacuum?",
        "answer":   "The speed of light in a vacuum is approximately 299,792 kilometres per second.",
        "facts": {
            "value": ["299,792", "299792", "300,000", "300000", "3 x 10", "3x10", "2.998"],
        }
    },
    {
        "question": "What is the boiling point of water at sea level in Celsius?",
        "answer":   "The boiling point of water at sea level is 100 degrees Celsius.",
        "facts": {
            "value": ["100"],
        }
    },
    {
        "question": "In what year did World War 2 end?",
        "answer":   "World War 2 ended in 1945.",
        "facts": {
            "year": ["1945"],
        }
    },
]
