"""
test_hallucination.py
=====================
Tests the full pipeline on a question likely to cause hallucination.
Run: python test_hallucination.py
"""

from primary_agent import query_primary_agent
from verification_agents import run_all_verification_agents
from aggregation_module import run_aggregation
from voting_module import run_voting

question = "What was the exact GDP of the Mongol Empire in 1270 AD and who was the finance minister at the time?"

print("\n" + "="*60)
print("HALLUCINATION DETECTION TEST")
print("="*60)
print(f"Question: {question}\n")

p = query_primary_agent(question)
v = run_all_verification_agents(question)
a = run_aggregation(p, v)
r = run_voting(a)

print("\n" + "="*60)
print("FINAL RESULT:")
print("="*60)
print(f"  Agreement Score  : {a['agreement_score']}")
print(f"  Final Risk Score : {r['final_verdict']['final_risk_score']}")
print(f"  Risk Level       : {r['final_verdict']['risk_level']}")
print(f"  Action           : {r['final_verdict']['action']}")
print(f"  {r['final_verdict']['label']}")