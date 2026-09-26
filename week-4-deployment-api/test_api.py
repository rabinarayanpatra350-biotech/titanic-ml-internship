"""
Titanic Survival Prediction API - Endpoint Tests
=================================================
Week 4 Task (Capstone): verifies the deployed API end-to-end using
Flask's test client - no server process needed.

Run: python test_api.py
"""

import json
from app import app

client = app.test_client()

print("=" * 70)
print("TEST 1 - GET /health")
print("=" * 70)
r = client.get("/health")
print(f"HTTP {r.status_code}")
print(json.dumps(r.get_json(), indent=2))

print("\n" + "=" * 70)
print("TEST 2 - GET / (documentation endpoint)")
print("=" * 70)
r = client.get("/")
print(f"HTTP {r.status_code}")
doc = r.get_json()
print("Service :", doc["service"])
print("Endpoints:", ", ".join(doc["endpoints"]))

print("\n" + "=" * 70)
print("TEST 3 - POST /predict (three passengers)")
print("=" * 70)
passengers = [
    {"label": "3rd-class man, 22, cheap ticket ('Jack')",
     "payload": {"pclass": 3, "sex": "male", "age": 22, "sibsp": 1,
                 "parch": 0, "fare": 7.25, "embarked": "S", "title": "Mr"}},
    {"label": "1st-class woman, 38, expensive ticket ('Rose')",
     "payload": {"pclass": 1, "sex": "female", "age": 38, "sibsp": 1,
                 "parch": 0, "fare": 71.28, "embarked": "C", "title": "Mrs"}},
    {"label": "7-year-old boy travelling with family ('Master')",
     "payload": {"pclass": 3, "sex": "male", "age": 7, "sibsp": 1,
                 "parch": 2, "fare": 21.08, "embarked": "S", "title": "Master"}},
]
results = []
for p in passengers:
    r = client.post("/predict", json=p["payload"])
    body = r.get_json()
    print(f"\n{p['label']}")
    print(f"HTTP {r.status_code}")
    print(json.dumps(body, indent=2))
    results.append((p["label"], r.status_code, body.get("survival_prediction"),
                    body.get("survival_probability")))

print("\n" + "=" * 70)
print("TEST 4 - POST /predict with invalid body (must return 400)")
print("=" * 70)
r = client.post("/predict", data="not json")
print(f"HTTP {r.status_code} (expected 400)")
print(json.dumps(r.get_json(), indent=2))

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
ok = all(code == 200 for _, code, _, _ in results)
for label, code, pred, prob in results:
    print(f"  [{code}] {label:52s} -> prediction {pred} (p={prob})")
print(f"Invalid-input handling: {'PASS' if r.status_code == 400 else 'FAIL'}")
print("ALL API TESTS PASSED" if ok else "SOME TESTS FAILED")
