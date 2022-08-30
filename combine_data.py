
import json


intents = []
with open('./backend/data/intent_knust.json','r') as f:
    data = json.loads(f.read())
    for intent in data["intents"]:
        intents.append({
            "intent": intent["intent"],
            "text": intent["text"],
            "responses": intent["responses"]
        })

with open('./backend/data/Intent_general.json','r') as f:
    data = json.loads(f.read())
    for intent in data["intents"]:
        intents.append({
            "intent": intent["intent"],
            "text": intent["text"],
            "responses": intent["responses"]
        })



with open('./backend/data/Intent.json','w') as f:
    f.write(json.dumps({
        "intents": intents
    }))