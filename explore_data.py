"""Quick data exploration script to understand dataset formats."""
import os

print("=" * 60)
print("1. KEYSTROKE DATA")
print("=" * 60)
with open(r"Keyboard_Strokes_DSL-StrongPasswordData.csv") as f:
    lines = f.readlines()[:3]
    for l in lines:
        print(l.strip()[:200])
    print(f"Total lines: {len(open(r'Keyboard_Strokes_DSL-StrongPasswordData.csv').readlines())}")

print("\n" + "=" * 60)
print("2. MOUSE DATA")
print("=" * 60)
with open(r"Mouse-Dynamics-Challenge-master\training_files\user7\session_0041905381") as f:
    for i, l in enumerate(f):
        if i < 6:
            print(l.strip())

print("\n" + "=" * 60)
print("3. WEB LOG DATA (human)")
print("=" * 60)
with open(r"web_bot_detection_dataset\phase1\data\web_logs\humans\access_1.log") as f:
    for i, l in enumerate(f):
        if i < 3:
            print(l.strip()[:300])

print("\n" + "=" * 60)
print("4. WEB LOG DATA (bot)")
print("=" * 60)
with open(r"web_bot_detection_dataset\phase1\data\web_logs\bots\access_advanced_bots.log") as f:
    for i, l in enumerate(f):
        if i < 3:
            print(l.strip()[:300])

print("\n" + "=" * 60)
print("5. NETWORK INTRUSION DATA")
print("=" * 60)
with open(r"Network Intrusion Dataset\Friday-WorkingHours-Morning.pcap_ISCX.csv", encoding="latin-1") as f:
    for i, l in enumerate(f):
        if i < 3:
            print(l.strip()[:400])

print("\n" + "=" * 60)
print("6. BROWSER FINGERPRINT (dataset.json sample)")
print("=" * 60)
import json
with open(r"Browser Fingerprint Dataset\dataset.json") as f:
    data = json.load(f)
if isinstance(data, list):
    print(f"JSON array, {len(data)} items")
    print("First item keys:", list(data[0].keys()) if data else "empty")
    if data:
        import json as j
        print(j.dumps(data[0], indent=2)[:500])
elif isinstance(data, dict):
    print(f"JSON object, keys: {list(data.keys())[:20]}")
