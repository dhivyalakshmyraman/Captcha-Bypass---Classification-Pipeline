import os, json, re

# Check web log data - parse Apache log format
with open(r'web_bot_detection_dataset\phase1\data\web_logs\humans\access_1.log') as f:
    lines = f.readlines()[:10]

# Apache log format with session id field
pattern = re.compile(
    r'^(.*?) - \[(.*?)\] "(.*?)" (\d+) (\d+) "(.*?)" (.*?) "(.*?)"$'
)

for line in lines[:5]:
    m = pattern.match(line.strip())
    if m:
        print('PARSED:', m.group(1)[:10], m.group(2)[:25], m.group(3)[:30], 'session:', m.group(7)[:30], 'ua:', m.group(8)[:50])
    else:
        print('NO MATCH:', line.strip()[:120])

# Count log lines by session
session_counts = {}
session_uas = {}
for logdir in ['humans', 'bots']:
    d = os.path.join(r'web_bot_detection_dataset\phase1\data\web_logs', logdir)
    for fn in os.listdir(d):
        with open(os.path.join(d, fn)) as f:
            for line in f:
                m = pattern.match(line.strip())
                if m:
                    sid = m.group(7)
                    ua = m.group(8)
                    if sid != '-':
                        session_counts[sid] = session_counts.get(sid, 0) + 1
                        session_uas[sid] = ua

print(f'\nTotal unique sessions in logs: {len(session_counts)}')

# Load annotations
annot_file = r'web_bot_detection_dataset\phase1\annotations\humans_and_advanced_bots\train'
annots = {}
with open(annot_file) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            annots[parts[0]] = parts[1]

print(f'Annotations train: {len(annots)} sessions')
print(f'Labels: {set(annots.values())}')

matched = sum(1 for s in session_counts if s in annots)
print(f'Sessions matching annotations: {matched}')

# Show sample session stats
for sid in list(annots.keys())[:3]:
    cnt = session_counts.get(sid, 0)
    ua = session_uas.get(sid, 'N/A')
    print(f'  Session {sid}: {annots[sid]}, {cnt} requests, UA: {ua[:60]}')

# Also check phase2
print('\nPhase2 annotations:')
for subdir in ['humans_and_advanced_bots', 'humans_and_moderate_bots']:
    d = os.path.join(r'web_bot_detection_dataset\phase2\annotations', subdir)
    if os.path.exists(d):
        for fn in os.listdir(d):
            fp = os.path.join(d, fn)
            with open(fp) as f:
                lines = f.readlines()
            print(f'  {subdir}/{fn}: {len(lines)} entries')
            if lines:
                print(f'    Sample: {lines[0].strip()}')

# Network data - check all label types
print('\n\nNetwork intrusion labels across files:')
all_labels = {}
net_dir = r'Network Intrusion Dataset'
for fn in os.listdir(net_dir):
    if fn.endswith('.csv'):
        with open(os.path.join(net_dir, fn), encoding='latin-1') as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split(',')
                lbl = parts[-1].strip()
                all_labels[lbl] = all_labels.get(lbl, 0) + 1
for lbl, cnt in sorted(all_labels.items(), key=lambda x: -x[1]):
    print(f'  {lbl}: {cnt}')
