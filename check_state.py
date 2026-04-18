import os
import sys

processed = r'd:\Captcha Bypass Project\data\processed'
files = os.listdir(processed)

print('=== PROCESSED FILES ===')
for f in sorted(files):
    size = os.path.getsize(os.path.join(processed, f))
    print(f'  {f}: {size:,} bytes')

print()
print('=== MISSING FILES ===')
expected = [
    'fp_pipeline.joblib', 'fp_test.parquet',
    'net_pipeline.joblib', 'net_test.parquet',
    'wb_pipeline.joblib', 'wb_test.parquet',
    'meta_model.joblib', 'meta_test.parquet'
]
missing = [f for f in expected if f not in files]
for f in missing:
    print(f'  MISSING: {f}')

if not missing:
    print('  All files present!')

print()
print('=== REPORTS DIR ===')
reports = r'd:\Captcha Bypass Project\reports'
if os.path.exists(reports):
    rfiles = os.listdir(reports)
    if rfiles:
        for f in sorted(rfiles):
            print(f'  {f}')
    else:
        print('  (empty)')
else:
    print('  (does not exist)')
