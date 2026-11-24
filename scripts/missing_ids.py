
import os
import re
import sys
import pandas as pd

if len(sys.argv) < 3:
    sys.exit(
        "Usage: python scripts/missing_ids.py "
        "manifest.csv folder1 folder2 ..."
    )

list_csv = sys.argv[1]
roots = sys.argv[2:]

have = set()
pattern = re.compile(r"^I(\d+)$")

# Collect Image IDs already present in local DICOM folders
for root in roots:
    if not os.path.exists(root):
        print(f"WARNING: Folder not found: {root}")
        continue

    for dp, dn, filenames in os.walk(root, followlinks=True):
        for directory in dn:
            match = pattern.match(directory)
            if match:
                have.add(match.group(1))

# Read selected MRI manifest
df = pd.read_csv(list_csv, dtype=str)

candidates = [
    c for c in df.columns
    if "image" in c.lower()
    and ("id" in c.lower() or "uid" in c.lower())
]

if not candidates:
    sys.exit(f"No Image ID column found. Columns: {list(df.columns)}")

column = candidates[0]

num = (
    df[column]
    .dropna()
    .astype(str)
    .str.replace(r"\D", "", regex=True)
)

listed = set(num[num != ""])

missing = sorted(listed - have)
matched = sorted(listed & have)

print("\n========== MRI IMAGE ID AUDIT ==========")
print(f"Manifest column       : {column}")
print(f"Unique manifest IDs   : {len(listed)}")
print(f"Already on disk       : {len(matched)}")
print(f"Still missing         : {len(missing)}")
print(f"Extra local IDs       : {len(have - listed)}")

os.makedirs("results/coverage", exist_ok=True)

# Save missing IDs in batches of 100
for i in range(0, len(missing), 100):
    filename = (
        f"results/coverage/ids_chunk_{i//100+1:02d}.txt"
    )
    with open(filename, "w") as f:
        f.write("\n".join("I" + x for x in missing[i:i+100]))

with open("results/coverage/ids_numeric_all.txt", "w") as f:
    f.write("\n".join(missing))

print("\nAudit files saved in results/coverage/")
