"""Single-stimulus trial list for the interactive-viewer MOS app - replaces
generate_trial_pairs.py's pairwise scheme logic entirely. No pairing needed: every row in
manifest_geometry.csv (single + mixed instances alike) becomes exactly one trial item.

CLI:  python -m pipeline.generate_trial_items --manifest <dir>/manifest_geometry.csv --out mos_app/trial_items.csv
"""
import csv
import argparse

TRIAL_ITEM_FIELDS = ['trial_id', 'subject', 'clip', 'distortion_type', 'severity', 'asset_id']


def generate_trial_items(manifest_path):
    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))
    return [
        dict(trial_id=i, subject=r['subject'], clip=r['clip'],
            distortion_type=r['distortion_type'], severity=r['severity'], asset_id=r['asset_id'])
        for i, r in enumerate(rows)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    items = generate_trial_items(a.manifest)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_ITEM_FIELDS)
        w.writeheader(); w.writerows(items)
    print(f'{len(items)} trial items written to {a.out}')


if __name__ == '__main__':
    main()
