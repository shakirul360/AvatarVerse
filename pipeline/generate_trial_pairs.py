"""Turns manifest_single.csv + manifest_mixed.csv (the flat list of rendered stimuli, split into
single/ and mixed/ folders - see vsc/merge_outputs.py) into the actual trial pairs a participant
will see in the pairwise-comparison MOS tool. Three comparison schemes, per the professor's
reference tool's methodology (pairwise 2AFC, not absolute rating):

1. reference_vs_distorted - the clean reference against each of the 27 single-distortion
   variants (9 types x 3 severities) for that (subject, clip). Answers "how much does this
   distortion/severity hurt perceived quality vs. the original."
2. adjacent_severity - mild-vs-moderate and moderate-vs-severe, within each of the 9 single
   distortion types, skipping the reference entirely. Answers "can participants reliably tell
   severity levels apart from each other," not just from clean.
3. reference_vs_mixed - the clean reference against each of the 12 mixed (compound) variants
   (6 curated pairs x 2 severity presets). Answers "does compounding two distortions hurt
   quality differently than either alone would suggest" - a genuinely different question from
   schemes 1/2, which only ever vary one distortion at a time.

That's 27 + 18 + 12 = 57 pairs per (subject, clip); 30 (subject, clip) combos in Phase 3 (6
subjects x 5 clips) = 1,710 pairs total across the whole corpus. One participant session is
scoped to all 57 pairs for ONE (subject, clip) - a manageable single-sitting length - with
different sessions/participants assigned different (subject, clip) combos so the full corpus
gets covered across several people rather than any one person seeing all 1,710.

Left/right (A/B) side is randomized per pair with a fixed seed, so it's reproducible but not
positionally biased toward, say, "the distorted one is always on the right."

Run: python3 -m pipeline.generate_trial_pairs
Reads:  <DATASET_DIR>/manifest_single.csv, <DATASET_DIR>/manifest_mixed.csv
Writes: <DATASET_DIR>/trial_pairs.csv
"""
import os, sys, csv, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import distortions as dist   # DISTORTION_TYPES, MIXED_PAIRS, MIXED_SEVERITY_PRESETS

DATASET_DIR = os.environ.get(
    'AVATARVERSE_FINAL_DATASET',
    '/Users/leeon/AvatarVerse/Datasets/THuman2.0/mos_dataset_textured_v1')
SINGLE_MANIFEST = os.path.join(DATASET_DIR, 'manifest_single.csv')
MIXED_MANIFEST = os.path.join(DATASET_DIR, 'manifest_mixed.csv')
OUT_PATH = os.path.join(DATASET_DIR, 'trial_pairs.csv')
SEED = 7

SEVERITIES = ['mild', 'moderate', 'severe']


def load_manifests():
    with open(SINGLE_MANIFEST) as f:
        single_rows = list(csv.DictReader(f))
    with open(MIXED_MANIFEST) as f:
        mixed_rows = list(csv.DictReader(f))
    # single/ and mixed/ are separate folders now - keep that in the stored path so a
    # consuming tool doesn't need to guess which one a filename lives in
    single_idx = {(r['subject'], r['clip'], r['distortion_type'], r['severity']):
                 f"single/{r['file']}" for r in single_rows}
    mixed_idx = {(r['subject'], r['clip'], r['distortion_type'], r['severity']):
                f"mixed/{r['file']}" for r in mixed_rows}
    return single_idx, mixed_idx


def build_pairs(single_idx, mixed_idx):
    rng = random.Random(SEED)
    subjects_clips = sorted({(k[0], k[1]) for k in single_idx})
    pairs = []
    pair_id = 0

    for subject, clip in subjects_clips:
        ref_file = single_idx[(subject, clip, 'none', 'none')]

        # scheme 1: reference vs each single-distortion variant
        for dist_type in dist.DISTORTION_TYPES:
            for severity in SEVERITIES:
                dist_file = single_idx[(subject, clip, dist_type, severity)]
                pairs.append(make_row(pair_id, subject, clip, 'reference_vs_distorted',
                                      dist_type, 'reference', severity, ref_file, dist_file, rng))
                pair_id += 1

        # scheme 2: adjacent severities within each single distortion type
        for dist_type in dist.DISTORTION_TYPES:
            for lo, hi in [('mild', 'moderate'), ('moderate', 'severe')]:
                lo_file = single_idx[(subject, clip, dist_type, lo)]
                hi_file = single_idx[(subject, clip, dist_type, hi)]
                pairs.append(make_row(pair_id, subject, clip, 'adjacent_severity',
                                      dist_type, lo, hi, lo_file, hi_file, rng))
                pair_id += 1

        # scheme 3: reference vs each mixed (compound) variant
        for t1, t2 in dist.MIXED_PAIRS:
            mixed_type = f'{t1}+{t2}'
            for sev in dist.MIXED_SEVERITY_PRESETS:
                mixed_sev = f'{sev}+{sev}'
                mixed_file = mixed_idx[(subject, clip, mixed_type, mixed_sev)]
                pairs.append(make_row(pair_id, subject, clip, 'reference_vs_mixed',
                                      mixed_type, 'reference', mixed_sev, ref_file, mixed_file, rng))
                pair_id += 1

    return pairs


def make_row(pair_id, subject, clip, comparison_type, dist_type, level_x, level_y, file_x, file_y, rng):
    """Randomly assign file_x/file_y to the A/B slots so the 'interesting' side isn't always
    on the same edge - a real 2AFC design consideration, not just cosmetic."""
    if rng.random() < 0.5:
        video_a, video_b, level_a, level_b = file_x, file_y, level_x, level_y
    else:
        video_a, video_b, level_a, level_b = file_y, file_x, level_y, level_x
    return dict(pair_id=pair_id, subject=subject, clip=clip, comparison_type=comparison_type,
               distortion_type=dist_type, video_a=video_a, video_b=video_b,
               level_a=level_a, level_b=level_b)


def main():
    single_idx, mixed_idx = load_manifests()
    pairs = build_pairs(single_idx, mixed_idx)
    fieldnames = ['pair_id', 'subject', 'clip', 'comparison_type', 'distortion_type',
                 'video_a', 'video_b', 'level_a', 'level_b']
    with open(OUT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)

    n_subjects_clips = len({(p['subject'], p['clip']) for p in pairs})
    print(f'{len(pairs)} trial pairs written to {OUT_PATH}')
    print(f'({n_subjects_clips} (subject, clip) combos x 57 pairs each: '
         f'27 reference_vs_distorted + 18 adjacent_severity + 12 reference_vs_mixed)')
    print(f'One session = all 57 pairs for one (subject, clip) combo.')


if __name__ == '__main__':
    main()
