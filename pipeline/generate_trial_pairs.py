"""Turns manifest.csv (the flat list of 225 rendered stimuli) into the actual trial pairs a
participant will see in the pairwise-comparison MOS tool. Two comparison schemes, per the
professor's reference tool's methodology (pairwise 2AFC, not absolute rating):

1. reference_vs_distorted - the clean reference against each of the 24 distorted variants
   for that (subject, clip). Answers "how much does this distortion/severity hurt perceived
   quality vs. the original."
2. adjacent_severity - mild-vs-moderate and moderate-vs-severe, within each distortion type,
   skipping the reference entirely. Answers "can participants reliably tell severity levels
   apart from each other," not just from clean - a different, complementary question.

That's 24 + 16 = 40 possible pairs per (subject, clip); 9 (subject, clip) combos in the pilot
= 360 pairs total across the whole corpus. One participant session is scoped to all 40 pairs
for ONE (subject, clip) - a manageable single-sitting length - with different sessions/
participants assigned different (subject, clip) combos so the full corpus gets covered across
several people rather than any one person seeing all 360.

Left/right (A/B) side is randomized per pair with a fixed seed, so it's reproducible but not
positionally biased toward, say, "the distorted one is always on the right."

Run: python3 generate_trial_pairs.py
Reads:  Datasets/THuman2.0/mos_pilot_dataset_v2/manifest.csv
Writes: Datasets/THuman2.0/mos_pilot_dataset_v2/trial_pairs.csv
"""
import os, csv, random

DATASET_DIR = '/Users/leeon/AvatarVerse/Datasets/THuman2.0/mos_pilot_dataset_v2'
MANIFEST_PATH = os.path.join(DATASET_DIR, 'manifest.csv')
OUT_PATH = os.path.join(DATASET_DIR, 'trial_pairs.csv')
SEED = 7

DISTORTION_TYPES = ['jitter', 'joint_noise', 'frame_drop', 'smoothing', 'motion_blur',
                    'mesh_decimation', 'vertex_quantization', 'param_quantization']
SEVERITIES = ['mild', 'moderate', 'severe']


def load_manifest():
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))
    # index: (subject, clip, distortion_type, severity) -> filename
    idx = {}
    for r in rows:
        key = (r['subject'], r['clip'], r['distortion_type'], r['severity'])
        idx[key] = r['file']
    return idx


def build_pairs(idx):
    rng = random.Random(SEED)
    subjects_clips = sorted({(k[0], k[1]) for k in idx})
    pairs = []
    pair_id = 0

    for subject, clip in subjects_clips:
        ref_file = idx[(subject, clip, 'none', 'none')]

        # scheme 1: reference vs each distorted variant
        for dist_type in DISTORTION_TYPES:
            for severity in SEVERITIES:
                dist_file = idx[(subject, clip, dist_type, severity)]
                pairs.append(make_row(pair_id, subject, clip, 'reference_vs_distorted',
                                       dist_type, 'reference', severity, ref_file, dist_file, rng))
                pair_id += 1

        # scheme 2: adjacent severities within each distortion type
        for dist_type in DISTORTION_TYPES:
            for lo, hi in [('mild', 'moderate'), ('moderate', 'severe')]:
                lo_file = idx[(subject, clip, dist_type, lo)]
                hi_file = idx[(subject, clip, dist_type, hi)]
                pairs.append(make_row(pair_id, subject, clip, 'adjacent_severity',
                                       dist_type, lo, hi, lo_file, hi_file, rng))
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
    idx = load_manifest()
    pairs = build_pairs(idx)
    fieldnames = ['pair_id', 'subject', 'clip', 'comparison_type', 'distortion_type',
                  'video_a', 'video_b', 'level_a', 'level_b']
    with open(OUT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairs)

    n_subjects_clips = len({(p['subject'], p['clip']) for p in pairs})
    print(f'{len(pairs)} trial pairs written to {OUT_PATH}')
    print(f'({n_subjects_clips} (subject, clip) combos x 40 pairs each: '
          f'24 reference_vs_distorted + 16 adjacent_severity)')
    print(f'One session = all 40 pairs for one (subject, clip) combo.')


if __name__ == '__main__':
    main()
