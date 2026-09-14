"""Merge render_array.slurm's per-task output directories into the final single/ + mixed/
layout, once all 30 array tasks have completed. Each task wrote to its own task_<idx>/ (see
render_array.slurm's docstring for why - manifest writes would clobber each other otherwise).

Run on VSC after checking every task's state (e.g. `sacct -M wice -j <jobid> --format=JobID,State`
shows COMPLETED for all 30 .batch entries):
    python -m vsc.merge_outputs $AVATARVERSE_OUT/final
"""
import os, sys, csv, glob, shutil

EXPECTED_PER_TASK = 40   # 1 reference + 27 single + 12 mixed... reference+single share one
                         # manifest (28 rows total there) + 12 mixed rows = 40


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.environ.get('AVATARVERSE_OUT', ''), 'final')
    final_single = os.path.join(base, 'single'); os.makedirs(final_single, exist_ok=True)
    final_mixed = os.path.join(base, 'mixed'); os.makedirs(final_mixed, exist_ok=True)

    single_rows, mixed_rows = [], []
    incomplete = []
    task_dirs = sorted(glob.glob(os.path.join(base, 'task_*')),
                       key=lambda p: int(p.rsplit('_', 1)[1]))
    for td in task_dirs:
        n_single_files = len(glob.glob(os.path.join(td, 'single', '*.mp4')))
        n_mixed_files = len(glob.glob(os.path.join(td, 'mixed', '*.mp4')))
        if n_single_files != 28 or n_mixed_files != 12:
            incomplete.append((os.path.basename(td), n_single_files, n_mixed_files))
            continue   # don't merge a partial task - rerun it instead (see below)
        for f in glob.glob(os.path.join(td, 'single', '*.mp4')):
            shutil.move(f, os.path.join(final_single, os.path.basename(f)))
        for f in glob.glob(os.path.join(td, 'mixed', '*.mp4')):
            shutil.move(f, os.path.join(final_mixed, os.path.basename(f)))
        with open(os.path.join(td, 'manifest_single.csv')) as fh:
            single_rows.extend(list(csv.DictReader(fh)))
        with open(os.path.join(td, 'manifest_mixed.csv')) as fh:
            mixed_rows.extend(list(csv.DictReader(fh)))
        shutil.rmtree(td)

    if incomplete:
        print(f'!! {len(incomplete)} task(s) incomplete - NOT merged, rerun these array indices:')
        for name, ns, nm in incomplete:
            print(f'   {name}: {ns} single files (want 28), {nm} mixed files (want 12)')

    with open(os.path.join(base, 'manifest_single.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'subject', 'clip', 'distortion_type', 'severity', 'params', 'n_frames'])
        w.writeheader(); w.writerows(single_rows)
    with open(os.path.join(base, 'manifest_mixed.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['file', 'subject', 'clip', 'distortion_type', 'severity', 'reference', 'params', 'n_frames'])
        w.writeheader(); w.writerows(mixed_rows)

    print(f'{len(single_rows)} single + {len(mixed_rows)} mixed = '
         f'{len(single_rows)+len(mixed_rows)} files merged into {base}')
    print(f'({len(task_dirs)-len(incomplete)}/{len(task_dirs)} tasks merged)')


if __name__ == '__main__':
    main()
