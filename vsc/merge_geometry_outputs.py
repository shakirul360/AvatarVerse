"""Merge geometry_export_array.slurm's per-task output directories into the final single/ +
mixed/ layout, once all 30 array tasks have completed. Each task wrote to its own task_<idx>/
(same clobbering-manifest reasoning as merge_outputs.py's docstring).

Unlike the video track, each "instance" here is a DIRECTORY of many files (base.drc,
frames/*.drc, texture.jpg, meta.json, metrics.json), not one file - a truncated/crashed instance
wouldn't show up as a wrong top-level file count the way a missing .mp4 would. Validated per
instance instead: meta.json exists and parses, base.drc/texture.jpg exist, and frames/ has
exactly meta.json['n_delta_frames'] .drc files (matches pipeline/geometry_export.py's
write_geometry_bundle() - frame 0's positions live in base.drc, so there are n_frames-1 delta
files, not n_frames).

Run on VSC after checking every task's state (e.g. `sacct -M wice -j <jobid> --format=JobID,State`
shows COMPLETED for all 30 .batch entries):
    python -m vsc.merge_geometry_outputs $AVATARVERSE_OUT/final_geom
"""
import os, sys, csv, glob, json, shutil

EXPECTED_PER_TASK = 37   # 1 reference + 24 single + 12 mixed (see distortions.py: 8 distortion
                         # types x 3 severities = 24; 6 mixed pairs x 2 severity presets = 12)


def _validate_instance(inst_dir):
    """Returns an error string, or None if the instance bundle is complete."""
    meta_path = os.path.join(inst_dir, 'meta.json')
    if not os.path.exists(meta_path):
        return 'missing meta.json'
    try:
        meta = json.load(open(meta_path))
    except (json.JSONDecodeError, OSError) as e:
        return f'unreadable meta.json ({e})'
    if not os.path.exists(os.path.join(inst_dir, 'base.drc')):
        return 'missing base.drc'
    if not os.path.exists(os.path.join(inst_dir, 'texture.jpg')):
        return 'missing texture.jpg'
    n_delta_frames = meta.get('n_delta_frames')
    if n_delta_frames is None:
        return 'meta.json missing n_delta_frames'
    n_found = len(glob.glob(os.path.join(inst_dir, 'frames', '*.drc')))
    if n_found != n_delta_frames:
        return f'frames/ has {n_found} files, want {n_delta_frames}'
    return None


def _validate_and_collect(kind_dir, kind):
    """kind_dir: task_<idx>/single or task_<idx>/mixed. Returns (valid_instance_dirs, errors)."""
    errors = []
    valid = []
    for inst_dir in sorted(glob.glob(os.path.join(kind_dir, '*'))):
        if not os.path.isdir(inst_dir):
            continue
        err = _validate_instance(inst_dir)
        if err:
            errors.append(f'{kind}/{os.path.basename(inst_dir)}: {err}')
        else:
            valid.append(inst_dir)
    return valid, errors


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.environ.get('AVATARVERSE_OUT', ''), 'final_geom')
    final_single = os.path.join(base, 'single'); os.makedirs(final_single, exist_ok=True)
    final_mixed = os.path.join(base, 'mixed'); os.makedirs(final_mixed, exist_ok=True)

    rows = []
    incomplete = []
    task_dirs = sorted(glob.glob(os.path.join(base, 'task_*')),
                       key=lambda p: int(p.rsplit('_', 1)[1]))
    for td in task_dirs:
        single_valid, single_errs = _validate_and_collect(os.path.join(td, 'single'), 'single')
        mixed_valid, mixed_errs = _validate_and_collect(os.path.join(td, 'mixed'), 'mixed')
        errs = single_errs + mixed_errs
        if len(single_valid) + len(mixed_valid) != EXPECTED_PER_TASK or errs:
            incomplete.append((os.path.basename(td), len(single_valid), len(mixed_valid), errs))
            continue   # don't merge a partial task - rerun it instead (see below)

        for inst_dir in single_valid:
            shutil.move(inst_dir, os.path.join(final_single, os.path.basename(inst_dir)))
        for inst_dir in mixed_valid:
            shutil.move(inst_dir, os.path.join(final_mixed, os.path.basename(inst_dir)))
        with open(os.path.join(td, 'manifest_geometry.csv')) as fh:
            rows.extend(list(csv.DictReader(fh)))
        shutil.rmtree(td)

    if incomplete:
        print(f'!! {len(incomplete)} task(s) incomplete - NOT merged, rerun these array indices:')
        for name, ns, nm, errs in incomplete:
            print(f'   {name}: {ns+nm} valid instances (want {EXPECTED_PER_TASK})')
            for e in errs[:5]:
                print(f'      {e}')
            if len(errs) > 5:
                print(f'      ...and {len(errs)-5} more')

    manifest_fields = ['asset_id', 'kind', 'subject', 'clip', 'distortion_type', 'severity',
                       'reference', 'params', 'bundle_bytes', 'chamfer_mean', 'chamfer_max', 'hausdorff_max']
    with open(os.path.join(base, 'manifest_geometry.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=manifest_fields)
        w.writeheader(); w.writerows(rows)

    print(f'{len(rows)} instances merged into {base}')
    print(f'({len(task_dirs)-len(incomplete)}/{len(task_dirs)} tasks merged)')


if __name__ == '__main__':
    main()
