# Patient twin workflow validation

Validated on 2026-09-21 using the s0011 CT from
`~/dev/data/Totalsegmentator_dataset_small_v201/s0011/ct.nii.gz`.
`examples/export_patient_twin.py` ran NV-Segment CT_BODY to build HumanBody,
then `HumanBody.export_patient_twin()` generated this bundle.

Both visible Isaac Sim runs used this directory's `patient_twin.yaml` and the
existing local controllers in `/home/mallan/dev/i4h-workflows`; no learned-policy
checkpoint was used. Each requested one successful episode with one allowed
attempt, within the existing scene step cap. Both exited with status 0 and
reported `1/1 episodes succeeded (1 attempts)`.

| Mode | Node completion step | Workflow completion step | Recorded frames |
| --- | ---: | ---: | ---: |
| demo | 118 | 119 | 118 |
| validate_fluoroscopy | 72 | 73 | 72 |

Recordings:

- `/home/mallan/dev/i4h-workflows/runs/endoluminal_navigation/20260921_112709/verify.hdf5`
- `/home/mallan/dev/i4h-workflows/runs/endoluminal_navigation/20260921_113100/verify.hdf5`

Both passed `i4h-dataset inspect --segments`. Each contains one successful
segment, three action and joint-state channels, and 1024×1024 fluoroscopy frames.
Joint states were finite. The demo exercised insertion, rotation, and C-arm orbit.
The fluoroscopy validation held the catheter controls fixed and changed C-arm
orbit by approximately 0.54 radians. Sampled camera frames were visually inspected:
the CT projection and catheter were visible, and anatomy projection changed with
C-arm rotation. Mean first-to-last pixel differences were 3.295 and 4.917 on the
0–255 RGB scale, respectively.

This establishes rollout compatibility with the two tested workflow modes; it
does not establish clinical accuracy or learned-policy navigation performance.
The patient package and example regression tests passed: 160 passed, 2 skipped.

Reproduce from the i4h-workflows checkout, replacing `BUNDLE` with this directory:

```bash
./run.sh endoluminal_navigation --mode demo --episodes 1 --attempts 1 \
  --patient-twin BUNDLE/patient_twin.yaml --record verify.hdf5
./run.sh endoluminal_navigation --mode validate_fluoroscopy --episodes 1 --attempts 1 \
  --patient-twin BUNDLE/patient_twin.yaml --record verify.hdf5
```
