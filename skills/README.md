# Repository Skills

Repository-specific agent skills live here. `.claude/skills` and `.codex/skills`
both reference this shared directory.

| Skill | Use when |
| --- | --- |
| [patient-digital-twin](patient-digital-twin/SKILL.md) | Importing anatomy (segmentations, NV-Segment, NV-Generate, meshes), attaching CT, visibility, centerlines, USD/bundle export, and the CLI |
| [patient-usd](patient-usd/SKILL.md) | Inspecting or consuming patient USD, handling transforms/CT/centerlines, viewing in Isaac Sim |

The skills link to the maintained package documentation and checked-in examples.
They do not require a separate global skill installation. Agents that support
named skill invocation can use `$patient-digital-twin` or `$patient-usd`.

## Skill format

Each skill follows the shared i4h NV-BASE layout used by
[i4h-workflows](https://github.com/isaac-for-healthcare/i4h-workflows/tree/main/skills):

| File | Contents |
| --- | --- |
| `SKILL.md` | Front matter with `name`, `description`, `license`, and `metadata` (`author`, `version`, `tags`), followed by the instructions |
| `evals/evals.json` | Eval prompts in the NV-ACES schema: `id`, `question`, `expected_skill`, `ground_truth`, `expected_behavior` |
| `BENCHMARK.md` | NVSkills-Eval benchmark report |
| `skill-card.md` | Skill card |
| `skill.oms.sig` | NVIDIA OMS signature over the skill files |

`BENCHMARK.md`, `skill-card.md`, and `skill.oms.sig` are produced by the NVSkills-Eval
publication pipeline and are signed together. Do not hand-edit published copies; pre-commit
and the link check skip them. Until NVSkills-Eval runs, `BENCHMARK.md` holds local Tier 1
results only, `skill-card.md` marks the evaluation results as pending, and there is no
`skill.oms.sig`. The signature also covers `SKILL.md` and `evals/evals.json`, so any change
to a skill needs a new NVSkills-Eval run to refresh the signed files.

Validate locally before pushing (see the
[i4h-workflows testing guide](https://github.com/isaac-for-healthcare/i4h-workflows/blob/main/TESTING.md)
for setup and the eval schema check):

```bash
uv tool install --index-url https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple nv-base
uv tool install --index-url https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple skillspector
nv-base validate skills -r cli json -o /tmp/i4h-digital-twin-nvbase \
  --catalog-path "$PWD" --no-dedup --no-llm -c
```
