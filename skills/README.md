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
