# Repository Skills

Repository-specific agent skills live here. `.claude/skills` and `.codex/skills`
both reference this shared directory.

| Skill | Use when |
| --- | --- |
| [patient-digital-twin](patient-digital-twin/SKILL.md) | Importing anatomy, attaching CT, configuring anatomy, extracting topology, and generating patient bundles |
| [patient-usd](patient-usd/SKILL.md) | Inspecting or consuming patient USD, handling transforms/CT/centerlines, viewing in Isaac Sim |

The skills link to the maintained package documentation and checked-in examples.
They do not require a separate global skill installation. Agents that support
named skill invocation can use `$patient-digital-twin` or `$patient-usd`.
