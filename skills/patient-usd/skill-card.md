## Description: <br>
Inspect, display, transform, or consume USD stages and `patient_twin.yaml` bundles exported by `patient_digital_twin`, including embedded CT, centerlines, and Isaac Sim viewing. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to inspect and consume patient USD assets and bundles, preserving coordinate frames, units, visibility, CT metadata, and centerlines, and to view them in Isaac Sim. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** No <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>
Risk: Applying a transform or unit conversion twice can silently misplace anatomy relative to CT or the simulator. <br>
Mitigation: The skill requires reading declared frames and units and reopening edited outputs to check transforms, visibility, and CT alignment. <br>

## Reference(s): <br>
- [Isaac for Healthcare Digital Twin](https://github.com/isaac-for-healthcare/i4h-digital-twin) <br>
- [Working with patient USD files](https://github.com/isaac-for-healthcare/i4h-digital-twin/blob/main/patient-digital-twin/docs/usd.md) <br>

## Skill Output: <br>
**Output Type(s):** [Shell commands, Python code, Analysis] <br>
**Output Format:** [Markdown with inline bash and Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
Pending NVSkills-Eval Tier 3 live agent evaluation. <br>

## Evaluation Tasks: <br>
5 positive tasks are defined in `evals/evals.json`. They have not been run against live agents yet. <br>

## Evaluation Metrics Used: <br>
NVSkills-Eval dimensions: Security, Correctness, Discoverability, Effectiveness, and Efficiency. <br>

## Evaluation Results: <br>
Pending. Only Tier 1 static validation has run; see `BENCHMARK.md`. No agent scores are reported until NVSkills-Eval completes. <br>

## Skill Version(s): <br>
0.9.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
