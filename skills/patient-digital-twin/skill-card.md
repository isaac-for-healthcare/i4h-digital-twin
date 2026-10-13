## Description: <br>
Import patient anatomy (segmentations, NV-Segment, NV-Generate, STL/OBJ), attach CT, control visibility, extract vessel centerlines, and export USD or i4h-workflows patient bundles with the `patient_digital_twin` package. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to turn medical imaging or meshes into named anatomy meshes, optional vessel centerlines, and USD or i4h-workflows patient bundles, and to change the `patient_digital_twin` API and CLI safely. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** No <br>
**Credential Type(s):** [None identified for the bundled s0011 sample. NV-Segment and NV-Generate require upstream model checkouts and weights; NV-Generate downloads its datasets unless `MONAI_DATA_DIRECTORY` is set.] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>
Risk: Exports are static geometry and native-HU CT and could be mistaken for a validated tissue simulation or an anonymized dataset. <br>
Mitigation: The skill instructs the agent not to describe exports as validated simulations or anonymization steps, and to report missing inputs instead of switching patients or backends. <br>

## Reference(s): <br>
- [Isaac for Healthcare Digital Twin](https://github.com/isaac-for-healthcare/i4h-digital-twin) <br>
- [Patient Digital Twin README](https://github.com/isaac-for-healthcare/i4h-digital-twin/blob/main/patient-digital-twin/README.md) <br>
- [Patient Digital Twin architecture](https://github.com/isaac-for-healthcare/i4h-digital-twin/blob/main/patient-digital-twin/docs/architecture.md) <br>

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
