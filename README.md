# Part 2: HRS-to-NLSY79 cognitive prediction

This project tests whether models trained on HRS can predict later cognitive scores in NLSY79, and whether HRS-derived predictions of cognitive change improve models that use NLSY79 life-course information.

| Data stage | Contents |
| --- | --- |
| **01 - Harmonized cohort data** | 32 processed tables covering cognition, shared variables, life-course features, source information, and analysis variants. |
| **02 - Model-ready cohort pairs** | HRS: 52,023 pairs from 8,299 people. NLSY79: 2,305 people, one pair each. |

The current analysis uses baseline ages 48-55, actual follow-up of 4-16 years, and observed 27-point cognition. It has 29 primary shared inputs, 77 expanded shared candidates, and 82 additional NLSY79 candidates. The two previously shared data archives remain unchanged.

- [Planning and original project presentation](planning/README.md)
- [Step-by-step AI agent handoff](planning/AGENT_HANDOFF.md)
- [Unified metadata](metadata/README.md)
- [Model code and persistence changes](planning/CODE_ADAPTATION.md)

Data preparation and software pilot checks are complete for this scope. Full modeling and locked-test evaluation are the next stage.
