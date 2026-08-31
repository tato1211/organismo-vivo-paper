# 🧬 Organismo Vivo — Research Paper

[![DOI](https://zenodo.org/badge/1351838165.svg)](https://doi.org/10.5281/zenodo.22182769)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/tato1211/organismo-vivo-paper)

This repository contains the research paper documenting **Organismo Vivo**, a multi-agent trading framework developed through collaboration between a human trader and multiple AI coding assistants.

---

## 📄 Paper

**Title:** Edge Evolution: A Multi-Agent Trading Framework with Honest Reporting of Limitations

- 📥 **[paper_v1_english.pdf](paper_v1_english.pdf)** — Main paper (PDF)
- 📝 **[paper_v1_english.md](paper_v1_english.md)** — Markdown source
- 🔗 **DOI:** [10.5281/zenodo.22182769](https://doi.org/10.5281/zenodo.22182769)

## 🎯 Abstract

We document Organismo Vivo, a multi-agent trading framework combining reinforcement learning agents with encoder-based regime detection, validated through walk-forward testing, preregistration, and adversarial auditing. We report edge magnitudes with explicit acknowledgment of limitations, document seven production bugs found through adversarial review, and discuss the philosophical foundations of the approach.

## 📚 What's in this paper

The paper covers:

- **System architecture**: Four cooperating modules (OV/RC/AE/Hermes)
- **The respiratory state framework**: Eight discrete market states inspired by biological metaphors
- **Encoder for regime detection**: InfoNCE contrastive learning with causal padding
- **PPO agent architecture**: Action space, reward function, training protocol
- **Multi-skill ensemble (Hermes)**: Skill definition, voting mechanism, statistical validation
- **Adversarial audit process**: Seven production bugs found through external review
- **Philosophical decisions**: Why manual filters fail, why retraining degrades, the role of AI assistants
- **Lessons learned**: 10 documented lessons including 14 specific weaknesses

## 🏗️ Architecture Overview

Organismo Vivo ├── OV (core PPO agent) → Magic 777001 ├── RC (regression channel + PPO gating) → Magic 25122013 ├── AE (encoder-based regime detector) → Magic 777003 └── Hermes (supervisor + persistent memory) → Reports via Telegram


All modules communicate through shared CSV files and a centralized state machine. A watchdog process monitors all live components and restarts any that fail.

## 🎓 Citation

If you use or reference this work, please cite it. GitHub will generate citation formats automatically from the [CITATION.cff](CITATION.cff) file.

### BibTeX

```bibtex
@misc{augusto2026organismo,
  title={Edge Evolution: A Multi-Agent Trading Framework with Honest Reporting of Limitations},
  author={Augusto},
  year={2026},
  howpublished={GitHub},
  url={https://github.com/tato1211/organismo-vivo-paper},
  note={Co-authored with AI coding assistants}
}
APA
Augusto. (2026). Edge Evolution: A Multi-Agent Trading Framework with Honest Reporting of Limitations. https://doi.org/10.5281/zenodo.22182769

📖 Related Documentation
📂 Project repository — Main trading system code
📝 HISTORIA.md — Narrative history of the project
🤝 Co-authorship
This work was developed through intensive collaboration between the human author and multiple AI coding assistants:

Human: Conceptual direction, domain expertise, final validation, and all trading decisions
AI Assistants (Claude, OpenWork, Hermes Agent): Code implementation, experimental design, statistical analysis, documentation, and adversarial review
All financial decisions remain the human author's responsibility. The system operates on a demo account.

⚠️ Disclaimer
The trading system described in this paper operates on a demo account with real-time market data. No results presented here imply fitness for real capital deployment. The edge magnitudes reported are illustrative; the methodology and validation framework are the primary contributions.

This paper does not constitute investment advice. Trading involves substantial risk of loss. Past performance does not guarantee future results.

⚖️ License
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License [blocked].

You are free to:

✅ Share — copy and redistribute the material
✅ Adapt — remix, transform, and build upon the material
Under the following terms:

📝 Attribution — You must give appropriate credit
🚫 NonCommercial — You may not use the material for commercial purposes without permission
🔄 ShareAlike — Derivatives must use the same license
For commercial use inquiries, contact the author.

📬 Contact
GitHub Issues: github.com/tato1211/organismo-vivo-paper/issues
Project: Organismo Vivo
🌟 Acknowledgments
We thank the adversarial auditor (Claude, in adversarial mode) for identifying seven production bugs during the audit process. We thank the meta-agent (Hermes) for providing continuous operational support and persistent memory across sessions. And we thank the open-source community for the libraries that made this project possible: Stable-Baselines3, MetaTrader5, PyTorch, NumPy, Pandas, and scikit-learn.
