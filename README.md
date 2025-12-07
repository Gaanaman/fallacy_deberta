# 🌍 Climate Fallacy Detector

A fine-tuned **DeBERTa-v2-xlarge** model for detecting logical fallacies in climate misinformation, with an interactive Terminal UI for real-time analysis.

[![Hugging Face](https://img.shields.io/badge/🤗%20Model-deberta--v2--xlarge--climate--fallacy-blue)](https://huggingface.co/Gaanaman/deberta-v2-xlarge-climate-fallacy)

---

## 📖 About

This project builds upon the research paper "[Detecting Fallacies in Climate Misinformation: A Technocognitive Approach to Identifying Misleading Argumentation](https://arxiv.org/abs/2405.08254)" by *Francisco Zanartu, John Cook, Markus Wagner, Julian Garcia*.

**My contribution:** Fine-tuned a **DeBERTa-v2-xlarge** model on the FLICC dataset, successfully replicating research results in climate fallacy detection across **12 fallacy classes**, and built an interactive TUI for real-time fallacy detection.

---

## 🎯 Fallacy Classes (12)

| Class | Description |
|-------|-------------|
| Ad Hominem | Attacking the person instead of the argument |
| Anecdote | Using personal experience over scientific evidence |
| Cherry Picking | Selectively choosing data |
| Conspiracy Theory | Claiming coordinated deception |
| Fake Experts | Citing unqualified sources |
| False Choice | Presenting limited options |
| False Equivalence | Equating unequal things |
| Impossible Expectations | Demanding unrealistic standards |
| Misrepresentation | Distorting facts or positions |
| Oversimplification | Reducing complex issues |
| Single Cause | Attributing to one factor |
| Slothful Induction | Ignoring strong evidence |

---

## 🚀 Quick Start

### Use the Model from HuggingFace

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("Gaanaman/deberta-v2-xlarge-climate-fallacy")
tokenizer = AutoTokenizer.from_pretrained("Gaanaman/deberta-v2-xlarge-climate-fallacy")
```

### Run the Terminal UI

```bash
cd TUI
python deberta_fallacy_detector_tui.py
```

---

## 📁 Project Structure

```
FLICC/
├── Data/                      # Original FLICC dataset
├── TUI/                       # Terminal User Interface
│   ├── deberta_fallacy_detector_tui.py
│   ├── deberta_fallacy_detector_tui.tcss
│   └── Widgets/               # Custom UI components
├── Experiments/               # Hyperparameter search notebooks
│   ├── all_search_focal.ipynb
│   ├── all_search_lora.ipynb
│   ├── all_search_lr.ipynb
│   └── all_search_wd.ipynb
├── Experiments_mps/           # Apple Silicon (MPS) experiments
├── FLICC_original_code/       # Original research code
├── train_deberta_pytorch.ipynb  # Main training notebook
├── evaluate_model.ipynb         # Model evaluation & metrics
└── best_model/                  # Trained model weights (gitignored)
```

---

## 📊 Dataset

The FLICC dataset is available in the `Data/` folder.

---

## 📚 Citation

If you use this work, please cite the original paper:

```bibtex
@misc{zanartu2024detecting,
      title={Detecting Fallacies in Climate Misinformation: A Technocognitive Approach to Identifying Misleading Argumentation}, 
      author={Francisco Zanartu and John Cook and Markus Wagner and Julian Garcia},
      year={2024},
      eprint={2405.08254},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
}
```

---

## 📬 Contact

- **Original Research:** Francisco Zanartu
- **This Implementation:** [Daniel K. Adotey](https://github.com/Gaanaman)