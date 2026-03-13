# ThaiSarc_V1: Sarcasm Detection Dataset for Thai Political News Headlines

ThaiSarc_V1 is an open dataset for sarcasm detection in Thai political news headlines, built as part of a deep learning research project at NIDA. The core question was whether discriminative models or generative models handle Thai sarcasm better — and the answer turned out to depend more on language specialization than model size.

The dataset contains 1,028 headlines from 12 Thai news sources, balanced between 514 sarcastic and 514 non-sarcastic samples. Four annotators labeled the data after a structured two-hour training session, reaching 92% inter-annotator agreement. This repository includes the full dataset, benchmark results, and materials for reproducing the experiments.

---

## 🌟 Key Features

- **Balanced by design**: 514 sarcastic and 514 non-sarcastic headlines — no class imbalance to work around
- **Reliable annotation**: 92% inter-annotator agreement, using an adapted version of Hiai and Shimada (2016)
- **Full model comparison**: benchmarked across five discriminative models and four GPT-based generative models
- **Open baseline**: available for anyone working on Thai NLP, sarcasm detection, or related classification tasks

---

## 📋 Dataset Overview

| **Class**         | **Count** | **Total Words** | **Avg Length** | **Min** | **Max** |
|-------------------|-----------|-----------------|----------------|---------|---------|
| **Sarcastic**     | 514       | 8,709           | 16.94          | 7       | 27      |
| **Non-Sarcastic** | 514       | 7,950           | 15.47          | 8       | 28      |

Collected on **December 19, 2024** via [WebScraper.io](https://webscraper.io) from 12 Thai political news websites. Headlines were reviewed and annotated following the sarcasm classification framework of Hiai and Shimada (2016).

### 📰 News Sources

| **News Agency**        | **URL**                                                                 |
|------------------------|-------------------------------------------------------------------------|
| ไทยรัฐ (Thairath)       | https://www.thairath.co.th/news/politic                                 |
| มติชน (Matichon)        | https://www.matichon.co.th/politics                                     |
| ประชาไท (Prachatai)     | https://prachatai.com/category/politics                                 |
| The Matter              | https://thematter.co/category/social/politics                           |
| ไทยพีบีเอส (Thai PBS)   | https://www.thaipbs.or.th/news/politics                                 |
| MCOT                    | https://tna.mcot.net/category/politics                                  |
| เดลินิวส์ (Daily News)  | https://www.dailynews.co.th/news_group/politics                         |
| ข่าวสด (Khaosod)        | https://www.khaosod.co.th/politics                                      |
| ไทยโพสต์ (Thai Post)    | https://www.thaipost.net/politics                                       |
| แนวหน้า (Naewna)        | https://www.naewna.com/politics                                         |
| ผู้จัดการ (MGR Online)  | https://mgronline.com/politics                                          |
| อีจัน (Ejan)            | https://www.ejan.co/category/politics                                   |

---

## 📥 Usage

```bash
git clone https://github.com/KunakornMart/ThaiSarc_V1.git
```

```python
import pandas as pd

data = pd.read_csv("path/to/ThaiSarc_V1.csv")
print(data.head())
```

Thai text tokenization can be handled with [PyThaiNLP](https://github.com/PyThaiNLP/pythainlp). The experiments here used the **NewMM** tokenizer, which processed all 1,028 headlines in under 1 second and worked well for this headline-length text.

---

## 🎯 Research Questions

Three things this project set out to explore:

1. How well do discriminative deep learning models perform on Thai sarcasm classification?
2. Do generative models (GPT-based) improve meaningfully with fine-tuning on this task?
3. Does training on Thai-specific data give encoder models an edge over larger, general-purpose models?

Short answer to the third one: yes, clearly.

---

## 🔧 Methodology

**Data collection**
Headlines were scraped from 12 news sites using WebScraper.io, targeting political news sections. Stratified random sampling was used to pull roughly equal numbers from each source.

**Annotation**
Four annotators completed a two-hour training session before labeling, using the sarcasm classification framework from Hiai and Shimada (2016). A separate set of 100 headlines was used for calibration. Final inter-annotator agreement on the main dataset: **92%**.

**Train/validation/test split**
70% train / 15% validation / 15% test (719 / 154 / 155 headlines).

**Discriminative model training**
All models trained in TensorFlow 2.17.1 on Google Colab with GPU. Optimizer: Adam (lr=0.0005), batch size 32, up to 20 epochs with early stopping on val_loss (patience=3). Results reported as mean ± SD over 5 runs.

**Generative model evaluation**
GPT-based models tested via OpenAI API. Fine-tuned model: `gpt-4o-2024-08-06`, trained with lr=0.1, batch size 16, 4 epochs.

---

## 📊 Results

### Discriminative Models (mean ± SD, 5 runs)

| Model          | Accuracy       | Precision      | Recall         | F1-Score       |
|----------------|----------------|----------------|----------------|----------------|
| WangchanBERTa  | 86.57 ± 2.25%  | 84.12 ± 4.19%  | 90.62 ± 3.91%  | 87.11 ± 2.02%  |
| CNN-BiLSTM     | 79.10 ± 1.39%  | 78.11 ± 3.09%  | 78.25 ± 5.81%  | 77.96 ± 1.92%  |
| BiLSTM         | 76.31 ± 1.35%  | 73.88 ± 2.76%  | 80.34 ± 4.81%  | 76.82 ± 1.37%  |
| CNN            | 74.90 ± 1.87%  | 76.46 ± 4.43%  | 73.94 ± 7.87%  | 74.72 ± 2.38%  |
| GRU            | 72.72 ± 4.12%  | 88.47 ± 4.88%  | 51.79 ± 12.43% | 64.10 ± 8.82%  |

### Generative Models

| Model                    | Accuracy | Precision | Recall | F1-Score |
|--------------------------|----------|-----------|--------|----------|
| Fine-Tuned GPT-4o        | 81.11%   | 81.39%    | 98.59% | 89.17%   |
| GPT-4o (non-fine-tuned)  | 67.09%   | 90.32%    | 36.84% | 52.33%   |
| GPT-4o-mini              | 60.00%   | 100.00%   | 18.42% | 31.11%   |
| GPT-3.5-turbo            | 54.83%   | 100.00%   | 7.89%  | 14.63%   |

---

## 🔍 Main Takeaway

WangchanBERTa came out on top overall — 86.57% accuracy and 87.11% F1, better than every other model including fine-tuned GPT-4o. The fine-tuned GPT-4o had the highest F1 (89.17%) driven by very high recall (98.59%), but its accuracy lagged behind at 81.11%, meaning it over-predicted sarcasm more than WangchanBERTa did.

The pattern across generative models is consistent: without fine-tuning, GPT-based models struggled badly on Thai sarcasm (GPT-3.5 F1: 14.63%). Fine-tuning helped substantially, but Thai-specific pretraining still gave WangchanBERTa a clearer overall edge for this dataset.

---

## 📈 Possible Uses

- Thai NLP benchmarking and model comparison
- Sarcasm and irony detection research
- Content moderation and tone classification experiments
- Baseline for future work on broader Thai political discourse

---

## 👥 Contributors

- Kunakorn Pruksakorn
- Niwat Wuttisrisiriporn
- Hafiz Benraheem
- Chalard Lertkittisuk

**Advisor**: Thitirat Siriborvornratanakul

---

## 📧 Contact

Questions or collaboration: [KunakornMart](https://github.com/KunakornMart)
