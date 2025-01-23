# ThaiSarc_V1: A Comprehensive Dataset for Sarcasm Detection in Thai Political News Headlines

Welcome to **ThaiSarc_V1**, a groundbreaking resource specifically designed for sarcasm detection in Thai political news headlines. This project represents a significant milestone in Natural Language Processing (NLP) for Thai-language tasks, combining state-of-the-art deep learning and generative AI models to address the unique challenges of sarcasm identification in the Thai context. ThaiSarc_V1 not only contributes to the advancement of computational linguistics but also fosters media literacy by enabling nuanced understanding of sarcastic language in political discourse.

---

## 🌟 Key Features
- **Balanced Dataset**: ThaiSarc_V1 consists of 1,028 carefully curated Thai political news headlines, evenly split into 514 sarcastic and 514 non-sarcastic samples to ensure unbiased analysis.
- **High Annotation Reliability**: A mutual agreement rate of 92% among annotators highlights the robustness of the dataset.
- **Novel Dataset**: The first publicly available dataset focusing on sarcasm detection in Thai political news headlines, offering a valuable resource for Thai NLP research.
- **Comprehensive Benchmarks**: Evaluation includes cutting-edge discriminative models (e.g., WangchanBERTa, CNN-BiLSTM) and generative models (e.g., fine-tuned GPT-4o).
- **Broad Applications**: Designed for sentiment analysis, content moderation, and improving media literacy.

---

## 📋 Dataset Overview

| **Class**        | **Number of Items** | **Total Words** | **Average Length** | **Min Length** | **Max Length** |
|-------------------|---------------------|-----------------|---------------------|----------------|----------------|
| **Sarcastic**     | 514                 | 8,709           | 16.94              | 7              | 27             |
| **Non-Sarcastic** | 514                 | 7,950           | 15.47              | 8              | 28             |

The dataset was collected on **December 19, 2024**, using [WebScraper.io](https://webscraper.io) from 12 prominent Thai news agency websites. Each headline was reviewed, annotated, and validated for sarcasm using an adaptation of the sarcasm detection framework by Hiai and Shimada (2016). The careful preprocessing and annotation processes ensure high-quality data for training and testing models.

### 📰 News Agencies

ThaiSarc_V1 headlines were sourced from the following news agencies:

| **News Agency**     | **Website URL**                                               |
|---------------------|-------------------------------------------------------------|
| ไทยรัฐ (Thairath)      | [https://www.thairath.co.th/news/politic](https://www.thairath.co.th/news/politic) |
| มติชน (Matichon)      | [https://www.matichon.co.th/politics](https://www.matichon.co.th/politics) |
| ประชาไท (Prachatai)    | [https://prachatai.com/category/politics](https://prachatai.com/category/politics) |
| The Matter           | [https://thematter.co/category/social/politics](https://thematter.co/category/social/politics) |
| ไทยพีบีเอส (Thai PBS)  | [https://www.thaipbs.or.th/news/politics](https://www.thaipbs.or.th/news/politics) |
| MCOT                | [https://tna.mcot.net/category/politics](https://tna.mcot.net/category/politics) |
| เดลินิวส์ (Daily News) | [https://www.dailynews.co.th/news_group/politics](https://www.dailynews.co.th/news_group/politics) |
| ข่าวสด (Khaosod)      | [https://www.khaosod.co.th/politics](https://www.khaosod.co.th/politics) |
| ไทยโพสต์ (Thai Post)  | [https://www.thaipost.net/politics](https://www.thaipost.net/politics) |
| แนวหน้า (Naewna)      | [https://www.naewna.com/politics](https://www.naewna.com/politics) |
| ผู้จัดการ (MGR Online)| [https://mgronline.com/politics](https://mgronline.com/politics) |
| อีจัน (Ejan)         | [https://www.ejan.co/category/politics](https://www.ejan.co/category/politics) |

---

## 📥 Download and Usage

Access ThaiSarc_V1 on GitHub: [ThaiSarc_V1 GitHub Repository](https://github.com/KunakornMart/ThaiSarc_V1)

### Steps to Use the Dataset:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/KunakornMart/ThaiSarc_V1.git
   ```

2. **Load the Dataset**:
   ```python
   import pandas as pd

   # Load the dataset
   data = pd.read_csv('path/to/ThaiSarc_V1.csv')

   # Display the first 5 rows
   print(data.head())
   ```

3. **Preprocess and Tokenize**:
   Utilize libraries such as **PyThaiNLP** for tokenization and processing Thai text effectively.

---

## 🎯 Objectives

The primary goal of this project is to compare the efficiency of **discriminative deep learning models** (e.g., CNN, BiLSTM, WangchanBERTa) and **generative AI models** (e.g., GPT-4o) in sarcasm detection, providing a comprehensive evaluation of their strengths and weaknesses.

This study also aims to answer key research questions:
1. How effectively do discriminative models perform in sarcasm detection for Thai political news headlines?
2. Can generative AI models, such as GPT-4o, surpass discriminative models in identifying sarcasm?
3. What are the implications of applying these models to real-world tasks, such as sentiment analysis and misinformation detection?

---

## 🚀 Models and Benchmarks

### Discriminative Models
- **WangchanBERTa**: Tailored for Thai language processing, this transformer-based model achieved an accuracy of **86.57%** and F1-score of **87.11%**.
- **CNN-BiLSTM Hybrid**: Developed to combine CNN’s pattern recognition with BiLSTM’s sequential understanding, achieving an accuracy of **79.10%** and F1-score of **77.96%**.
- **CNN**: Delivered an accuracy of **74.90%** and F1-score of **74.72%**.
- **GRU**: Achieved an accuracy of **72.72%** and F1-score of **64.10%**.
- **BiLSTM**: Reached an accuracy of **76.31%** and F1-score of **76.82%**.

### Generative Models
- **Fine-Tuned GPT-4o**: Demonstrated strong performance with an accuracy of **81.11%** and F1-score of **89.17%**.
- **Non-Fine-Tuned GPT-4o**: Delivered moderate performance with an accuracy of **67.09%** and F1-score of **52.33%**.
- **GPT-4o-mini**: Achieved an accuracy of **60.00%** and F1-score of **31.11%**.
- **GPT-3.5-turbo**: Reached an accuracy of **54.83%** and F1-score of **14.63%**.

### Evaluation Metrics
Key metrics included **Accuracy**, **Precision**, **Recall**, and **F1-Score**, ensuring a comprehensive analysis of model performance.

---

## 🔧 Methodology

### Data Preparation
- Tokenization: Implemented with **NewMM** (PyThaiNLP), achieving **93% accuracy** and processing 1,028 headlines in **3.25 seconds**.
- Preprocessing: Headlines were normalized and encoded for model training.

### Annotation Process
- Conducted by four trained annotators.
- Achieved high inter-annotator reliability (92%) through consensus discussions.

### Experimental Settings
- **Discriminative Models**: Implemented using TensorFlow 2.17.1 on Google Colab with GPU acceleration.
- **Generative Models**: Fine-tuned on GPT-4o (latest version) for sarcasm detection.

---

## 📊 Results

### Discriminative Models Performance
| Model            | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|-------------------|-------------|---------------|------------|--------------|
| WangchanBERTa    | 86.57       | 84.12         | 90.62      | 87.11        |
| CNN-BiLSTM       | 79.10       | 78.11         | 78.25      | 77.96        |
| CNN              | 74.90       | 76.46         | 73.94      | 74.72        |
| GRU              | 72.72       | 88.47         | 51.79      | 64.10        |
| BiLSTM           | 76.31       | 73.88         | 80.34      | 76.82        |

### Generative Models Performance
| Model                       | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|-----------------------------|-------------|---------------|------------|--------------|
| Fine-Tuned GPT-4o           | 81.11       | 81.39         | 98.59      | 89.17        |
| GPT-4o (non-fine-tuned)     | 67.09       | 90.32         | 36.84      | 52.33        |
| GPT-4o-mini                 | 60.00       | 100.00        | 18.42      | 31.11        |
| GPT-3.5-turbo               | 54.83       | 100.00        | 7.89       | 14.63        |

---

## 📈 Applications

- **Media Literacy**: Enhances understanding of sarcasm in Thai political discourse.
- **NLP Research**: Serves as a benchmark for testing and improving sarcasm detection models.
- **Content Moderation**: Assists in filtering sarcastic or misleading content in online platforms.
- **Sentiment Analysis**: Supports improved sentiment categorization in Thai language processing.
- **Combatting Misinformation**: Provides tools to identify sarcasm in political narratives, reducing the spread of false information.
- **Future Work**: Potential areas include expanding the dataset, incorporating multimodal sarcasm detection (e.g., combining text and images), and applying models to broader domains beyond politics.

---

## 👥 Contributors
- **Kunakorn Pruksakorn**
- **Niwat Wuttisrisiriporn**
- **Hafiz Benraheem**
- **Chalard Lertkittisuk**
- **Advisor**: Thitirat Siriborvornratanakul

---

## 📧 Contact
For inquiries, please contact [KunakornMart](https://github.com/KunakornMart).
