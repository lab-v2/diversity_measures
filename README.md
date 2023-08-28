# Diversity Measures: Domain Independent Proxies for Failure in Language Model Queries
<a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/Python-3.11+-1f425f.svg?color=purple">
</a>

## **Computing Infrastructure used**
### **Hardware**:
- **CPU**: 9th Gen Intel(R) Core(TM) i7-9750H, 2.60GHz, 6 Cores, 12 Logical Processors  
- **GPU**: Intel(R) UHD Graphics 630
- **Memory**: 12GB RAM
- **Operating System**: Windows 10

### **Software**:
- **Python Version**: 3.11.4

#### **Relevant Libraries**
- **Scikit-learn**: 1.2.2  
- **Tensorflow**: 2.12.0  
- **Sentence Transformers**: 2.12.0  
- **Imbalanced Learn**: 0.11.0  
- **Pandas**: 2.0.1  
- **OpenAI**: 0.27.2

## Instructions for running experiment
### 1. Environment setup
Install the required packages from PyPI:
```powershell
pip install -r requirements.txt
```
### 2. Data collection
In this step, GPT-3.5's responses are collected and evaluated.  
First, create an API key on OpenAI.  
Then, create a `.env` file with the following contents:
```
OPENAI_ORGANIZATION=[YOUR OPENAI ORGANIZATION ID]
OPENAI_API_KEY=[YOUR OPENAI API KEY]
```
Then, run the following script in Powershell:  
```powershell
scripts/powershell/data-collection
```

Results from data collection and analysis are stored in `./cache`. We have provided the prefilled since data collection might be quite expensive. In order to run a fresh test, the files in `./cache` and its subfolders must first be removed.  
To do that, run the following script in Powershell:  
```powershell
scripts/powershell/clear-cache
```

### 3. Data Analysis
Data analysis is performed in various Jupyter Notebooks.

**(Section 5.2)** Shannon Entropy diversity measures are evaluated on DRAW-1K, CSQA and Last Letters dataset with 5 different temperature settings in the following notebook:
```
scripts/analysis/eval_entropy.ipynb
```  
<br/>

**(Section 5.2)** Gini impurity diversity measures are evaluated on DRAW-1K, CSQA and Last Letters dataset with 5 different temperature settings in the following notebook:
```
scripts/analysis/eval_gini.ipynb
```  
<br/>

**(Section 5.3)** Centroid-based diversity measures are evaluated on DRAW-1K, CSQA and Last Letters dataset with 5 different temperature settings in the following notebook:
```
scripts/analysis/eval_centroid.ipynb
```  
<br/>

**(Section 5.4)** Experiments regarding few-shot prompting are performed in the following notebook:
```
scripts/analysis/few_shot.ipynb
```  
<br/>

**(Section 5.5)** Experiments regarding few-shot chain-of-thought prompting are performed in the following notebook. This experiment is only performed on DRAW-1K since it was the only dataset which provided intermediary steps which allows for chain-of-thought style few-shot prompting:
```
scripts/analysis/few_shot_cot.ipynb
```  
<br/>

**(Section 5.6)** The effects of ablating various diversity measures for the 10 Layer Multi-Perceptron model are analyzed in the following notebook:
```
scripts/analysis/ablation_test.ipynb
```
<br/>

**(Section 5.6)** The performance of various machine learning models are tried in the following notebook:
```
scripts/analysis/classifier_analysis.ipynb
```

**(Section 5.6)** The precision-recall curves for the 10 Layer Multi-Perceptron model are produced in the following notebook:
```
scripts/analysis/pr_curves.ipynb
```