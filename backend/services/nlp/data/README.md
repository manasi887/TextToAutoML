# NLP Intent Classification Dataset

## Purpose

This dataset is designed to train a BERT-based intent classifier for the TextToAutoML system. The classifier enables the system to understand natural language user requests and automatically identify the underlying machine learning task type (classification, regression, clustering, or unknown).

## Intent Classes

The dataset contains four intent classes:

### 1. **Classification**
Requests where the user wants to predict discrete categories or binary outcomes. Examples include:
- Predicting whether a customer will churn
- Detecting fraudulent transactions
- Identifying loan approval likelihood
- Predicting disease presence
- Classifying email as spam or not spam
- Predicting employee attrition

**Count:** ~50 examples

### 2. **Regression**
Requests where the user wants to predict continuous numerical values. Examples include:
- Predicting house prices
- Forecasting sales or revenue
- Estimating salary
- Predicting delivery time
- Forecasting temperature
- Predicting customer lifetime value

**Count:** ~50 examples

### 3. **Clustering**
Requests where the user wants to group or segment data into natural clusters or cohorts. Examples include:
- Customer segmentation
- Grouping similar products
- Finding user communities
- Identifying transaction patterns
- Market segmentation
- Identifying natural groupings in data

**Count:** ~50 examples

### 4. **Unknown**
Requests that are genuinely ambiguous or lack sufficient context to determine the ML task type. These examples represent situations where user intent is unclear or multiple interpretations are possible. This class is essential for teaching the classifier to recognize when human clarification is needed.

**Count:** ~50 examples

## Dataset Statistics

- **Total Examples:** 200+
- **Class Distribution:** Roughly balanced (~50 examples per class)
- **Format:** CSV (Comma-Separated Values)

## CSV Format

The dataset contains exactly two columns:

| Column | Description |
|--------|-------------|
| `text` | Natural language user request (string) |
| `intent` | ML task type label: `classification`, `regression`, `clustering`, or `unknown` |

**Example rows:**
```
text,intent
Will my customers leave us?,classification
What will be the house price?,regression
I want to group my customers,clustering
Can you analyze this data?,unknown
```

## Why the Unknown Class is Necessary

The `unknown` class serves several critical purposes:

1. **Real-world scenarios**: Users don't always have clear ML objectives. The classifier needs to recognize ambiguous requests.

2. **Calibration**: By including genuinely uncertain examples, the model learns confidence bounds and when to defer decisions.

3. **Fallback mechanism**: The unknown class acts as a safety net, allowing the system to ask for clarification rather than misclassifying requests.

4. **Robustness**: Training with unknown examples prevents the model from overconfidently assigning labels to out-of-distribution requests.

## Use for BERT Fine-Tuning

This dataset will be used to fine-tune a BERT transformer model with the following workflow:

1. **Tokenization**: User text examples will be tokenized using BERT's WordPiece tokenizer
2. **Embedding**: Each example will be encoded as BERT embeddings
3. **Fine-tuning**: The classification head will be trained on top of the frozen or unfrozen BERT backbone
4. **Validation**: The model will be evaluated on a held-out test set
5. **Deployment**: The trained model will power the NLP intent detection service in TextToAutoML

## Quality Assurance

The dataset has been created with the following principles:

- **Diversity of language**: Examples use varying levels of formality, question formats, and domain-specific terminology
- **No artificial repetition**: Each example is unique and represents realistic user input
- **No explicit task labels in text**: Examples describe what users want without explicitly saying "classification" or "regression"
- **Balanced representation**: All four classes are represented proportionally
- **Non-technical language**: Examples prioritize how real users (non-ML experts) would phrase requests

## Integration with TextToAutoML

Once fine-tuned, the resulting BERT model will:
- Accept natural language user queries
- Classify them into one of the four intent categories
- Feed the predicted intent to the task-specific ML pipeline (AutoML engines, preprocessing routines, etc.)
- Enable fully automated ML workflows based on plain-English descriptions

---

**Created:** 2026-09-05  
**Version:** 1.0  
**Status:** Ready for BERT fine-tuning
