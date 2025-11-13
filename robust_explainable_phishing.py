# # Colab setup: Ensure these libraries are installed
# You must run the following commands in your environment (e.g., Colab or terminal)
# !pip install transformers datasets accelerate torch numpy pandas scikit-learn lime spellchecker sentence-transformers spacy
# !python -m spacy download en_core_web_sm
# !python -m spacy link en_core_web_sm en --force

import os
import sys
import json
import time
import logging
import random
import re
import difflib
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    pipeline,
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Optional libraries 
try:
    from lime.lime_text import LimeTextExplainer
except Exception:
    LimeTextExplainer = None

# spellchecker
try:
    from spellchecker import SpellChecker
    SPELLCHECKER = SpellChecker()
except Exception:
    SPELLCHECKER = None

# sentence-transformers for embedding-based matching
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    # Increased model size for better embedding quality
    EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    EMBED_MODEL = None

# NEW: SpaCy NER Library Setup
try:
    import spacy
    # Attempt to load the model (assuming user ran the install steps above)
    NER_NLP = spacy.load("en_core_web_sm")
    logger.info("spaCy NER model loaded successfully.")
except Exception:
    NER_NLP = None
    logger.warning("spaCy NER model failed to load. Falling back to basic regex for PII masking. Please ensure 'spacy' is installed and 'en_core_web_sm' is downloaded.")


# Set a fixed random seed for reproducibility
SEED = 50
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# EXTENDED CONFIGURATION

DATASET_NAME = "zefang-liu/phishing-email-dataset"
SAMPLES_TO_USE = 15000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# MODEL_NAME = "bert-base-uncased"
MODEL_NAME = "distilbert-base-uncased"


# HF_LLM_MODEL = "google/flan-t5-base"
HF_LLM_MODEL = "google/flan-t5-small"

OUTPUT_DIR = "./results_phishing_detector_extended"
MAX_LENGTH = 256        
LEARNING_RATE = 3e-5   
BATCH_SIZE = 2         
NUM_TRAIN_EPOCHS = 10
FGM_EPSILON = 1e-6

# IMPROVEMENT: Configurable adversarial loss weight (lambda)

ADV_LOSS_WEIGHT = 0.5

# Only local HF provider is supported now
LLM_PROVIDERS = ["hf"]

# LIME stability
LIME_NUM_SAMPLES = 5000 # increase for more stability 

# Embedding similarity threshold
EMBEDDING_SIMILARITY_CUTOFF = 0.65

# Whitelist (common safe phrases)
WHITELIST_PHRASES = [
    "let me know a good time",
    "looking forward to meeting",
    "best regards",
    "kind regards",
    "sincerely",
    "please find attached",
    "attached please find",
]

# END EXTENDED CONFIGURATION


# LLM Templates (JSON embedded) - unchanged
LLM_TEMPLATES_JSON = r'''
[
    {
        "id": "legitimate_routine",
        "keywords": ["agenda", "report", "meeting", "attached", "progress", "quarterly", "final"],
        "template": "The message appears routine and contains no obvious social-engineering cues or suspicious tokens.",
        "requires": []
    },
    {
        "id": "urgency_clickbait",
        "keywords": ["urgent", "immediately", "verify", "account", "login", "password", "click here", "link"],
        "template": "The email uses a high level of urgency and includes clickbait keywords ({tokens}) suggesting a fraudulent attempt to capture credentials or financial information.",
        "requires": [
            ["urgent", "immediately"],
            ["click", "link", "verify", "login"]
        ]
    },
    {
        "id": "financial_risk",
        "keywords": ["bank", "invoice", "payment", "billing", "wire transfer", "transfer", "financial"],
        "template": "The message mentions sensitive financial terms ({tokens}) and attempts to create a sense of financial risk or obligation.",
        "requires": [
            ["bank", "payment", "invoice"]
        ]
    },
    {
        "id": "promotion_reward",
        "keywords": ["prize", "reward", "congratulations", "win", "gift", "claim"],
        "template": "The email uses terms like 'prize' or 'reward' ({tokens}) to lure the user, a common tactic in promotional phishing attempts.",
        "requires": [
            ["prize", "reward"]
        ]
    },
    {
        "id": "suspicious_request",
        "keywords": ["send me", "transfer", "document", "confidential"],
        "template": "The message contains suspicious requests for action or confidential information, using tokens like {tokens}.",
        "requires": []
    },
    {
        "id": "default",
        "keywords": [],
        "template": "The message lacks strong indicators but the model's prediction suggests a possible issue. Key tokens: {tokens}.",
        "requires": []
    }
]
'''
try:
    LLM_TEMPLATES = json.loads(LLM_TEMPLATES_JSON)
except Exception:
    LLM_TEMPLATES = []


# FGM Implementation
class FGM(object):
    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and "embeddings" in name:
                self.backup[name] = param.data.clone()
                if param.grad is not None:
                    norm = torch.linalg.norm(param.grad)
                    if norm != 0:
                        r_adv = self.epsilon * param.grad / norm
                        param.data.add_(r_adv)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and "embeddings" in name:
                if name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup = {}

class FGMTrainer(Trainer):
    # adv_loss_weight parameter
    def __init__(self, fgm_epsilon=FGM_EPSILON, adv_loss_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.fgm = FGM(self.model, epsilon=fgm_epsilon)
        self.adv_loss_weight = adv_loss_weight # Store the configurable weight
        logger.info(f"FGM Adversarial Training initialized with epsilon={fgm_epsilon}, lambda={adv_loss_weight}")

    # adv_loss_weight usage
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss_clean = outputs.loss

        if loss_clean.requires_grad:
            loss_clean.backward(retain_graph=True)
        self.fgm.attack()
        outputs_adv = model(**inputs)
        loss_adv = outputs_adv.loss
        self.fgm.restore()

        # Use configurable loss weight (ADV_LOSS_WEIGHT)
        total_loss = loss_clean + self.adv_loss_weight * loss_adv
        if return_outputs:
            return (total_loss, outputs)
        return total_loss


# Sensitive Information Masking Utilities (NER-BASED IMPLEMENTATION)
# Regex helpers for entities spaCy sometimes misses or for which regex is more precise
_EMAIL_REGEX = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
_PHONE_REGEX = re.compile(r'\b(?:\+?\d[\d\-\s\(\)]{6,}\d)\b')

def mask_sensitive_info(text: str) -> str:
    """
    Uses spaCy NER for masking PII entities like Name, Location, etc.
    Falls back to basic regex for common patterns like emails/phones.
    """
    masked_text = text
    if NER_NLP:
        doc = NER_NLP(text)
        # Process entities in reverse order so character offsets remain valid
        for ent in reversed(doc.ents):
            replacement = None
            # Check for common PII types
            if ent.label_ in ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC"]:
                replacement = f"[{ent.label_}]"
            elif ent.label_ in ["DATE", "TIME", "MONEY", "PERCENT", "QUANTITY"]:
                replacement = "[VALUE]"
            elif ent.label_ in ["CARDINAL"]:
                # Heuristic check for potential account numbers/IDs that NER might misclassify
                digits = re.sub(r'\D', '', ent.text)
                if len(digits) >= 6:
                    replacement = "[ACCOUNT]"
            if replacement:
                masked_text = masked_text[:ent.start_char] + replacement + masked_text[ent.end_char:]

    # Regex masking (run after NER to fill potential gaps)
    # Phone numbers -> [PHONE]
    masked_text = _PHONE_REGEX.sub('[PHONE]', masked_text)
    # Email addresses -> [EMAIL]
    masked_text = _EMAIL_REGEX.sub('[EMAIL]', masked_text)
    return masked_text


# Data Loading & Preprocessing 
def load_and_preprocess_data(dataset_name, samples, seed, train_ratio, val_ratio, test_ratio, tokenizer):
    logger.info(f"Loading dataset: {dataset_name}")
    raw_dataset = load_dataset(dataset_name)
    df_raw = raw_dataset['train'].to_pandas()

    # expected column names in this dataset
    df_raw.rename(columns={'Email Text': 'email', 'Email Type': 'phishing'}, inplace=True)
    df_raw['phishing'] = df_raw['phishing'].map({'Safe Email': 0, 'Phishing Email': 1})
    if 'Unnamed: 0' in df_raw.columns:
        df_raw.drop(columns=['Unnamed: 0'], inplace=True)

    if len(df_raw) > samples:
        logger.info("Performing initial stratified sampling using train_test_split...")
        keep_fraction = samples / len(df_raw)
        _, X_keep, _, _ = train_test_split(
            df_raw,
            df_raw['phishing'],
            test_size=keep_fraction,
            random_state=seed,
            stratify=df_raw['phishing']
        )
        df_sampled = X_keep.copy()
    else:
        df_sampled = df_raw.copy()

    logger.info(f"Using {len(df_sampled)} samples.")

    # Apply Sensitive Information Masking using spaCy
    # Masking is applied to the original email text
    df_sampled['masked_email'] = df_sampled['email'].apply(lambda x: mask_sensitive_info(x) if isinstance(x, str) else '')

    # The 'email' column is updated to be the lowercased, masked text for DistilBERT training
    df_sampled['email'] = df_sampled['masked_email'].apply(lambda x: x.lower())
    df_sampled.drop(columns=['masked_email'], inplace=True, errors='ignore')

    if 'phishing' not in df_sampled.columns:
        raise KeyError("Required 'phishing' column is missing after renaming/sampling.")

    df_train_val, df_test = train_test_split(
        df_sampled,
        test_size=test_ratio,
        random_state=seed,
        stratify=df_sampled['phishing']
    )

    val_size_relative = val_ratio / (train_ratio + val_ratio)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=val_size_relative,
        random_state=seed,
        stratify=df_train_val['phishing']
    )

    logger.info(f"Splits: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    def tokenize_function(examples):
        # This uses the lowercased, masked 'email' column
        return tokenizer(examples['email'], truncation=True, padding='max_length', max_length=MAX_LENGTH)

    dataset_dict = {
        'train': Dataset.from_pandas(df_train[['email', 'phishing']].rename(columns={'phishing': 'labels'})),
        'validation': Dataset.from_pandas(df_val[['email', 'phishing']].rename(columns={'phishing': 'labels'})),
        'test': Dataset.from_pandas(df_test[['email', 'phishing']].rename(columns={'phishing': 'labels'}))
    }

    tokenized_datasets = {}
    for key, ds in dataset_dict.items():
        cols_to_remove = [col for col in ds.column_names if col not in ['email', 'labels']]
        ds = ds.remove_columns(cols_to_remove)
        tokenized_datasets[key] = ds.map(
            tokenize_function,
            batched=True,
            remove_columns=['email']
        )

    return tokenized_datasets

# Training
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def train_model(tokenized_datasets):
    logger.info("Initializing Tokenizer and Model...")
    # Using Auto classes
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    except Exception:
        # Fallback to original DistilBert classes if Auto fails 
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")


    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
     
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=SEED,
        # IMPROVEMENT: Prefer bf16 for stability if available (T4 supports it), otherwise fallback to fp16
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not (torch.cuda.is_bf16_supported()),
        report_to="none"
    )

    trainer = FGMTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        fgm_epsilon=FGM_EPSILON,
        # ADDED: Pass the configurable loss weight
        adv_loss_weight=ADV_LOSS_WEIGHT
    )

    logger.info("Starting FGM Adversarial Training...")
    trainer.train()
    logger.info("Training complete. Saving final model weights locally.")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    test_results = trainer.evaluate(tokenized_datasets['test'])
    logger.info(f"Test Set Evaluation Results: {test_results}")
    return model, tokenizer

# Explainability

class Predictor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def predict_proba(self, texts: List[str]):
        encoded_inputs = self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        return probs

def format_user_friendly_explanation(top_tokens_with_scores: List[Tuple[str, float]], prediction_label: str, confidence: float) -> str:
    """Create a compact human-friendly explanation using simple heuristics over tokens."""
    tokens = [t for t, s in top_tokens_with_scores]
    scores = {t: s for t, s in top_tokens_with_scores}

    cues = {
        'urgency': ['urgent', 'immediately', 'asap', 'now', 'verify', 'hurry', 'limited'],
        'clickbait': ['click', 'link', 'here', 'bit.ly', 'tinyurl'],
        'financial_lure': ['bank', 'account', 'password', 'verify', 'billing', 'payment', 'invoice'],
        'login_credential': ['login', 'password', 'username', 'signin'],
        'unknown_url': ['http://', 'https://']
    }

    detected = []
    for label, keywords in cues.items():
        for kw in keywords:
            for tok in tokens:
                if kw in tok:
                    detected.append(label)
                    break
            if label in detected:
                break

    reasons = []
    if 'urgency' in detected:
        reasons.append('uses urgency or time pressure')
    if 'clickbait' in detected:
        reasons.append('encourages clicking a link')
    if 'financial_lure' in detected:
        reasons.append('mentions financial/account-related terms')
    if 'login_credential' in detected:
        reasons.append("asks for login/credential-related info")
    if 'unknown_url' in detected:
        reasons.append('contains a URL or suspicious link')

    if not reasons:
        # fallback: mention top tokens
        top_shown = ', '.join(tokens[:5]) if tokens else 'no suspicious tokens'
        explanation = f"The email was classified as {prediction_label} with confidence {confidence:.2f}. Key tokens: {top_shown}."
    else:
        explanation = f"The email was classified as {prediction_label} with confidence {confidence:.2f} because it {', and '.join(reasons)}."

    return explanation

# normalization, spell-correction, fuzzy matching, embedding logic 

def has_whitelist_phrase(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    for ph in WHITELIST_PHRASES:
        if ph in t:
            return True
    return False

def spell_correct_token(tok: str) -> str:
    """Attempt to correct a single token using pyspellchecker if available; otherwise return original."""
    if not tok or SPELLCHECKER is None:
        return tok
    # SpellChecker expects words without punctuation
    try:
        corrected = SPELLCHECKER.correction(tok)
        return corrected if corrected is not None else tok
    except Exception:
        return tok

def normalize_and_correct_tokens(tokens: List[str], keyword_pool: List[str] = None, max_matches=1) -> List[str]:
    """
    Lowercase, strip, remove weird chars (similar to your LIME cleaning),
    attempt to spell-correct tokens by spellchecker (if available) then
    fuzzy-match against keyword_pool.
    """
    cleaned = []
    for t in tokens:
        if not t:
            continue
        # 1. Light cleaning (similar to LIME)
        tok_cleaned = re.sub(r'[^a-z0-9\-:/_.]', '', t.lower()).strip()
        if not tok_cleaned:
            continue

        # 2. Spell correction
        tok_corrected = spell_correct_token(tok_cleaned)

        # 3. Fuzzy matching against keyword pool if provided
        if keyword_pool and keyword_pool:
            # Check for exact match first
            if tok_corrected in keyword_pool:
                cleaned.append(tok_corrected)
                continue
            # Then fuzzy match
            matches = difflib.get_close_matches(tok_corrected, keyword_pool, n=max_matches, cutoff=0.9)
            if matches:
                cleaned.append(matches[0])
                continue

        # If no fuzzy match or no keyword pool, use the spell-corrected token
        cleaned.append(tok_corrected)

    # remove duplicates preserving order
    seen = set()
    out = []
    for c in cleaned:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

# build keyword embeddings cache
_keyword_embedding_cache = None
_keyword_list_cache = None

def build_keyword_embeddings(keyword_pool: List[str]):
    global _keyword_embedding_cache, _keyword_list_cache
    if EMBED_MODEL is None:
        return None
    if _keyword_list_cache == keyword_pool and _keyword_embedding_cache is not None:
        return _keyword_embedding_cache
    try:
        emb = EMBED_MODEL.encode(keyword_pool, convert_to_numpy=True)
        _keyword_embedding_cache = emb
        _keyword_list_cache = list(keyword_pool)
        return emb
    except Exception:
        return None

def choose_template(top_tokens: List[str], prediction_label: str, confidence: float, original_text: Optional[str] = None) -> dict:
    """
    Choose the most relevant template from the in-file JSON templates based on:
    - Model prediction (LEGITIMATE vs PHISHING)
    - Keyword matching scores
    """
    if not LLM_TEMPLATES:
        return None

    # build keyword pool
    keyword_pool = []
    template_kw_map = {}
    for t in LLM_TEMPLATES:
        kws = [k.lower() for k in t.get('keywords', [])]
        template_kw_map[t.get('id')] = kws
        for kw in kws:
            keyword_pool.append(kw)
    keyword_pool = sorted(set(keyword_pool))

    # normalize + correct tokens
    tokens_norm = normalize_and_correct_tokens(top_tokens, keyword_pool=keyword_pool)

    # optional embedding setup
    keyword_embeddings = None
    if EMBED_MODEL is not None and keyword_pool:
        keyword_embeddings = build_keyword_embeddings(keyword_pool)

    whitelist_present = has_whitelist_phrase(original_text) if original_text else False
    is_legitimate = (prediction_label == 'LEGITIMATE')

    # Determine which templates are candidates for scoring
    candidate_templates = []
    if is_legitimate:
        # For LEGITIMATE prediction, only consider the safe/routine template and default.
        # This prevents low-scoring benign tokens from triggering phishing templates.
        for tmpl in LLM_TEMPLATES:
            if tmpl.get('id') in ['legitimate_routine', 'default']:
                candidate_templates.append(tmpl)
    else:
        # For PHISHING prediction, consider all phishing templates and the default.
        for tmpl in LLM_TEMPLATES:
            if tmpl.get('id') not in ['legitimate_routine']:
                candidate_templates.append(tmpl)

    best = None
    best_score = -1.0
    for tmpl in candidate_templates: # Iterate only over candidates
        tmpl_id = tmpl.get('id')
        kws = template_kw_map.get(tmpl_id, [])
        score = 0.0

        # 1. Simple keyword overlap score
        for kw in kws:
            if kw in tokens_norm:
                score += 1.0

        # 2. Fuzzy/Partial match scoring
        for tok in tokens_norm:
            # Fuzzy match only if no exact match was found above
            if tok not in kws and keyword_pool:
                fuzzy = difflib.get_close_matches(tok, kws, n=1, cutoff=0.8)
                if fuzzy:
                    score += 1 # Give 1 point for a strong fuzzy match

        # 3. Embedding-based scoring (token vs template keyword embeddings)
        if keyword_embeddings is not None and EMBED_MODEL is not None and kws:
            try:
                # embed tokens
                tok_embs = EMBED_MODEL.encode(tokens_norm, convert_to_numpy=True)
                # compute similarity with template keywords
                kw_indices = [keyword_pool.index(k) for k in kws if k in keyword_pool]
                if kw_indices and len(tok_embs) > 0:
                    sim = cosine_similarity(tok_embs, keyword_embeddings[kw_indices])
                    # take max similarity per token
                    max_sim = sim.max() if sim.size > 0 else 0.0
                    if max_sim >= EMBEDDING_SIMILARITY_CUTOFF:
                        score += (1.5 * float(max_sim))
            except Exception:
                pass

        # 4. Enforce template-specific 'requires' constraints if present
        reqs = tmpl.get('requires', []) or []
        if reqs:
            group_ok = True
            for group in reqs:
                found = False
                for gkw in group:
                    for tok in tokens_norm:
                        if gkw in tok or tok in gkw:
                            found = True
                            break
                    if not found:
                        group_ok = False
                        break
            if not group_ok:
                score -= 3 # Penalize for unmet requirements

        # 5. Penalize legitimate routine template if whitelist is present
        if tmpl_id == 'legitimate_routine' and whitelist_present:
            score -= 1.0

        # Update best template
        # small tie-breaker: prefer templates with keywords when tied
        if score > best_score or (score == best_score and kws and (best is not None and not best.get('keywords'))):
            best = tmpl
            best_score = score

    # If no template scored well, return the most appropriate default/routine template.
    if best is None or best_score <= 0:
        if is_legitimate:
            # For legitimate, always default to the routine template if no candidate scored well.
            for tmpl in LLM_TEMPLATES:
                if tmpl.get('id') == 'legitimate_routine':
                    return tmpl # Guaranteed to return 'legitimate_routine'
        else:
            # For phishing/uncertain, default to the generic template.
            for tmpl in LLM_TEMPLATES:
                if tmpl.get('id') == 'default':
                    return tmpl # Guaranteed to return 'default'
        return LLM_TEMPLATES[0] # Failsafe

    return best


def build_prompt_from_template(template_obj: dict, top_tokens: List[str], prediction_label: str, confidence: float, top_scores: List[float] = None) -> Tuple[str, str]:
    """
    Returns (prompt_for_llm, template_fallback_text).
    - prompt_for_llm: used when calling HF pipeline (still based on template)
    - template_fallback_text: fully formatted explanation you can return immediately if HF pipeline fails
    """
    tokens_display = ", ".join(normalize_and_correct_tokens(top_tokens)[:8])

    # Prepend prediction and confidence to ensure it's always included in the final output
    prediction_prefix = f"Prediction: The email was classified as **{prediction_label}** with confidence **{confidence:.2f}**. "

    fmt = {'tokens': tokens_display, 'confidence': f"{confidence:.2f}", 'prediction': prediction_label}

    # Default fallback template text in case of parsing error
    fallback = f"{prediction_prefix} Explanation (LLM-fallback): The most important tokens were: {tokens_display}."

    try:
        template_text = template_obj.get('template', '')
        formatted_template_text = template_text.format(**fmt)
        fallback = f"{prediction_prefix} {formatted_template_text}"
    except Exception:
        fallback = f"{prediction_prefix} Explanation (LLM-fallback): The most important tokens were: {tokens_display}."

    # Calibrated prompt: ask LLM not to repeat the template verbatim, to validate signals, and to avoid hallucination
    if template_obj.get('id') == 'legitimate_routine':
        # Give the LLM a softer instruction for routine emails to avoid misinterpretation
        prompt = (
            formatted_template_text + "\n\nPlease provide a concise (1-2 sentence) explanation confirming the email is safe/routine. "
            + "Start your response directly with the natural language explanation. "
            + "Do NOT repeat the template verbatim or include the prediction/confidence in your final answer. "
            + "Only claim it is safe/routine if it truly contains no suspicious cues."
        )
    else:
        # Original (more aggressive) prompt for potential phishing
        prompt = (
            formatted_template_text + "\n\nPlease provide a concise (1-2 sentence) explanation referencing the tokens and common phishing cues like 'urgency', 'clickbait', 'financial lure'. "
            + "Start your response directly with the natural language explanation. "
            + "Do NOT repeat the template verbatim or include the prediction/confidence in your final answer. "
            + "Only claim 'encourages clicking' if the email contains an explicit link or the token 'click' or a shortened URL. "
            + "If unsure, be conservative and state uncertainty (e.g., 'possible phishing cues')."
        )

    return prompt, fallback

def generate_prompt_for_llm(top_tokens: List[str], prediction_label: str, confidence: float, top_scores: List[float]) -> str:
    tokens_str = ", ".join([f"'{t}'" for t in top_tokens])
    prompt = (
        f"You are an AI security analyst. The model predicted {prediction_label} with confidence {confidence:.2f}.\n"
        f"The most important tokens are: {tokens_str}.\n"
        "Provide a concise (1-2 sentence) explanation that justifies the classification by referencing the tokens and using terms like 'urgency', 'clickbait', or 'financial lure'.\n"
        "Start your response directly with the natural language explanation.\n"
        "Do NOT repeat the template verbatim or include the prediction/confidence in your final answer.\n"
        "Only state 'encourages clicking' if there is an explicit link token or the word 'click' appears.\n"
        "If uncertain, say 'possible' or 'may'. \n"
        "Do not invent facts."
    )
    return prompt

def get_llm_explanation(top_tokens: List[str], prediction_label: str, confidence: float, provider: str, hf_pipeline=None, top_scores: List[float] = None, original_text: Optional[str] = None) -> str:
    """
    Attempts to use local HF pipeline to improve the templated explanation.
    Always returns a usable explanation string (template fallback used when HF pipeline fails).
    The final output is formatted to always include the prediction and confidence at the start.
    """
    template_obj = choose_template(top_tokens, prediction_label, confidence, original_text=original_text)
    prompt, template_fallback = build_prompt_from_template(template_obj, top_tokens, prediction_label, confidence, top_scores)

    # This prefix is guaranteed to be in the final output
    prediction_prefix = f"Prediction: The email was classified as **{prediction_label}** with confidence **{confidence:.2f}**. "

    # Only HF provider supported here
    if provider == 'hf' and hf_pipeline is not None:
        try:
            # Using the template-based prompt
            out = hf_pipeline(prompt, max_new_tokens=60, do_sample=False, early_stopping=True)

            text = None
            if isinstance(out, list) and len(out) > 0:
                elem = out[0]
                if isinstance(elem, dict):
                    text = elem.get('generated_text') or elem.get('summary_text') or elem.get('text')
                else:
                    text = str(elem)
            elif isinstance(out, dict):
                text = out.get('generated_text') or out.get('text')
            else:
                text = str(out)

            if text and text.strip():
                # final cleanup: single-line, conservative phrasing
                cleaned_llm_output = text.strip().replace('\n', ' ').strip()

                # Ensure the LLM didn't accidentally include the prediction/confidence
                if cleaned_llm_output.lower().startswith('prediction:'):
                    # If it did, try to strip it, otherwise use fallback
                    llm_expl = re.sub(r'(?i)prediction:\s*The email was classified as.*?\.', '', cleaned_llm_output).strip()
                else:
                    llm_expl = cleaned_llm_output

                if llm_expl:
                    # Prepend the guaranteed prediction prefix and return the LLM's explanation
                    return prediction_prefix + llm_expl
                else:
                    logger.info("HF pipeline returned an unhelpful response after cleanup — using template fallback.")
                    return template_fallback
            else:
                logger.info("HF pipeline returned empty text — using template fallback.")
                return template_fallback
        except Exception:
            logger.exception("HF pipeline error (using template fallback):")
            return template_fallback

    # If no HF pipeline or provider mismatch, return template fallback
    logger.info("HF pipeline not available or provider not 'hf' — returning template fallback.")
    return template_fallback


# Interactive Demo
def run_interactive_demo(model, tokenizer, hf_llm_pipeline=None):
    if LimeTextExplainer is None:
        logger.error("LIME is not installed. Install lime to get token-level explanations.")
        return

    logger.info("Starting Interactive Demo with HF local LLM explanation support")
    predictor = Predictor(model, tokenizer)
    class_names = ['LEGITIMATE', 'PHISHING']
    # Pass random_state=SEED for reproducible LIME explanations
    explainer = LimeTextExplainer(class_names=class_names, random_state=SEED)

    # Only HF provider is supported, so we won't ask for external API keys
    print('\nProvider options: only local HF pipeline (hf).')
    print('If you do not have the HF model downloaded, HF pipeline may fail gracefully and the code will return template fallback text.')
    print('Provide a comma-separated preference list (e.g. hf) or press Enter for default:')
    providers_input = input('Providers: ').strip()
    providers = LLM_PROVIDERS
    if providers_input:
        chosen = [p.strip().lower() for p in providers_input.split(',') if p.strip()]
        if chosen:
            providers = chosen

    print("\nEnter 'quit' or 'exit' to leave the demo.")
    while True:
        email_body = input('\n>>> Enter Email Body:\n')
        if email_body is None:
            break
        email_body = email_body.strip()
        if email_body.lower() in ['quit', 'exit']:
            break
        if not email_body:
            continue

        # NEW: Apply Masking before feeding to the model
        masked_text = mask_sensitive_info(email_body)
        original_display = email_body # Keep original for display
        input_text = masked_text.lower() # Model input is masked and lowercased

        print('\n--- Model Output ---')
        # Print the masked text for the user to confirm the PII masking is working
        print(f"Input Email Body (masked for model):\n{masked_text}\n")

        try:
            probs = predictor.predict_proba([input_text])[0]
            prediction_idx = int(np.argmax(probs))
            confidence = float(probs[prediction_idx])
            prediction_label = class_names[prediction_idx]
            print(f"Prediction: {prediction_label} (confidence {confidence:.2f})")

            # LIME explanation (using the masked, lowercased input)
            explanation = explainer.explain_instance(
                input_text,
                predictor.predict_proba,
                num_features=10,
                labels=(prediction_idx,)
            )
            lime_features = explanation.as_list(label=prediction_idx)

            # Clean tokens and keep scores (light cleaning)
            cleaned = [(re.sub(r'[^a-z0-9\-:/_.]', '', f[0]).strip(), float(f[1])) for f in lime_features]
            cleaned = [(t if t else '<empty>', s) for t, s in cleaned]

            print('\nLIME Top Contributing Tokens (token, score):')
            for t, s in cleaned:
                print(f" {t:30s} {s:+0.4f}")

            # Build user-friendly explanation
            top_tokens_with_scores = cleaned[:8]
            friendly_expl = format_user_friendly_explanation(top_tokens_with_scores, prediction_label, confidence)
            print(f"\nUser-friendly explanation (heuristic): {friendly_expl}")

            # Now try HF local pipeline explanation using templates + the first available provider
            llm_text = None
            used = 'none'
            token_list = [t for t, s in top_tokens_with_scores]
            score_list = [s for t, s in top_tokens_with_scores]

            if providers:
                provider = providers[0]
                llm_text = get_llm_explanation(
                    token_list, prediction_label, confidence, provider, hf_pipeline, score_list, original_text=original_display
                )
                used = provider

            if llm_text:
                print(f"\nLLM Explanation (via {used}): {llm_text}")
            else:
                # Should not happen if template fallback works, but for safety
                print("\nCould not generate LLM explanation. Using heuristic explanation.")

        except Exception as e:
            logger.exception("An error occurred during prediction/explanation.")
            print(f"\nError: Could not process email. {e}")
            print("Generic Warning: Predictions may fail in some situations.")
            print("")


# Main
if __name__ == '__main__':
    logger.info('Prototype enhanced start')

    # 1. Initialize Tokenizer and Load Data
    # Use AutoTokenizer if possible for model flexibility
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    tokenized_datasets = load_and_preprocess_data(
        DATASET_NAME, SAMPLES_TO_USE, SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, tokenizer
    )

    # 2. Train Model
    trained_model, final_tokenizer = train_model(tokenized_datasets)

    # 3. Prepare HF LLM pipeline as fallback (fast local explanatory LLM)
    hf_pipeline = None
    try:
        # Ensure you have 'google/flan-t5-small' downloaded or can access it
        hf_pipeline = pipeline(
            'text2text-generation',
            model=HF_LLM_MODEL,
            tokenizer=HF_LLM_MODEL,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
        )
    except Exception:
        logger.exception("Could not load HF LLM pipeline (this is optional). Template fallback will be used instead.")

    # 4. Run interactive demo
    run_interactive_demo(trained_model, final_tokenizer, hf_pipeline)
