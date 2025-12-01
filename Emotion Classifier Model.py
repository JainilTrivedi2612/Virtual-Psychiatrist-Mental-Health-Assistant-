import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import nlpaug.augmenter.word as naw
import nltk
from collections import Counter

# NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')

# random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data Preparation
df = pd.read_csv("VIRTUALPSYCHTRAIN500-csv.csv")
df = df.dropna(subset=["student_emotion_before", "student_first_dialogue"])
print(f"Original dataset size: {len(df)}")

le = LabelEncoder()
df["emotion_label"] = le.fit_transform(df["student_emotion_before"])
print(f"Emotion classes: {le.classes_}")

# Check class distribution
class_counts = Counter(df["emotion_label"])
print("Class distribution:", {le.inverse_transform([k])[0]: v for k, v in class_counts.items()})

# Data Augmentation with Oversampling for Rare Classes
aug_syn = naw.SynonymAug(aug_src='wordnet')
aug_bt = naw.BackTranslationAug(device='cuda' if torch.cuda.is_available() else 'cpu')
aug_context = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute", device='cuda' if torch.cuda.is_available() else 'cpu')

augmented_texts = []
augmented_labels = []

for text, label in zip(df["student_first_dialogue"], df["emotion_label"]):
    if not isinstance(text, str):
        print(f"Skipping non-string text: {text}")
        continue

    # Append original
    augmented_texts.append(text)
    augmented_labels.append(label)

    # Synonym augmentation (one instance)
    try:
        synonym_result = aug_syn.augment(text)
        aug_text = synonym_result[0] if synonym_result and isinstance(synonym_result, (list, tuple)) and len(synonym_result) > 0 else text
        augmented_texts.append(aug_text)
    except Exception as e:
        print(f"Synonym augmentation failed for: {text[:50]}... Error: {e}")
        augmented_texts.append(text)
    augmented_labels.append(label)

    # Back-translation (one instance)
    print(f"Processing back-translation for: {text[:50]}...")
    try:
        bt_result = aug_bt.augment(text)
        aug_text = bt_result[0] if bt_result and isinstance(bt_result, (list, tuple)) and len(bt_result) > 0 else text
        augmented_texts.append(aug_text)
        print("Back-translation completed.")
    except Exception as e:
        print(f"Back-translation failed for: {text[:50]}... Error: {e}")
        augmented_texts.append(text)
    augmented_labels.append(label)

    # Oversample rare classes (support < 10)
    if class_counts[label] < 10:
        # Add two additional synonym-augmented samples
        for _ in range(2):
            try:
                synonym_result = aug_syn.augment(text)
                aug_text = synonym_result[0] if synonym_result and isinstance(synonym_result, (list, tuple)) and len(synonym_result) > 0 else text
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
                print(f"Added synonym augmentation for rare class: {le.inverse_transform([label])[0]}")
            except Exception as e:
                print(f"Synonym augmentation failed for rare class: {text[:50]}... Error: {e}")
                augmented_texts.append(text)
                augmented_labels.append(label)

        # Add two additional back-translated samples
        for _ in range(2):
            try:
                bt_result = aug_bt.augment(text)
                aug_text = bt_result[0] if bt_result and isinstance(bt_result, (list, tuple)) and len(bt_result) > 0 else text
                augmented_texts.append(aug_text)
                augmented_labels.append(label)
                print(f"Added back-translation for rare class: {le.inverse_transform([label])[0]}")
            except Exception as e:
                print(f"Back-translation failed for rare class: {text[:50]}... Error: {e}")
                augmented_texts.append(text)
                augmented_labels.append(label)

        # Add one contextual augmentation
        try:
            context_result = aug_context.augment(text)
            aug_text = context_result[0] if context_result and isinstance(context_result, (list, tuple)) and len(context_result) > 0 else text
            augmented_texts.append(aug_text)
            augmented_labels.append(label)
            print(f"Added contextual augmentation for rare class: {le.inverse_transform([label])[0]}")
        except Exception as e:
            print(f"Contextual augmentation failed for: {text[:50]}... Error: {e}")
            augmented_texts.append(text)
            augmented_labels.append(label)

# Verifying length of augmented dataset
print(f"Length of augmented_texts: {len(augmented_texts)}")
print(f"Length of augmented_labels: {len(augmented_labels)}")

augmented_df = pd.DataFrame({
    "student_first_dialogue": augmented_texts,
    "emotion_label": augmented_labels
})
print(f"Augmented dataset size: {len(augmented_df)}")

# Split into train+val (90%) and test (10%)
train_val_df, test_df = train_test_split(augmented_df, test_size=0.1, stratify=augmented_df["emotion_label"], random_state=42)

# Tokenization of input data
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_texts(texts, max_length=128):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

# 4. Emotion Classification Model Definition 
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(EmotionClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# Training and Evaluation of the classifier model
def train_and_evaluate(model, train_loader, val_loader, epochs=10, lr=2e-5):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    class_weights = compute_class_weight('balanced', classes=np.unique(train_val_df["emotion_label"]), y=train_val_df["emotion_label"])
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))

    best_val_loss = float("inf")
    patience, early_stop_count = 3, 0  

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, val_preds, val_true = 0, [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [x.to(device) for x in batch]
                logits = model(input_ids, attention_mask)
                val_loss += loss_fn(logits, labels).item()
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(val_true, val_preds)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print("Early stopping triggered.")
                break

    return val_true, val_preds

# K-Fold Cross-Validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
all_true, all_preds = [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_df)):
    print(f"\nFold {fold+1}/{k_folds}")
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]

    train_texts = train_df["student_first_dialogue"].tolist()
    train_labels = train_df["emotion_label"].tolist()
    val_texts = val_df["student_first_dialogue"].tolist()
    val_labels = val_df["emotion_label"].tolist()

    train_encodings = tokenize_texts(train_texts)
    val_encodings = tokenize_texts(val_texts)

    train_dataset = TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"], torch.tensor(train_labels))
    val_dataset = TensorDataset(val_encodings["input_ids"], val_encodings["attention_mask"], torch.tensor(val_labels))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = EmotionClassifier(num_classes=len(le.classes_))
    model.to(device)

    val_true, val_preds = train_and_evaluate(model, train_loader, val_loader, lr=2e-5)
    all_true.extend(val_true)
    all_preds.extend(val_preds)

# Final Model Evaluation Across the different K Folds 
unique_labels_cv = np.unique(all_true + all_preds)
target_names_cv = [le.classes_[i] for i in unique_labels_cv]
print("\nCross-Validation Results:")
print(f"Accuracy: {accuracy_score(all_true, all_preds):.4f}")
macro_f1_cv = f1_score(all_true, all_preds, average='macro', labels=unique_labels_cv)
print(f"Macro F1-Score: {macro_f1_cv:.4f}")
print("Confusion Matrix:")
cm_cv = confusion_matrix(all_true, all_preds, labels=unique_labels_cv)
print(cm_cv)
print("Classification Report:")
print(classification_report(all_true, all_preds, labels=unique_labels_cv, target_names=target_names_cv))

# Train Final Model and Test 
full_encodings = tokenize_texts(train_val_df["student_first_dialogue"].tolist())
test_encodings = tokenize_texts(test_df["student_first_dialogue"].tolist())
full_dataset = TensorDataset(full_encodings["input_ids"], full_encodings["attention_mask"], torch.tensor(train_val_df["emotion_label"].tolist()))
test_dataset = TensorDataset(test_encodings["input_ids"], test_encodings["attention_mask"], torch.tensor(test_df["emotion_label"].tolist()))
full_loader = DataLoader(full_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

model = EmotionClassifier(num_classes=len(le.classes_))
model.to(device)
test_true, test_preds = train_and_evaluate(model, full_loader, test_loader, lr=2e-5)
torch.save(model.state_dict(), "emotion_classifier_final.pth")
print("Final model saved to 'emotion_classifier_final.pth'")

# Test Set Evaluation
unique_labels_test = np.unique(test_true + test_preds)
target_names_test = [le.classes_[i] for i in unique_labels_test]
print("\nTest Set Results:")
print(f"Accuracy: {accuracy_score(test_true, test_preds):.4f}")
macro_f1_test = f1_score(test_true, test_preds, average='macro', labels=unique_labels_test)
print(f"Macro F1-Score: {macro_f1_test:.4f}")
print("Confusion Matrix:")
cm_test = confusion_matrix(test_true, test_preds, labels=unique_labels_test)
print(cm_test)
print("Classification Report:")
print(classification_report(test_true, test_preds, labels=unique_labels_test, target_names=target_names_test))
