"""
Proyecto: Clasificación Automática de Cyberbullying en Tweets
=============================================================
Pipeline completo: EDA → Preprocesamiento NLP → Vectorización → Modelos → Evaluación
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import re
import string
import warnings
import os
warnings.filterwarnings('ignore')

from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# COLORES Y ESTILOS
# ─────────────────────────────────────────────────────────────
COLORS = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2','#937860']
CLASS_COLORS = {
    'not_cyberbullying': '#55A868',
    'gender':            '#4C72B0',
    'religion':          '#DD8452',
    'other_cyberbullying':'#C44E52',
    'age':               '#8172B2',
    'ethnicity':         '#937860',
}
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13})

# ─────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────
print("="*60)
print("1. CARGA Y EXPLORACIÓN DE DATOS")
print("="*60)

df = pd.read_csv(os.path.join(BASE_DIR, 'cyberbullying_tweets.csv'))
print(f"Shape: {df.shape}")
print(df['cyberbullying_type'].value_counts())

# ─────────────────────────────────────────────────────────────
# 2. ANÁLISIS EXPLORATORIO (EDA)
# ─────────────────────────────────────────────────────────────
print("\n2. ANÁLISIS EXPLORATORIO")

# 2.1 Distribución de clases
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
counts = df['cyberbullying_type'].value_counts()
bars = axes[0].bar(counts.index, counts.values,
                   color=[CLASS_COLORS.get(c, '#999') for c in counts.index])
axes[0].set_title('Distribución de Clases')
axes[0].set_xlabel('Tipo de Cyberbullying')
axes[0].set_ylabel('Número de Tweets')
axes[0].tick_params(axis='x', rotation=30)
for bar, val in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 str(val), ha='center', va='bottom', fontsize=9)

wedges, texts, autotexts = axes[1].pie(
    counts.values, labels=counts.index,
    colors=[CLASS_COLORS.get(c, '#999') for c in counts.index],
    autopct='%1.1f%%', startangle=90, pctdistance=0.82)
axes[1].set_title('Proporción de Clases (%)')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '01_distribucion_clases.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → Figura 1: distribución de clases guardada")

# 2.2 Longitud de tweets
df['tweet_len'] = df['tweet_text'].str.len()
df['word_count'] = df['tweet_text'].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for cls, color in CLASS_COLORS.items():
    subset = df[df['cyberbullying_type'] == cls]['tweet_len']
    axes[0].hist(subset, bins=40, alpha=0.5, label=cls, color=color, density=True)
axes[0].set_title('Distribución de Longitud de Tweet (chars)')
axes[0].set_xlabel('Caracteres')
axes[0].set_ylabel('Densidad')
axes[0].legend(fontsize=8)

df.boxplot(column='word_count', by='cyberbullying_type', ax=axes[1],
           patch_artist=True, medianprops=dict(color='black', linewidth=2))
axes[1].set_title('Palabras por Tweet según Clase')
axes[1].set_xlabel('Tipo')
axes[1].set_ylabel('Número de palabras')
plt.suptitle('')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '02_longitud_tweets.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → Figura 2: longitud de tweets guardada")

print(f"\n  Estadísticas de longitud (chars):")
print(df.groupby('cyberbullying_type')['tweet_len'].describe().round(1))

# ─────────────────────────────────────────────────────────────
# 3. PRE-PROCESAMIENTO NLP
# ─────────────────────────────────────────────────────────────
print("\n3. PRE-PROCESAMIENTO NLP")

# Stop-words en inglés (lista compacta embebida, sin NLTK)
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further get going got had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself
him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other ought our
ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some
such than that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up very was wasn't
we we'd we'll we're we've were weren't what what's when when's where where's which while
who who's whom why why's will with won't would wouldn't you you'd you'll you're you've
your yours yourself yourselves just get got said also like even still one two rt
""".split())

# Diccionario de lematización básica (sufijos comunes en inglés)
SUFFIX_MAP = [
    (r'ies$',   'y'),    # bullies → bully
    (r'ied$',   'y'),    # bullied → bully
    (r'ing$',   ''),     # bullying → bully
    (r'ness$',  ''),     # sadness → sad
    (r'tion$',  'te'),   # discrimination → discriminate
    (r'er$',    ''),     # hater → hat  (aproximado)
    (r'ly$',    ''),     # hatefully → hateful
    (r'ed$',    ''),     # hated → hat  (aproximado)
    (r'es$',    ''),     # hates → hat  (aproximado)
    (r's$',     ''),     # tweets → tweet
]

def simple_lemmatize(word):
    """Lematización basada en reglas de sufijos."""
    if len(word) <= 4:
        return word
    for pattern, repl in SUFFIX_MAP:
        if re.search(pattern, word):
            stem = re.sub(pattern, repl, word)
            if len(stem) >= 3:
                return stem
    return word

def preprocess_tweet(text):
    """Pipeline NLP: limpieza → normalización → tokenización → lematización."""
    # Limpieza
    text = re.sub(r'http\S+|www\S+', '', text)         # URLs
    text = re.sub(r'@\w+', '', text)                    # menciones
    text = re.sub(r'#(\w+)', r'\1', text)               # hashtags → solo palabra
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)         # no-ASCII / emojis
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)  # puntuación
    text = re.sub(r'\d+', '', text)                     # dígitos
    # Normalización
    text = text.lower()
    # Tokenización + stop-words + lematización
    tokens = [simple_lemmatize(w) for w in text.split()
              if w not in STOPWORDS and len(w) > 2]
    return ' '.join(tokens)

print("  Aplicando preprocesamiento...")
df['tweet_clean'] = df['tweet_text'].apply(preprocess_tweet)

# Mostrar ejemplos
sample_idx = [0, 1000, 5000]
print("\n  Ejemplos de preprocesamiento:")
for i in sample_idx:
    print(f"  [{df['cyberbullying_type'].iloc[i]}]")
    print(f"    ORIGINAL: {df['tweet_text'].iloc[i][:100]}")
    print(f"    LIMPIO  : {df['tweet_clean'].iloc[i][:100]}")

# Figura: Palabras más frecuentes por clase
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
classes = df['cyberbullying_type'].unique()
for ax, cls in zip(axes, classes):
    tokens = ' '.join(df[df['cyberbullying_type'] == cls]['tweet_clean']).split()
    freq = Counter(tokens).most_common(15)
    words, counts_w = zip(*freq)
    color = CLASS_COLORS.get(cls, '#888')
    ax.barh(list(reversed(words)), list(reversed(counts_w)), color=color, alpha=0.85)
    ax.set_title(f'Top 15 palabras: {cls}')
    ax.set_xlabel('Frecuencia')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '03_top_palabras.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → Figura 3: top palabras por clase guardada")

# ─────────────────────────────────────────────────────────────
# 4. PARTICIÓN TRAIN/TEST
# ─────────────────────────────────────────────────────────────
print("\n4. PARTICIÓN TRAIN / TEST (80/20 estratificado)")

X = df['tweet_clean']
y = df['cyberbullying_type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
print("  Distribución en Train:")
print(y_train.value_counts())

# ─────────────────────────────────────────────────────────────
# 5. VECTORIZACIÓN
# ─────────────────────────────────────────────────────────────
print("\n5. VECTORIZACIÓN")

# BoW
bow_vect = CountVectorizer(max_features=30000, ngram_range=(1,2), min_df=3)
X_train_bow = bow_vect.fit_transform(X_train)
X_test_bow  = bow_vect.transform(X_test)
print(f"  BoW  shape train: {X_train_bow.shape}")

# TF-IDF
tfidf_vect = TfidfVectorizer(max_features=30000, ngram_range=(1,2),
                              min_df=3, sublinear_tf=True)
X_train_tfidf = tfidf_vect.fit_transform(X_train)
X_test_tfidf  = tfidf_vect.transform(X_test)
print(f"  TFIDF shape train: {X_train_tfidf.shape}")

# ─────────────────────────────────────────────────────────────
# 6. ENTRENAMIENTO DE MODELOS
# ─────────────────────────────────────────────────────────────
print("\n6. ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
print("-"*60)

results = {}

def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)
    acc_tr  = accuracy_score(y_tr, y_pred_tr)
    acc_te  = accuracy_score(y_te, y_pred_te)
    f1_tr   = f1_score(y_tr, y_pred_tr, average='macro')
    f1_te   = f1_score(y_te, y_pred_te, average='macro')
    prec_te = precision_score(y_te, y_pred_te, average='macro')
    rec_te  = recall_score(y_te, y_pred_te, average='macro')
    print(f"  [{name}]  Acc_train={acc_tr:.4f}  Acc_test={acc_te:.4f}  F1_test={f1_te:.4f}")
    results[name] = {
        'model': model,
        'acc_train': acc_tr, 'acc_test': acc_te,
        'f1_train': f1_tr, 'f1_test': f1_te,
        'precision': prec_te, 'recall': rec_te,
        'y_pred': y_pred_te,
        'report': classification_report(y_te, y_pred_te)
    }
    return model

# ── 6.1 Naive Bayes (BoW) ──
nb = evaluate("Naive Bayes (BoW)", MultinomialNB(alpha=0.3),
              X_train_bow, X_test_bow, y_train, y_test)

# ── 6.2 Naive Bayes (TF-IDF) ──
nb_tfidf = evaluate("Naive Bayes (TF-IDF)", MultinomialNB(alpha=0.3),
                    X_train_tfidf, X_test_tfidf, y_train, y_test)

# ── 6.3 Regresión Logística ──
lr = evaluate("Regresión Logística (TF-IDF)",
              LogisticRegression(C=5, max_iter=1000, solver='lbfgs', random_state=42),
              X_train_tfidf, X_test_tfidf, y_train, y_test)

# ── 6.4 SVM lineal ──
svm = evaluate("SVM Lineal (TF-IDF)",
               LinearSVC(C=1.0, max_iter=3000, random_state=42),
               X_train_tfidf, X_test_tfidf, y_train, y_test)

# ── 6.5 Random Forest ──
rf = evaluate("Random Forest (TF-IDF)",
              RandomForestClassifier(n_estimators=200, max_depth=None,
                                     min_samples_leaf=2, n_jobs=-1, random_state=42),
              X_train_tfidf, X_test_tfidf, y_train, y_test)

# ── 6.6 Gradient Boosting (subsample para velocidad) ──
gb = evaluate("Gradient Boosting (TF-IDF)",
              GradientBoostingClassifier(n_estimators=150, learning_rate=0.15,
                                         max_depth=5, subsample=0.8,
                                         random_state=42),
              X_train_tfidf, X_test_tfidf, y_train, y_test)

# ─────────────────────────────────────────────────────────────
# 7. TABLA COMPARATIVA
# ─────────────────────────────────────────────────────────────
print("\n7. TABLA COMPARATIVA DE MODELOS")
summary = pd.DataFrame([
    {
        'Modelo': k,
        'Acc Train': f"{v['acc_train']:.4f}",
        'Acc Test':  f"{v['acc_test']:.4f}",
        'F1 Train':  f"{v['f1_train']:.4f}",
        'F1 Test':   f"{v['f1_test']:.4f}",
        'Precision': f"{v['precision']:.4f}",
        'Recall':    f"{v['recall']:.4f}",
    }
    for k, v in results.items()
])
print(summary.to_string(index=False))
summary.to_csv(os.path.join(FIGURES_DIR, 'tabla_comparativa.csv'), index=False)

# ─────────────────────────────────────────────────────────────
# 8. VISUALIZACIONES DE RESULTADOS
# ─────────────────────────────────────────────────────────────
print("\n8. GENERANDO GRÁFICAS DE RESULTADOS")

# 8.1 Gráfica comparativa de métricas
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
names   = list(results.keys())
short_names = ['NB-BoW','NB-TFIDF','LR','SVM','RF','GB']
acc_tr_vals  = [results[n]['acc_train'] for n in names]
acc_te_vals  = [results[n]['acc_test']  for n in names]
f1_te_vals   = [results[n]['f1_test']   for n in names]

x = np.arange(len(names))
w = 0.35
axes[0].bar(x - w/2, acc_tr_vals, w, label='Train', color='#4C72B0', alpha=0.85)
axes[0].bar(x + w/2, acc_te_vals, w, label='Test',  color='#DD8452', alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(short_names, rotation=15)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy: Train vs Test')
axes[0].set_ylim(0, 1.05)
axes[0].legend()
for xi, (tr, te) in enumerate(zip(acc_tr_vals, acc_te_vals)):
    axes[0].text(xi - w/2, tr + 0.005, f'{tr:.3f}', ha='center', fontsize=7)
    axes[0].text(xi + w/2, te + 0.005, f'{te:.3f}', ha='center', fontsize=7)

axes[1].bar(x, f1_te_vals, color=COLORS[:len(names)], alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels(short_names, rotation=15)
axes[1].set_ylabel('F1-Score Macro')
axes[1].set_title('F1-Score Macro en Test')
axes[1].set_ylim(0, 1.05)
for xi, v in enumerate(f1_te_vals):
    axes[1].text(xi, v + 0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '04_comparativa_modelos.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → Figura 4: comparativa de modelos guardada")

# 8.2 Matrices de confusión para los 3 mejores
best_models = sorted(results.items(), key=lambda x: x[1]['f1_test'], reverse=True)[:3]
classes_order = sorted(y_test.unique())
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, res) in zip(axes, best_models):
    cm = confusion_matrix(y_test, res['y_pred'], labels=classes_order)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(classes_order)))
    ax.set_yticks(range(len(classes_order)))
    short_cls = [c.replace('_cyberbullying','').replace('not_','no_') for c in classes_order]
    ax.set_xticklabels(short_cls, rotation=40, ha='right', fontsize=8)
    ax.set_yticklabels(short_cls, fontsize=8)
    ax.set_title(name.replace(' (TF-IDF)',''), fontsize=10)
    ax.set_xlabel('Predicho')
    ax.set_ylabel('Real')
    for i in range(len(classes_order)):
        for j in range(len(classes_order)):
            val = cm_norm[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if val > 0.55 else 'black')
plt.suptitle('Matrices de Confusión Normalizadas (Top 3 Modelos)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '05_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → Figura 5: matrices de confusión guardadas")

# 8.3 Reporte del mejor modelo
best_name, best_res = best_models[0]
print(f"\n  MEJOR MODELO: {best_name}")
print(f"  Accuracy Test: {best_res['acc_test']:.4f}")
print(f"  F1 Macro Test: {best_res['f1_test']:.4f}")
print("\n  Classification Report:")
print(best_res['report'])

# 8.4 Gráfica por clase del mejor modelo
report_dict = {}
for cls in classes_order:
    y_true_bin = (y_test == cls).astype(int)
    y_pred_bin = (best_res['y_pred'] == cls).astype(int)
    p = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    r = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    report_dict[cls] = {'precision': p, 'recall': r, 'f1': f}

fig, ax = plt.subplots(figsize=(12, 5))
cls_short = [c.replace('_cyberbullying','').replace('not_','no_') for c in classes_order]
x_pos = np.arange(len(classes_order))
w = 0.25
ax.bar(x_pos - w, [report_dict[c]['precision'] for c in classes_order], w,
       label='Precision', color='#4C72B0', alpha=0.85)
ax.bar(x_pos,     [report_dict[c]['recall']    for c in classes_order], w,
       label='Recall',    color='#DD8452', alpha=0.85)
ax.bar(x_pos + w, [report_dict[c]['f1']        for c in classes_order], w,
       label='F1',        color='#55A868', alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels(cls_short, rotation=25)
ax.set_ylim(0, 1.1)
ax.set_title(f'Métricas por Clase — {best_name.replace(" (TF-IDF)","")}')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, '06_metricas_por_clase.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → Figura 6: métricas por clase guardada")

# ─────────────────────────────────────────────────────────────
# 9. TWEETS BIEN Y MAL CLASIFICADOS
# ─────────────────────────────────────────────────────────────
print("\n9. EJEMPLOS DE CLASIFICACIÓN")

X_test_list = X_test.reset_index(drop=True)
y_test_list = y_test.reset_index(drop=True)
y_pred_list = pd.Series(best_res['y_pred'])

correct_mask   = y_test_list == y_pred_list
incorrect_mask = ~correct_mask

print("\n  Tweets BIEN clasificados (5 ejemplos):")
for i in y_test_list[correct_mask].sample(5, random_state=1).index:
    print(f"  Real: {y_test_list[i]:<22} | Pred: {y_pred_list[i]:<22} | '{X_test_list[i][:60]}...'")

print("\n  Tweets MAL clasificados (5 ejemplos):")
for i in y_test_list[incorrect_mask].sample(5, random_state=1).index:
    print(f"  Real: {y_test_list[i]:<22} | Pred: {y_pred_list[i]:<22} | '{X_test_list[i][:60]}...'")

# ─────────────────────────────────────────────────────────────
# 10. FUNCIÓN DE PREDICCIÓN (Prueba de Usuario)
# ─────────────────────────────────────────────────────────────
print("\n10. FUNCIÓN DE PREDICCIÓN — PRUEBA DE USUARIO")

# Reentrenar mejor modelo sobre todos los datos con su vectorizador
print(f"  Modelo seleccionado: {best_name}")

# El mejor es SVM con TF-IDF típicamente; reentrenar pipeline completo
best_model_obj = best_res['model']

def predict_tweet(tweet_text):
    """
    Recibe un tweet en texto plano y devuelve la predicción de tipo de cyberbullying.
    """
    clean = preprocess_tweet(tweet_text)
    vec   = tfidf_vect.transform([clean])  # usa el vectorizador ya ajustado
    pred  = best_model_obj.predict(vec)[0]
    return pred

# Pruebas de usuario
test_tweets = [
    "Women should not be allowed to vote or work, they belong in the kitchen #gender",
    "Happy birthday! Hope you have an amazing day with your family!",
    "Old people are so slow and useless, they should just stay home",
    "Muslims are terrorists and should be banned from our country",
    "I love how diverse our school community is becoming each year",
    "Gay people are disgusting and should not be allowed near children",
]

print("\n  Resultados de prueba de usuario:")
print("  " + "-"*70)
for tw in test_tweets:
    pred = predict_tweet(tw)
    print(f"  [{pred:<22}] {tw[:65]}...")
print("  " + "-"*70)

# ─────────────────────────────────────────────────────────────
# 11. GUARDAR FIGURAS EN OUTPUT
# ─────────────────────────────────────────────────────────────
import shutil
for f in os.listdir(FIGURES_DIR):
    shutil.copy(os.path.join(FIGURES_DIR, f), os.path.join(OUTPUTS_DIR, f))

print("\n✓ Proyecto completado. Todas las figuras guardadas.")
print(f"✓ Mejor modelo: {best_name}")
print(f"  Accuracy: {best_res['acc_test']:.4f} | F1 Macro: {best_res['f1_test']:.4f}")
