"""
Auto Train AI - Production-Level AutoML Streamlit Application
=============================================================
A complete end-to-end machine learning pipeline with intelligent
preprocessing, multi-model training, evaluation, and explanation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pickle
import io
import time
import logging
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    r2_score, mean_squared_error, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Auto Train AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark theme overrides */
.main { background-color: #0d0f14; }
.stApp { background-color: #0d0f14; color: #e8eaf0; }

/* Header */
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4ff 0%, #7b2fff 50%, #ff6b6b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    line-height: 1.1;
}

.hero-sub {
    color: #8892a4;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.5px;
    margin-top: 0.3rem;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #151821 0%, #1a1f2e 100%);
    border: 1px solid #252a3a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}

.metric-card h4 {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #556;
    margin: 0 0 0.4rem 0;
}

.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00d4ff;
}

/* Health score */
.health-score {
    font-family: 'Space Mono', monospace;
    font-size: 4rem;
    font-weight: 700;
    text-align: center;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #00d4ff;
    border-bottom: 1px solid #1e2435;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* Warning / info boxes */
.warn-box {
    background: rgba(255, 193, 7, 0.08);
    border-left: 3px solid #ffc107;
    padding: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.4rem 0;
    font-size: 0.88rem;
    color: #ffd54f;
}

.info-box {
    background: rgba(0, 212, 255, 0.08);
    border-left: 3px solid #00d4ff;
    padding: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.4rem 0;
    font-size: 0.88rem;
    color: #80e8ff;
}

.success-box {
    background: rgba(0, 230, 118, 0.08);
    border-left: 3px solid #00e676;
    padding: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.4rem 0;
    font-size: 0.88rem;
    color: #69f0ae;
}

/* Log box */
.log-box {
    background: #0a0c10;
    border: 1px solid #1e2435;
    border-radius: 8px;
    padding: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #4caf50;
    max-height: 250px;
    overflow-y: auto;
}

/* Leaderboard */
.leaderboard-row {
    display: flex;
    justify-content: space-between;
    padding: 0.6rem 1rem;
    border-radius: 8px;
    margin: 0.3rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}

.leaderboard-row.gold { background: rgba(255, 215, 0, 0.1); border-left: 3px solid gold; }
.leaderboard-row.silver { background: rgba(192, 192, 192, 0.08); border-left: 3px solid silver; }
.leaderboard-row.bronze { background: rgba(205, 127, 50, 0.08); border-left: 3px solid #cd7f32; }
.leaderboard-row.default { background: rgba(255,255,255,0.03); border-left: 3px solid #2a2f42; }

/* Sidebar */
.css-1d391kg, [data-testid="stSidebar"] {
    background: #0a0c12 !important;
    border-right: 1px solid #1a1f2e;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #7b2fff);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    opacity: 0.85;
    transform: translateY(-1px);
}

/* Download button */
.stDownloadButton > button {
    background: rgba(0, 230, 118, 0.12);
    color: #00e676;
    border: 1px solid #00e676;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
}

/* Tables */
.dataframe { font-size: 0.8rem !important; }

/* Progress */
.stProgress > div > div { background: linear-gradient(90deg, #00d4ff, #7b2fff); }

/* Selectbox, inputs */
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background: #151821 !important;
    border-color: #252a3a !important;
    color: #e8eaf0 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
logger = logging.getLogger("AutoTrainAI")
logger.setLevel(logging.DEBUG)

if "logs" not in st.session_state:
    st.session_state.logs = []

def log(msg: str, level: str = "INFO"):
    """Append log entry to session state."""
    ts = time.strftime("%H:%M:%S")
    entry = f"[{ts}] [{level}] {msg}"
    st.session_state.logs.append(entry)
    if level == "ERROR":
        logger.error(msg)
    else:
        logger.info(msg)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "df": None, "target": None, "problem_type": None,
        "X_train": None, "X_test": None, "y_train": None, "y_test": None,
        "preprocessor": None, "results": None, "best_model": None,
        "best_model_name": None, "best_score": None, "feature_names": None,
        "tuned_results": None, "logs": st.session_state.get("logs", []),
        "preprocessing_done": False, "training_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────

def load_data(uploaded_file) -> pd.DataFrame:
    """Load CSV into a DataFrame."""
    log("Loading dataset...")
    df = pd.read_csv(uploaded_file)
    log(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def compute_health_score(df: pd.DataFrame, target: str) -> dict:
    """
    Compute a 0-100 dataset health score based on:
    - Missing value ratio
    - Duplicate rows
    - Class imbalance (if applicable)
    Returns score + breakdown + suggestions.
    """
    score = 100
    issues = []
    suggestions = []

    # Missing values penalty
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    missing_penalty = min(30, int(missing_ratio * 100))
    score -= missing_penalty
    if missing_ratio > 0:
        issues.append(f"Missing values: {missing_ratio*100:.1f}% of all cells")
        suggestions.append("Consider imputing or dropping high-missing columns.")

    # Duplicates penalty
    dup_ratio = df.duplicated().sum() / len(df)
    dup_penalty = min(20, int(dup_ratio * 100))
    score -= dup_penalty
    if dup_ratio > 0:
        issues.append(f"Duplicate rows: {df.duplicated().sum()} ({dup_ratio*100:.1f}%)")
        suggestions.append("Remove duplicate rows before training.")

    # Imbalance (classification only)
    imbalance_ratio = 0
    if df[target].dtype == object or df[target].nunique() < 15:
        vc = df[target].value_counts(normalize=True)
        imbalance_ratio = 1 - vc.min()
        imb_penalty = min(20, int(imbalance_ratio * 30))
        score -= imb_penalty
        if imbalance_ratio > 0.3:
            issues.append(f"Class imbalance: minority class = {vc.min()*100:.1f}%")
            suggestions.append("Consider SMOTE or class_weight='balanced' to handle imbalance.")

    score = max(0, score)
    return {
        "score": score,
        "issues": issues,
        "suggestions": suggestions,
        "missing_ratio": missing_ratio,
        "dup_ratio": dup_ratio,
        "imbalance_ratio": imbalance_ratio,
    }


def analyze_data(df: pd.DataFrame, target: str) -> dict:
    """Return comprehensive data analysis results."""
    log("Analyzing dataset...")
    analysis = {}

    # Basic
    analysis["shape"] = df.shape
    analysis["dtypes"] = df.dtypes.astype(str)
    analysis["missing"] = df.isnull().sum()
    analysis["missing_pct"] = (df.isnull().sum() / len(df) * 100).round(2)
    analysis["describe"] = df.describe(include="all")

    # Constant columns
    analysis["constant_cols"] = [c for c in df.columns if df[c].nunique() <= 1]

    # Duplicate rows
    analysis["duplicate_count"] = df.duplicated().sum()

    # Correlation (numeric only)
    num_df = df.select_dtypes(include=np.number)
    if len(num_df.columns) > 1:
        corr = num_df.corr()
        analysis["corr"] = corr
        # Highly correlated pairs (|r| > 0.9, excluding self)
        high_corr = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.9:
                    high_corr.append((corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 3)))
        analysis["high_corr"] = high_corr
    else:
        analysis["corr"] = None
        analysis["high_corr"] = []

    # Warnings
    warnings_list = []
    for col in df.columns:
        pct = analysis["missing_pct"][col]
        if pct > 40:
            warnings_list.append(f"⚠️  Column '{col}' has {pct:.0f}% missing values — consider dropping.")
    if analysis["duplicate_count"] > 0:
        warnings_list.append(f"⚠️  {analysis['duplicate_count']} duplicate rows detected.")
    for col in analysis["constant_cols"]:
        warnings_list.append(f"⚠️  Column '{col}' is constant and provides no information.")
    for c1, c2, r in analysis["high_corr"]:
        warnings_list.append(f"⚠️  '{c1}' and '{c2}' are highly correlated (r={r}) — consider removing one.")
    analysis["warnings"] = warnings_list

    # Class imbalance
    if target:
        vc = df[target].value_counts(normalize=True)
        analysis["target_dist"] = vc
        if len(vc) > 1 and vc.min() < 0.2:
            analysis["imbalanced"] = True
        else:
            analysis["imbalanced"] = False

    log("Data analysis complete.")
    return analysis


def detect_problem_type(df: pd.DataFrame, target: str) -> tuple:
    """
    Detect classification vs regression.
    Returns (problem_type: str, explanation: str)
    """
    col = df[target]
    n_unique = col.nunique()
    dtype = col.dtype

    if dtype == object or n_unique < 15:
        reason = (
            f"Target column '{target}' has dtype={dtype} and {n_unique} unique values "
            f"(threshold < 15). → **Classification** problem detected."
        )
        return "classification", reason
    else:
        reason = (
            f"Target column '{target}' has dtype={dtype} and {n_unique} unique values "
            f"(threshold ≥ 15). → **Regression** problem detected."
        )
        return "regression", reason


def preprocess_data(df: pd.DataFrame, target: str):
    """
    Full preprocessing pipeline:
    - Drop ID / constant / high-cardinality columns
    - Impute missing values
    - Encode categoricals
    - Scale numerics
    Returns X_train, X_test, y_train, y_test, preprocessor, feature_names
    """
    log("Starting preprocessing pipeline...")
    df = df.copy()

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    log(f"Removed {before - len(df)} duplicate rows.")

    # Separate target
    y = df[target].copy()
    X = df.drop(columns=[target])

    # Encode target if classification string
    if y.dtype == object:
        le_target = LabelEncoder()
        y = pd.Series(le_target.fit_transform(y), name=target)
        log("Target label-encoded.")
    else:
        le_target = None

    # Drop ID-like and high-cardinality columns
    to_drop = []
    for col in X.columns:
        # High cardinality text
        if X[col].dtype == object and X[col].nunique() / len(X) > 0.5:
            to_drop.append(col)
            log(f"Dropping high-cardinality column: {col}")
        # Constant
        if X[col].nunique() <= 1:
            to_drop.append(col)
            log(f"Dropping constant column: {col}")
    X.drop(columns=list(set(to_drop)), inplace=True)

    # Identify column types
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    log(f"Numeric features: {len(num_cols)}, Categorical features: {len(cat_cols)}")

    # Numeric pipeline: impute with median + scale
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical pipeline: impute with mode + one-hot encode
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if num_cols:
        transformers.append(("num", num_pipeline, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit & transform
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Feature names
    feature_names = num_cols.copy()
    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"]["encoder"]
        cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names += cat_feature_names

    log("Preprocessing complete.")
    return X_train_proc, X_test_proc, y_train, y_test, preprocessor, feature_names


def get_models(problem_type: str) -> dict:
    """Return dict of model_name → model instance."""
    if problem_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(),
        }
    else:
        return {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        }


def train_models(X_train, X_test, y_train, y_test, problem_type: str) -> pd.DataFrame:
    """Train all models and return evaluation results DataFrame."""
    models = get_models(problem_type)
    results = []
    trained = {}

    for name, model in models.items():
        log(f"Training: {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        y_pred = model.predict(X_test)

        if problem_type == "classification":
            score = accuracy_score(y_test, y_pred)
            metric = "Accuracy"
        else:
            score = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            metric = "R² Score"

        row = {
            "Model": name,
            metric: round(score, 4),
            "Train Time (s)": round(elapsed, 3),
        }
        if problem_type == "regression":
            row["RMSE"] = round(rmse, 4)

        results.append(row)
        trained[name] = model
        log(f"  {name}: {metric} = {score:.4f} | Time = {elapsed:.3f}s")

    df_results = pd.DataFrame(results)
    metric_col = "Accuracy" if problem_type == "classification" else "R² Score"
    df_results.sort_values(metric_col, ascending=False, inplace=True)
    df_results.reset_index(drop=True, inplace=True)

    return df_results, trained


def cross_validate_models(X_train, y_train, problem_type: str) -> pd.DataFrame:
    """K-Fold cross-validation for all models."""
    log("Running 5-fold cross-validation...")
    models = get_models(problem_type)
    scoring = "accuracy" if problem_type == "classification" else "r2"
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        cv_results.append({
            "Model": name,
            "CV Mean": round(scores.mean(), 4),
            "CV Std": round(scores.std(), 4),
            "Min": round(scores.min(), 4),
            "Max": round(scores.max(), 4),
        })
        log(f"  CV {name}: mean={scores.mean():.4f} ± {scores.std():.4f}")

    df_cv = pd.DataFrame(cv_results).sort_values("CV Mean", ascending=False).reset_index(drop=True)
    return df_cv


def tune_models(X_train, y_train, top_models: list, trained: dict, problem_type: str) -> dict:
    """Hyperparameter tuning on top 2 models."""
    log("Starting hyperparameter tuning...")
    param_grids = {
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
        },
        "Random Forest Regressor": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
        },
        "Decision Tree": {
            "max_depth": [None, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
        },
        "Decision Tree Regressor": {
            "max_depth": [None, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
        },
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"],
        },
        "K-Nearest Neighbors": {
            "n_neighbors": [3, 5, 7, 11],
            "weights": ["uniform", "distance"],
        },
        "Linear Regression": {},  # No hyper-params to tune
    }

    scoring = "accuracy" if problem_type == "classification" else "r2"
    tuned = {}

    for name in top_models[:2]:
        if name not in param_grids or not param_grids[name]:
            log(f"  Skipping tuning for {name} (no param grid).")
            tuned[name] = {
                "best_params": {},
                "best_score": None,
                "model": trained[name],
            }
            continue

        log(f"  Tuning {name}...")
        model = trained[name].__class__(**{k: v for k, v in trained[name].get_params().items()
                                           if k in ["random_state"]})
        gs = RandomizedSearchCV(
            model, param_grids[name], n_iter=10,
            cv=3, scoring=scoring, random_state=42, n_jobs=-1
        )
        gs.fit(X_train, y_train)
        tuned[name] = {
            "best_params": gs.best_params_,
            "best_score": round(gs.best_score_, 4),
            "model": gs.best_estimator_,
        }
        log(f"  {name} best score: {gs.best_score_:.4f} | params: {gs.best_params_}")

    return tuned


def select_best_model(df_results: pd.DataFrame, trained: dict, problem_type: str):
    """Select best model by primary metric."""
    metric = "Accuracy" if problem_type == "classification" else "R² Score"
    best_row = df_results.iloc[0]
    best_name = best_row["Model"]
    best_score = best_row[metric]
    best_model = trained[best_name]
    log(f"Best model selected: {best_name} ({metric} = {best_score:.4f})")
    return best_name, best_score, best_model


def get_feature_importance(model, feature_names: list, problem_type: str) -> pd.DataFrame:
    """Extract feature importance from tree-based models."""
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False).head(15)
        return fi
    elif hasattr(model, "coef_"):
        coef = model.coef_[0] if problem_type == "classification" and model.coef_.ndim > 1 else model.coef_
        fi = pd.DataFrame({
            "Feature": feature_names,
            "Importance": np.abs(coef),
        }).sort_values("Importance", ascending=False).head(15)
        return fi
    return None


def auto_recommend(df: pd.DataFrame, target: str, problem_type: str) -> str:
    """Auto-recommend model type based on dataset characteristics."""
    n_rows, n_cols = df.shape
    cat_ratio = len(df.select_dtypes("object").columns) / n_cols

    if cat_ratio > 0.5:
        return "🌲 Tree-based models (Random Forest / Decision Tree) are recommended — your dataset has many categorical features."
    elif n_rows > 10000:
        return "⚡ Linear models are recommended for large datasets — they are fast and often competitive."
    elif n_cols > 20:
        return "🌲 Random Forest is recommended — handles high-dimensional data well and is robust to noise."
    else:
        return "🎯 Random Forest or Logistic Regression are good starting points for this dataset size."


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1.2rem 0 2rem 0;'>
        <div style='font-family: Space Mono, monospace; font-size: 1.3rem; font-weight: 700;
                    background: linear-gradient(135deg, #00d4ff, #7b2fff);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            AUTO TRAIN AI
        </div>
        <div style='font-size: 0.72rem; color: #445; letter-spacing: 2px; margin-top: 2px;'>
            END-TO-END AUTOML PIPELINE
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠  Overview", "📁  Upload Data", "🔍  Data Analysis",
         "⚙️  Preprocessing", "🚀  Train Models", "📊  Model Comparison",
         "🎯  Prediction", "📋  Logs"],
        label_visibility="collapsed"
    )

    if st.session_state.df is not None:
        st.divider()
        st.markdown(f"""
        <div style='font-size:0.75rem; color:#556; font-family: Space Mono, monospace;'>
        DATASET STATUS<br>
        <span style='color:#00d4ff;'>✓</span> Loaded — {st.session_state.df.shape[0]}×{st.session_state.df.shape[1]}
        </div>
        """, unsafe_allow_html=True)
        if st.session_state.target:
            st.markdown(f"""
            <div style='font-size:0.75rem; color:#556; font-family: Space Mono, monospace; margin-top:0.5rem;'>
            TARGET<br><span style='color:#7b2fff;'>{st.session_state.target}</span>
            </div>
            """, unsafe_allow_html=True)
        if st.session_state.problem_type:
            col_map = {"classification": "#ff6b6b", "regression": "#ffd54f"}
            st.markdown(f"""
            <div style='font-size:0.75rem; color:#556; font-family: Space Mono, monospace; margin-top:0.5rem;'>
            PROBLEM TYPE<br><span style='color:{col_map.get(st.session_state.problem_type,"#fff")};'>
            {st.session_state.problem_type.upper()}</span>
            </div>
            """, unsafe_allow_html=True)
        if st.session_state.best_model_name:
            st.markdown(f"""
            <div style='font-size:0.75rem; color:#556; font-family: Space Mono, monospace; margin-top:0.5rem;'>
            BEST MODEL<br><span style='color:#00e676;'>🏆 {st.session_state.best_model_name}</span>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────

# ── OVERVIEW ─────────────────────────────────
if page == "🏠  Overview":
    st.markdown("""
    <div class='hero-title'>Auto Train AI</div>
    <div class='hero-sub'>Production-grade AutoML · Upload a CSV, train models, get insights.</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    features = [
        ("🧠", "Smart Preprocessing", "Auto-handles missing values, encoding, and scaling."),
        ("🤖", "Multi-Model Training", "Trains 4+ models and ranks them automatically."),
        ("🔬", "Cross Validation", "K-Fold CV for reliable, unbiased evaluation."),
        ("💡", "Explainability", "Feature importance and model explanation."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], features):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='font-size:2rem;'>{icon}</div>
                <div style='font-family: Space Mono, monospace; font-size:0.85rem; font-weight:700; color:#e8eaf0; margin: 0.5rem 0 0.3rem;'>{title}</div>
                <div style='font-size:0.78rem; color:#667; line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>PIPELINE FLOW</div>", unsafe_allow_html=True)

    steps = ["Upload CSV", "Analyze", "Preprocess", "Detect Type", "Train", "Evaluate", "Tune", "Explain", "Predict"]
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            arrow = "→" if i < len(steps) - 1 else "✓"
            st.markdown(f"""
            <div style='text-align:center;'>
                <div style='background:linear-gradient(135deg,#151821,#1a1f2e); border:1px solid #252a3a;
                           border-radius:8px; padding:0.6rem 0.2rem; font-family:Space Mono,monospace;
                           font-size:0.65rem; color:#00d4ff;'>{step}</div>
                <div style='color:#334; font-size:1rem; margin-top:0.2rem;'>{arrow}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    👈 Start by navigating to <strong>Upload Data</strong> in the sidebar to load your CSV dataset.
    </div>
    """, unsafe_allow_html=True)

# ── UPLOAD DATA ───────────────────────────────
elif page == "📁  Upload Data":
    st.markdown("<div class='section-header'>UPLOAD DATASET</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded:
        df = load_data(uploaded)
        st.session_state.df = df

        st.markdown(f"""
        <div class='success-box'>✓ Dataset loaded — {df.shape[0]} rows × {df.shape[1]} columns</div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]}")
        c3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>DATASET PREVIEW</div>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>SELECT TARGET COLUMN</div>", unsafe_allow_html=True)

        target = st.selectbox("Choose the column you want to predict:", df.columns.tolist())
        if st.button("Confirm Target →"):
            st.session_state.target = target
            ptype, reason = detect_problem_type(df, target)
            st.session_state.problem_type = ptype
            log(f"Target set: {target} | Problem: {ptype}")
            st.markdown(f"""
            <div class='success-box'>✓ Target: <strong>{target}</strong></div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class='info-box'>{reason}</div>
            """, unsafe_allow_html=True)

# ── DATA ANALYSIS ─────────────────────────────
elif page == "🔍  Data Analysis":
    st.markdown("<div class='section-header'>DATA ANALYSIS</div>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state.df
        target = st.session_state.target

        analysis = analyze_data(df, target)

        # Health Score
        if target:
            health = compute_health_score(df, target)
            hs = health["score"]
            color = "#00e676" if hs >= 75 else ("#ffc107" if hs >= 50 else "#ff5252")
            st.markdown(f"""
            <div class='metric-card' style='text-align:center;'>
                <div class='section-header'>DATASET HEALTH SCORE</div>
                <div class='health-score' style='color:{color};'>{hs}</div>
                <div style='color:#556; font-size:0.8rem; font-family:Space Mono,monospace;'>/100</div>
            </div>
            """, unsafe_allow_html=True)

            if health["issues"]:
                for iss in health["issues"]:
                    st.markdown(f"<div class='warn-box'>{iss}</div>", unsafe_allow_html=True)
            if health["suggestions"]:
                st.markdown("<br>**Suggestions to improve score:**")
                for sug in health["suggestions"]:
                    st.markdown(f"<div class='info-box'>💡 {sug}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🔗 Correlation", "⚠️ Warnings", "📈 Distributions"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='section-header'>MISSING VALUES</div>", unsafe_allow_html=True)
                mv = pd.DataFrame({
                    "Column": analysis["missing"].index,
                    "Missing": analysis["missing"].values,
                    "% Missing": analysis["missing_pct"].values,
                })
                mv = mv[mv["Missing"] > 0]
                if len(mv):
                    st.dataframe(mv, use_container_width=True, hide_index=True)
                else:
                    st.markdown("<div class='success-box'>✓ No missing values</div>", unsafe_allow_html=True)

            with c2:
                st.markdown("<div class='section-header'>DATA TYPES</div>", unsafe_allow_html=True)
                st.dataframe(
                    pd.DataFrame({"Column": analysis["dtypes"].index, "Type": analysis["dtypes"].values}),
                    use_container_width=True, hide_index=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>SUMMARY STATISTICS</div>", unsafe_allow_html=True)
            st.dataframe(analysis["describe"], use_container_width=True)

        with tab2:
            if analysis["corr"] is not None:
                st.markdown("<div class='section-header'>CORRELATION HEATMAP</div>", unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(10, 6))
                fig.patch.set_facecolor('#0d0f14')
                ax.set_facecolor('#0d0f14')
                sns.heatmap(
                    analysis["corr"], annot=True, fmt=".2f", cmap="coolwarm",
                    ax=ax, linewidths=0.3, linecolor="#1a1f2e",
                    annot_kws={"size": 8}, cbar_kws={"shrink": 0.8}
                )
                ax.tick_params(colors='#8892a4', labelsize=8)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                if analysis["high_corr"]:
                    st.markdown("<div class='section-header'>HIGHLY CORRELATED PAIRS (|r| > 0.9)</div>", unsafe_allow_html=True)
                    st.dataframe(
                        pd.DataFrame(analysis["high_corr"], columns=["Feature A", "Feature B", "Correlation"]),
                        use_container_width=True, hide_index=True
                    )
            else:
                st.info("Not enough numeric columns for correlation analysis.")

        with tab3:
            if analysis["warnings"]:
                for w in analysis["warnings"]:
                    st.markdown(f"<div class='warn-box'>{w}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='success-box'>✓ No major data quality issues found.</div>", unsafe_allow_html=True)

            if target and analysis.get("imbalanced"):
                st.markdown("""
                <div class='warn-box'>
                ⚠️ <strong>Class Imbalance Detected</strong> — minority class &lt; 20% of data.<br>
                Consider using SMOTE (Synthetic Minority Oversampling Technique) to balance classes,
                or set <code>class_weight='balanced'</code> in your model.
                </div>
                """, unsafe_allow_html=True)

        with tab4:
            if target:
                fig, ax = plt.subplots(figsize=(7, 3))
                fig.patch.set_facecolor('#0d0f14')
                ax.set_facecolor('#151821')
                vc = df[target].value_counts()
                colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(vc)))
                ax.bar(vc.index.astype(str), vc.values, color=colors, edgecolor='#1a1f2e')
                ax.set_title(f"Target Distribution: {target}", color='#8892a4', fontsize=10, pad=10)
                ax.tick_params(colors='#8892a4', labelsize=8)
                ax.spines[:].set_color('#1a1f2e')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# ── PREPROCESSING ─────────────────────────────
elif page == "⚙️  Preprocessing":
    st.markdown("<div class='section-header'>INTELLIGENT PREPROCESSING</div>", unsafe_allow_html=True)

    if st.session_state.df is None or st.session_state.target is None:
        st.warning("Please upload a dataset and select a target column first.")
    else:
        st.markdown("""
        <div class='info-box'>
        The preprocessing pipeline will: remove duplicates → impute missing values
        → drop ID/constant/high-cardinality columns → encode categoricals → scale numerics.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        df = st.session_state.df
        target = st.session_state.target

        num_cols = df.drop(columns=[target]).select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.drop(columns=[target]).select_dtypes(include="object").columns.tolist()

        with c1:
            st.markdown(f"<div class='metric-card'><h4>NUMERIC FEATURES</h4><div class='value'>{len(num_cols)}</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-card'><h4>CATEGORICAL FEATURES</h4><div class='value'>{len(cat_cols)}</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("▶ Run Preprocessing Pipeline"):
            with st.spinner("Preprocessing data..."):
                try:
                    X_train, X_test, y_train, y_test, preprocessor, feature_names = preprocess_data(
                        df, target
                    )
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.preprocessor = preprocessor
                    st.session_state.feature_names = feature_names
                    st.session_state.preprocessing_done = True

                    st.markdown(f"""
                    <div class='success-box'>
                    ✓ Preprocessing complete!<br>
                    Training set: {X_train.shape[0]} samples × {X_train.shape[1]} features<br>
                    Test set: {X_test.shape[0]} samples × {X_test.shape[1]} features
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Train Samples", f"{X_train.shape[0]:,}")
                    c2.metric("Test Samples", f"{X_test.shape[0]:,}")
                    c3.metric("Features (after enc)", f"{X_train.shape[1]}")

                except Exception as e:
                    st.error(f"Preprocessing failed: {e}")
                    log(str(e), "ERROR")

# ── TRAIN MODELS ──────────────────────────────
elif page == "🚀  Train Models":
    st.markdown("<div class='section-header'>MODEL TRAINING & EVALUATION</div>", unsafe_allow_html=True)

    if not st.session_state.preprocessing_done:
        st.warning("Please complete preprocessing first.")
    else:
        problem_type = st.session_state.problem_type
        recommendation = auto_recommend(st.session_state.df, st.session_state.target, problem_type)
        st.markdown(f"<div class='info-box'>🤖 Auto-Recommendation: {recommendation}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Train All Models"):
            progress = st.progress(0, text="Initializing...")

            with st.spinner("Training models..."):
                try:
                    progress.progress(20, text="Training models...")
                    df_results, trained = train_models(
                        st.session_state.X_train, st.session_state.X_test,
                        st.session_state.y_train, st.session_state.y_test,
                        problem_type
                    )
                    progress.progress(50, text="Cross-validating...")
                    df_cv = cross_validate_models(
                        st.session_state.X_train, st.session_state.y_train, problem_type
                    )
                    progress.progress(75, text="Tuning top models...")
                    top2 = df_results["Model"].head(2).tolist()
                    tuned = tune_models(
                        st.session_state.X_train, st.session_state.y_train,
                        top2, trained, problem_type
                    )
                    progress.progress(90, text="Selecting best model...")

                    # Re-evaluate tuned models
                    metric_col = "Accuracy" if problem_type == "classification" else "R² Score"
                    for name, info in tuned.items():
                        if info["model"] is not None:
                            info["model"].fit(st.session_state.X_train, st.session_state.y_train)
                            y_pred = info["model"].predict(st.session_state.X_test)
                            if problem_type == "classification":
                                score = accuracy_score(st.session_state.y_test, y_pred)
                            else:
                                score = r2_score(st.session_state.y_test, y_pred)
                            # Update if improved
                            idx = df_results[df_results["Model"] == name].index
                            if len(idx) and score > df_results.loc[idx[0], metric_col]:
                                df_results.loc[idx[0], metric_col] = round(score, 4)
                                trained[name] = info["model"]
                                log(f"  {name} improved after tuning → {score:.4f}")

                    df_results.sort_values(metric_col, ascending=False, inplace=True)
                    df_results.reset_index(drop=True, inplace=True)

                    best_name, best_score, best_model = select_best_model(df_results, trained, problem_type)

                    st.session_state.results = df_results
                    st.session_state.df_cv = df_cv
                    st.session_state.trained = trained
                    st.session_state.tuned_results = tuned
                    st.session_state.best_model = best_model
                    st.session_state.best_model_name = best_name
                    st.session_state.best_score = best_score
                    st.session_state.training_done = True
                    progress.progress(100, text="Done!")

                except Exception as e:
                    st.error(f"Training failed: {e}")
                    log(str(e), "ERROR")
                    progress.empty()

        if st.session_state.training_done and st.session_state.results is not None:
            df_results = st.session_state.results
            problem_type = st.session_state.problem_type
            metric_col = "Accuracy" if problem_type == "classification" else "R² Score"

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>🏆 MODEL LEADERBOARD</div>", unsafe_allow_html=True)

            medals = ["🥇", "🥈", "🥉"] + ["  "] * 10
            classes = ["gold", "silver", "bronze"] + ["default"] * 10
            for i, (_, row) in enumerate(df_results.iterrows()):
                best_flag = " ← BEST" if i == 0 else ""
                st.markdown(f"""
                <div class='leaderboard-row {classes[i]}'>
                    <span>{medals[i]} {row['Model']}{best_flag}</span>
                    <span>{metric_col}: <strong>{row[metric_col]}</strong> &nbsp;|&nbsp; {row['Train Time (s)']}s</span>
                </div>
                """, unsafe_allow_html=True)

            # Cross-validation results
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>5-FOLD CROSS VALIDATION</div>", unsafe_allow_html=True)
            st.markdown("""
            <div class='info-box'>
            Cross-validation splits data into 5 folds, trains on 4 and tests on 1 (repeated 5×).
            This gives a much more reliable estimate of model performance than a single train/test split.
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(st.session_state.df_cv, use_container_width=True, hide_index=True)

            # Hyperparameter tuning results
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>HYPERPARAMETER TUNING</div>", unsafe_allow_html=True)
            for name, info in st.session_state.tuned_results.items():
                if info["best_params"]:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div style='font-family:Space Mono,monospace;font-size:0.85rem;color:#e8eaf0;'>{name}</div>
                        <div style='color:#8892a4;font-size:0.78rem;margin-top:0.5rem;'>Best Score: <span style='color:#00d4ff;'>{info['best_score']}</span></div>
                        <div style='color:#8892a4;font-size:0.78rem;'>Best Params: <code style='color:#7b2fff;'>{info['best_params']}</code></div>
                    </div>
                    """, unsafe_allow_html=True)

            # Feature importance
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>FEATURE IMPORTANCE</div>", unsafe_allow_html=True)

            fi = get_feature_importance(
                st.session_state.best_model,
                st.session_state.feature_names,
                problem_type
            )
            if fi is not None and len(fi):
                top_feat = fi.iloc[0]["Feature"]
                st.markdown(f"""
                <div class='info-box'>
                💡 <strong>Feature '{top_feat}'</strong> contributes most to the prediction.
                The model relies heavily on this feature when making decisions.
                </div>
                """, unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_facecolor('#0d0f14')
                ax.set_facecolor('#151821')
                colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(fi)))
                ax.barh(fi["Feature"], fi["Importance"], color=colors[::-1], edgecolor='#1a1f2e')
                ax.set_xlabel("Importance", color='#8892a4')
                ax.set_title(f"Top Features — {st.session_state.best_model_name}", color='#8892a4', pad=10)
                ax.tick_params(colors='#8892a4', labelsize=8)
                ax.spines[:].set_color('#1a1f2e')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Feature importance not available for this model type.")

            # Confusion matrix / error analysis
            if problem_type == "classification":
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>CONFUSION MATRIX & ERROR ANALYSIS</div>", unsafe_allow_html=True)

                y_pred = st.session_state.best_model.predict(st.session_state.X_test)
                cm = confusion_matrix(st.session_state.y_test, y_pred)

                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor('#0d0f14')
                ax.set_facecolor('#151821')
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(ax=ax, cmap="Blues", colorbar=False)
                ax.set_title("Confusion Matrix", color='#8892a4', pad=10)
                ax.tick_params(colors='#8892a4')
                ax.xaxis.label.set_color('#8892a4')
                ax.yaxis.label.set_color('#8892a4')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Misclassified samples
                X_test_orig = st.session_state.X_test
                misclassified_idx = np.where(y_pred != st.session_state.y_test)[0]
                if len(misclassified_idx):
                    st.markdown(f"""
                    <div class='warn-box'>
                    ⚠️ {len(misclassified_idx)} samples misclassified ({len(misclassified_idx)/len(y_pred)*100:.1f}%).
                    Common causes: overlapping feature distributions, insufficient training data, or noisy labels.
                    </div>
                    """, unsafe_allow_html=True)

            # Download model
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>EXPORT MODEL</div>", unsafe_allow_html=True)

            model_bytes = io.BytesIO()
            pickle.dump(st.session_state.best_model, model_bytes)
            model_bytes.seek(0)

            prep_bytes = io.BytesIO()
            pickle.dump(st.session_state.preprocessor, prep_bytes)
            prep_bytes.seek(0)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "⬇ Download Best Model (.pkl)",
                    data=model_bytes,
                    file_name=f"best_model_{st.session_state.best_model_name.replace(' ','_')}.pkl",
                    mime="application/octet-stream"
                )
            with c2:
                st.download_button(
                    "⬇ Download Preprocessor (.pkl)",
                    data=prep_bytes,
                    file_name="preprocessor.pkl",
                    mime="application/octet-stream"
                )

# ── MODEL COMPARISON ──────────────────────────
elif page == "📊  Model Comparison":
    st.markdown("<div class='section-header'>MODEL COMPARISON VISUALIZATION</div>", unsafe_allow_html=True)

    if not st.session_state.training_done:
        st.warning("Please train models first.")
    else:
        df_results = st.session_state.results
        problem_type = st.session_state.problem_type
        metric_col = "Accuracy" if problem_type == "classification" else "R² Score"

        # Bar chart
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor('#0d0f14')
        ax.set_facecolor('#151821')

        colors = ["#00d4ff" if i == 0 else "#2a2f42" for i in range(len(df_results))]
        bars = ax.bar(df_results["Model"], df_results[metric_col], color=colors, edgecolor='#1a1f2e', width=0.5)

        # Value labels
        for bar, val in zip(bars, df_results[metric_col]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', color='#8892a4', fontsize=8, fontfamily='monospace')

        ax.set_ylabel(metric_col, color='#8892a4')
        ax.set_title(f"Model Comparison — {metric_col}", color='#e8eaf0', pad=12)
        ax.tick_params(colors='#8892a4', labelsize=8)
        ax.spines[:].set_color('#1a1f2e')
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Training time chart
        st.markdown("<br>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(9, 3))
        fig2.patch.set_facecolor('#0d0f14')
        ax2.set_facecolor('#151821')
        colors2 = plt.cm.plasma(np.linspace(0.2, 0.8, len(df_results)))
        ax2.bar(df_results["Model"], df_results["Train Time (s)"], color=colors2, edgecolor='#1a1f2e', width=0.5)
        ax2.set_ylabel("Time (seconds)", color='#8892a4')
        ax2.set_title("Training Time per Model", color='#e8eaf0', pad=12)
        ax2.tick_params(colors='#8892a4', labelsize=8)
        ax2.spines[:].set_color('#1a1f2e')
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # Best model summary
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='metric-card' style='border-color:#00d4ff;'>
            <div class='section-header'>🏆 BEST MODEL</div>
            <div style='font-family:Space Mono,monospace;font-size:1.5rem;color:#00d4ff;'>{st.session_state.best_model_name}</div>
            <div style='color:#8892a4;margin-top:0.5rem;'>{metric_col}: <span style='color:#00e676;font-size:1.2rem;font-family:Space Mono,monospace;'>{st.session_state.best_score:.4f}</span></div>
            <div style='color:#556;font-size:0.8rem;margin-top:0.5rem;'>
            This model achieved the highest {metric_col} on the held-out test set
            and performed consistently across cross-validation folds.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Full results table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>FULL RESULTS TABLE</div>", unsafe_allow_html=True)
        st.dataframe(df_results, use_container_width=True, hide_index=True)

# ── PREDICTION ────────────────────────────────
elif page == "🎯  Prediction":
    st.markdown("<div class='section-header'>MAKE A PREDICTION</div>", unsafe_allow_html=True)

    if not st.session_state.training_done:
        st.warning("Please train models first.")
    else:
        df = st.session_state.df
        target = st.session_state.target
        problem_type = st.session_state.problem_type

        feature_cols = [c for c in df.columns if c != target]
        st.markdown(f"""
        <div class='info-box'>
        Enter values for each feature below. The best model
        (<strong>{st.session_state.best_model_name}</strong>) will make a prediction.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        input_data = {}
        cols = st.columns(min(3, len(feature_cols)))

        for i, col in enumerate(feature_cols):
            with cols[i % len(cols)]:
                dtype = df[col].dtype
                if dtype == object:
                    unique_vals = df[col].dropna().unique().tolist()
                    input_data[col] = st.selectbox(col, unique_vals, key=f"pred_{col}")
                elif dtype in [np.float64, np.float32]:
                    mn, mx = float(df[col].min()), float(df[col].max())
                    med = float(df[col].median())
                    input_data[col] = st.number_input(col, min_value=mn, max_value=mx, value=med, key=f"pred_{col}")
                else:
                    mn, mx = int(df[col].min()), int(df[col].max())
                    med = int(df[col].median())
                    input_data[col] = st.number_input(col, min_value=mn, max_value=mx, value=med, step=1, key=f"pred_{col}")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🎯 Predict"):
            try:
                input_df = pd.DataFrame([input_data])
                processed = st.session_state.preprocessor.transform(input_df)
                prediction = st.session_state.best_model.predict(processed)[0]

                st.markdown(f"""
                <div class='metric-card' style='border-color:#00e676; text-align:center;'>
                    <div class='section-header'>PREDICTION RESULT</div>
                    <div style='font-family:Space Mono,monospace;font-size:2.5rem;color:#00e676;'>{prediction}</div>
                    <div style='color:#556;font-size:0.78rem;margin-top:0.4rem;'>
                    Model: {st.session_state.best_model_name} | Target: {target}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Show probability (classification only)
                if problem_type == "classification" and hasattr(st.session_state.best_model, "predict_proba"):
                    proba = st.session_state.best_model.predict_proba(processed)[0]
                    classes = st.session_state.best_model.classes_
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<div class='section-header'>MODEL CONFIDENCE</div>", unsafe_allow_html=True)
                    for cls, prob in zip(classes, proba):
                        bar_w = int(prob * 100)
                        color = "#00e676" if cls == prediction else "#2a2f42"
                        st.markdown(f"""
                        <div style='margin:0.3rem 0;'>
                            <div style='font-family:Space Mono,monospace;font-size:0.75rem;color:#8892a4;margin-bottom:3px;'>
                                Class {cls}: {prob*100:.1f}%
                            </div>
                            <div style='background:#1a1f2e;border-radius:4px;height:8px;width:100%;'>
                                <div style='background:{color};height:8px;border-radius:4px;width:{bar_w}%;'></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                log(f"Prediction made: {prediction}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                log(str(e), "ERROR")

# ── LOGS ──────────────────────────────────────
elif page == "📋  Logs":
    st.markdown("<div class='section-header'>PIPELINE LOGS</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    Real-time log of all pipeline steps — useful for debugging and understanding
    what the AutoML system is doing at each stage.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.logs:
        log_text = "\n".join(st.session_state.logs)
        st.markdown(f"<div class='log-box'>{log_text.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
        st.download_button("⬇ Download Logs", data=log_text, file_name="autotrain_logs.txt")
    else:
        st.info("No logs yet. Start by uploading a dataset.")

    if st.button("🗑 Clear Logs"):
        st.session_state.logs = []
        st.rerun()
