import pandas as pd 
import numpy as np 
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class IEEEFraudSMOTE:
  """
  This class performs SMOTE data augmentation on the tabular IEEE data.
  This class is compatible with feature engineering (StandardScaler).
  """

  # Constructor
  def __init__(self, target_ratio=0.05, k_neighbors=2, random_state=42):
    self.target_ratio = target_ratio
    self.smote = SMOTE(
      sampling_strategy=target_ratio,
      k_neighbors=k_neighbors,  # safer for imbalanced data
      random_state=random_state
    )
    self.scaler = StandardScaler()

  # Augmentation function
  def augment(self, df, label_col="isFraud"):

    # ── 1. Separate features + label ──────────────
    y = df[label_col]
    X = df.drop(columns=[label_col]).copy()

    # ── 2: Handle categorical features ──────────────
    for col in X.select_dtypes(include='object').columns:
      X[col] = X[col].astype('category').cat.codes

    # ── 3: Fill missing values ──────────────
    X = X.fillna(0)

    # ── 4: Scale ──────────────
    X_scaled = self.scaler.fit_transform(X)

    print(f"Original fraud ratio: {y.mean():.4f}")

    # ── 5: Apply SMOTE ──────────────
    X_resampled, y_resampled = self.smote.fit_resample(X_scaled, y)

    print(f"New fraud ratio: {y_resampled.mean():.4f}")

    # ── 6: Inverse scale ──────────────
    X_resampled = self.scaler.inverse_transform(X_resampled)
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)

    # ── 7: Fix relational columns ──────────────
    rel_cols = ["card1", "addr1", "TransactionDT"]

    for col in rel_cols:
      if col in df_resampled.columns:
        df_resampled[col] = np.round(df_resampled[col]).astype(int)

    # ── 8. Replace relational values using nearest neighbors ──────────────

    X_real = X_scaled[:len(X)]

    nbrs = NearestNeighbors(n_neighbors=1).fit(X_real)
    _, indices = nbrs.kneighbors(X_resampled)

    for col in rel_cols:
      if col in df_resampled.columns:
        df_resampled[col] = df.iloc[indices.flatten()][col].values

    # ── 9. Add label back ──────────────
    df_resampled[label_col] = y_resampled

    return df_resampled