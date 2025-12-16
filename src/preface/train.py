#!/usr/bin/env python3

import os
import time
import json
from typing import List, Optional, Any, Dict, Union

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import typer



# Constants
EXCLUDE_CHRS: List[str] = ['13', '18', '21', 'X', 'Y']
COLOR_A: str = '#8DD1C6'
COLOR_B: str = '#E3C88A'
COLOR_C: str = '#C87878'



def get_m_diff(v1: np.ndarray, v2: np.ndarray, abs_val: bool = True) -> float:
    if not abs_val:
        return float(np.mean(v1 - v2))
    return float(np.mean(np.abs(v1 - v2)))

def get_sd_diff(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.std(np.abs(v1 - v2), ddof=1)) # ddof=1 for sample sd

def plot_performance(
    v1: np.ndarray, 
    v2: np.ndarray, 
    pca_explained_variance_ratio: np.ndarray, 
    n_feat: int, 
    xlab: str, 
    ylab: str, 
    path: str
) -> List[float]:
    
    # R plot layout: 1 row, 3 columns.
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: PCA Importance
    ax = axes[0]
    y_vals = pca_explained_variance_ratio
    x_vals = np.arange(1, len(y_vals) + 1)
    
    # Filtering zeros for log
    mask = y_vals > 0
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]
    
    ax.plot(np.log(x_vals), np.log(y_vals), color=COLOR_A, linewidth=2)
    ax.set_xlabel('Principal components (log scale)')
    ax.set_ylabel('Proportion of variance (log scale)')
    ax.set_title('PCA')
    
    # Vertical line at n_feat
    log_n_feat = np.log(n_feat)
    ylim = ax.get_ylim()
    ax.vlines(log_n_feat, ylim[0], ylim[1] * 0.99, colors=COLOR_C, linestyles='dotted', linewidth=3)
    ax.text(log_n_feat, ylim[1], 'Number of features', color=COLOR_C, ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Scatter Plot
    ax = axes[1]
    mx = max(float(np.max(v1)), float(np.max(v2)))
    ax.scatter(v1, v2, s=10, c='black', alpha=0.6)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(0, mx)
    ax.set_ylim(0, mx)
    ax.set_title('Scatter plot')
    
    # OLS fit
    intercept: float = 0.0
    slope: float = 0.0
    correlation: float = 0.0
    
    if len(np.unique(v1)) > 1:
        reg = LinearRegression().fit(v1.reshape(-1, 1), v2)
        fit_line = reg.predict(np.array([[0], [mx]]))
        ax.plot([0, mx], fit_line, color=COLOR_A, linestyle='--', linewidth=2, label='OLS fit')
        intercept = float(reg.intercept_)
        slope = float(reg.coef_[0])
        correlation = float(np.corrcoef(v1, v2)[0, 1])
    else:
        # Fallback if v1 is constant
        mean_v2 = float(np.mean(v2))
        ax.plot([0, mx], [mean_v2, mean_v2], color=COLOR_A, linestyle='--', linewidth=2, label='Mean fit')
        intercept = mean_v2
        slope = 0.0
        correlation = 0.0

    # Identity line
    ax.plot([0, mx], [0, mx], color=COLOR_B, linestyle=':', linewidth=3, label='f(x)=x')
    
    ax.legend()
    ax.text(0, mx * 1.03, f'(r = {correlation:.3g})', fontsize=9, ha='left')
    
    # Plot 3: Histogram of errors
    ax = axes[2]
    errors = v1 - v2
    n_bins = max(20, len(v1)//10)
    counts, bins, patches = ax.hist(errors, bins=n_bins, density=True, color='black', alpha=0.5)
    ax.set_xlabel(f'{xlab} - {ylab}')
    ax.set_ylabel('Density')
    ax.set_title('Histogram')
    
    mx_hist = float(np.max(counts)) if len(counts) > 0 else 0.1
    # Mean error line
    mean_err = get_m_diff(v1, v2, abs_val=False)
    ax.vlines(mean_err, 0, mx_hist, colors=COLOR_A, linestyles='--', linewidth=3, label='mean error')
    ax.vlines(0, 0, mx_hist, colors=COLOR_B, linestyles=':', linewidth=3, label='x=0')
    ax.legend()
    
    mae = get_m_diff(v1, v2)
    sd_diff = get_sd_diff(v1, v2)
    min_bin = float(min(bins)) if len(bins) > 0 else 0.0
    ax.text(min_bin, mx_hist * 1.03, f'(MAE = {mae:.3g} ± {sd_diff:.3g})', fontsize=9, ha='left')
    
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    
    return [intercept, slope, mae, sd_diff, correlation]

def load_bed_file(filepath: str) -> Optional[np.ndarray]:
    try:
        # Assuming BED has header
        df = pd.read_csv(filepath, sep='\t')
        # Check columns
        if 'ratio' not in df.columns:
            # Fallback if no header
            pass
        
        df = df[df['chr'] != 'Y']
        return df['ratio'].values
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def load_bed_full(filepath: str) -> pd.DataFrame:
    # For creating the full training frame with coordinates
    df = pd.read_csv(filepath, sep='\t')
    return df

def train_neural_network(X_train: np.ndarray, Y_train: np.ndarray, hidden_units: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(hidden_units, activation='sigmoid'),
        layers.Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    model.fit(X_train, Y_train, epochs=200, batch_size=32, verbose=0, callbacks=[early_stop])
    return model


def preface_train(
    config_file: str = typer.Option(..., "--config", help="Path to config file"),
    out_dir: str = typer.Option(..., "--outdir", help="Output directory"),
    n_feat: int = typer.Option(50, "--nfeat", help="Number of features (PCA components)"),
    hidden: int = typer.Option(2, "--hidden", help="Hidden units in NN"),
    cpus: int = typer.Option(1, "--cpus", help="Number of CPUs"),
    femprop: bool = typer.Option(False, "--femprop", help="Include females in training"),
    olm: bool = typer.Option(False, "--olm", help="Use Ordinary Linear Model instead of NN"),
    noskewcorrect: bool = typer.Option(False, "--noskewcorrect", help="Disable skew correction")
) -> None:
    """
    Train the PREFACE model.
    """
    start_time = time.time()
    
    if not os.path.exists(config_file):
        typer.echo(f"The file '{config_file}' does not exist.")
        raise typer.Exit(code=1)
        
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    skewcorrect: bool = not noskewcorrect
    train_gender: List[str] = ['M']
    if femprop:
        train_gender = ['M', 'F']
        
    out_dir_path: str = os.path.join(out_dir, '')
    
    # Load config
    config = pd.read_csv(config_file, sep='\t', comment='#', dtype={'gender': str, 'ID': str})
    
    # Check samples
    labeled_samples = config[config['gender'].isin(train_gender)]
    if len(labeled_samples) < n_feat:
        typer.echo(f"Please provide at least {n_feat} labeled samples.")
        raise typer.Exit(code=1)
        
    # Shuffle
    config = config.sample(frac=1, random_state=1).reset_index(drop=True)
    
    # Load first file for structure
    first_path: str = str(config['filepath'].iloc[0])
    training_frame_meta = load_bed_full(first_path)
    training_frame_meta = training_frame_meta[['chr', 'start', 'end']]
    
    # Load all ratios in parallel
    typer.echo("Loading samples...")
    results: List[Optional[np.ndarray]] = Parallel(n_jobs=cpus)(delayed(load_bed_file)(str(f)) for f in config['filepath'])
    
    lengths: List[int] = [len(x) for x in results if x is not None]
    if len(set(lengths)) > 1:
        typer.echo("Error: Input BED files have different numbers of bins (excluding Y).")
        raise typer.Exit(code=1)
        
    # Filter out Nones and stack
    valid_results: List[np.ndarray] = [res for res in results if res is not None]
    training_frame_sub = np.column_stack(valid_results)
    
    # Ensure meta alignment
    training_frame_meta = training_frame_meta[training_frame_meta['chr'] != 'Y']
    if len(training_frame_meta) != training_frame_sub.shape[0]:
        typer.echo("Mismatch in row counts between metadata and loaded data.")
        raise typer.Exit(code=1)
        
    is_x = training_frame_meta['chr'] == 'X'
    x_ratios_raw = training_frame_sub[is_x, :]
    x_ratios = 2 ** np.nanmean(x_ratios_raw, axis=0)
    
    typer.echo("Creating training frame...")
    
    mask_keep = ~training_frame_meta['chr'].isin(EXCLUDE_CHRS)
    
    training_frame_filtered = training_frame_sub[mask_keep, :]
    training_frame_meta_filtered = training_frame_meta[mask_keep]
    
    # Transpose: Samples as rows, Features as columns
    training_frame = training_frame_filtered.T
    
    feature_names = (training_frame_meta_filtered['chr'].astype(str) + ':' + 
                     training_frame_meta_filtered['start'].astype(str) + '-' + 
                     training_frame_meta_filtered['end'].astype(str)).values
    
    training_df = pd.DataFrame(training_frame, columns=feature_names)
    
    # Filter NAs
    na_threshold = len(config) * 0.01
    cols_to_keep = training_df.isna().sum() < na_threshold
    training_df = training_df.loc[:, cols_to_keep]
    
    possible_features = training_df.columns.values
    mean_features = training_df.mean()
    
    training_df = training_df.fillna(mean_features)
    
    typer.echo(f"Remaining training features after 'NA' filtering: {len(possible_features)}")
    
    os.makedirs(os.path.join(out_dir_path, 'training_repeats'), exist_ok=True)
    
    repeats: int = 10
    test_percentage: float = 1.0 / repeats
    
    train_mask = config['gender'].isin(train_gender)
    train_indices_all: List[int] = config.index[train_mask].tolist()
    
    n_train_samples: int = len(train_indices_all)
    test_number: int = int(n_train_samples * test_percentage)
    
    max_feat: int = n_train_samples - test_number - 1
    if n_feat > max_feat:
        typer.echo(f"Too few samples were provided for --nfeat {n_feat}, using --nfeat {max_feat}")
        n_feat = max_feat
        
    train_subset_df = training_df.iloc[train_indices_all].reset_index(drop=True)
    y_all: np.ndarray = config.loc[train_indices_all, 'FF'].astype(float).values
    
    results_repeats: List[Dict[str, Any]] = []
    
    for i in range(1, repeats + 1):
        typer.echo(f"Model training | Repeat {i}/{repeats} ...")
        
        start_idx: int = int((i - 1) * test_number)
        end_idx: int = int(i * test_number)
        
        test_idxs_local: List[int] = list(range(start_idx, end_idx))
        train_idxs_local: List[int] = list(set(range(len(train_subset_df))) - set(test_idxs_local))
        
        X_tr_local = train_subset_df.iloc[train_idxs_local]
        X_te_local = train_subset_df.iloc[test_idxs_local]
        Y_tr_local = y_all[train_idxs_local]
        Y_te_local = y_all[test_idxs_local]
        
        typer.echo("\tExecuting principal component analysis ...")
        n_components_pca: int = min(n_feat * 10, len(X_tr_local) - 1)
        pca = PCA(n_components=n_components_pca)
        pca.fit(X_tr_local)
        
        X_train_pca = pca.transform(X_tr_local)
        X_test_pca = pca.transform(X_te_local)
        
        X_train_model = X_train_pca[:, :n_feat]
        X_test_model = X_test_pca[:, :n_feat]
        
        prediction: Optional[np.ndarray] = None
        if olm:
            typer.echo("\tTraining ordinary linear model ...")
            model = LinearRegression()
            model.fit(X_train_model, Y_tr_local)
            prediction = model.predict(X_test_model)
        else:
            typer.echo("\tTraining neural network ...")
            model = train_neural_network(X_train_model, Y_tr_local, hidden)
            prediction = model.predict(X_test_model).flatten()
            
        info = plot_performance(prediction, Y_te_local, pca.explained_variance_ratio_, n_feat, 
                                'PREFACE (%)', 'FF (%)', os.path.join(out_dir_path, 'training_repeats', f'repeat_{i}.png'))
        
        results_repeats.append({
            'intercept': info[0],
            'slope': info[1],
            'prediction': prediction
        })
        
    predictions: np.ndarray = np.concatenate([r['prediction'] for r in results_repeats])
    
    the_intercept: float = 0.0
    the_slope: float = 1.0
    
    n_used: int = int(test_number * repeats)
    y_used: np.ndarray = y_all[:n_used]
    
    if skewcorrect:
        np.random.seed(1)
        n_pred: int = len(predictions)
        p: np.ndarray = np.random.choice(n_pred, max(1, n_pred // 4), replace=False)
        
        pred_p = predictions[p]
        y_p = y_used[p]
        
        reg_skew = LinearRegression().fit(pred_p.reshape(-1, 1), y_p)
        the_intercept = float(reg_skew.intercept_)
        the_slope = float(reg_skew.coef_[0])
        
        typer.echo("Correction for skew:")
        typer.echo(f"\tIntercept: {the_intercept}")
        typer.echo(f"\tSlope: {the_slope}")

    typer.echo("Training FFX Model...")
    
    mask_m = config['gender'] == 'M'
    v1_m: np.ndarray = config.loc[mask_m, 'FF'].astype(float).values
    v2_m: np.ndarray = x_ratios[mask_m]
    
    X_rlm = sm.add_constant(v1_m)
    rlm_model = sm.RLM(v2_m, X_rlm, M=sm.robust.norms.HuberT())
    rlm_results = rlm_model.fit()
    
    fit_params = rlm_results.params
    the_intercept_X: float = fit_params[0]
    the_slope_X: float = fit_params[1]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    ax.scatter(v1_m, v2_m, s=10, c='black', alpha=0.6)
    ax.set_xlabel('FF (%)')
    ax.set_ylabel('μ(ratio X)')
    mx = max(v1_m) if len(v1_m) > 0 else 1
    ax.set_xlim(0, mx)
    
    x_range = np.array([min(v1_m) if len(v1_m)>0 else 0, max(v1_m) if len(v1_m)>0 else 1])
    y_range = the_intercept_X + the_slope_X * x_range
    ax.plot(x_range, y_range, color=COLOR_A, linestyle='--', linewidth=2, label='RLS fit')
    ax.legend()
    
    ax = axes[1]
    v2_corrected = (v2_m - the_intercept_X) / the_slope_X if the_slope_X != 0 else v2_m
    ax.scatter(v1_m, v2_corrected, s=10, c='black', alpha=0.6)
    ax.set_xlabel('FF (%)')
    ax.set_ylabel('FFX (%)')
    ax.set_xlim(0, mx)
    ax.plot([x_range[0], x_range[1]], [x_range[0], x_range[1]], color=COLOR_B, linestyle=':', linewidth=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_path, 'FFX.png'), dpi=300)
    plt.close()
    
    predictions_corrected = the_intercept + the_slope * predictions
    
    typer.echo("Executing final principal component analysis ...")
    
    pca_final = PCA(n_components=min(n_feat * 10, len(train_subset_df) - 1))
    pca_final.fit(train_subset_df)
    
    X_train_final = pca_final.transform(train_subset_df)[:, :n_feat]
    Y_train_final = y_all
    
    model_final: Union[LinearRegression, keras.Model]
    if olm:
        typer.echo("Training final ordinary linear model ...")
        model_final = LinearRegression()
        model_final.fit(X_train_final, Y_train_final)
    else:
        typer.echo("Training final neural network ...")
        model_final = train_neural_network(X_train_final, Y_train_final, hidden)
        
    info_overall = plot_performance(predictions_corrected, y_used, pca_final.explained_variance_ratio_, 
                                    n_feat, 'PREFACE (%)', 'FF (%)', os.path.join(out_dir_path, 'overall_performance.png'))
    
    deviations = np.abs(predictions_corrected - y_used)
    mae = info_overall[2]
    sd = info_overall[3]
    outlier_threshold = mae + 3 * sd
    outlier_indices = np.where(deviations > outlier_threshold)[0]
    
    with open(os.path.join(out_dir_path, 'training_statistics.txt'), 'w') as f:
        f.write('PREFACE - PREdict FetAl ComponEnt\n\n')
        
        if len(outlier_indices) > 0:
            f.write('Below, some of the top candidates for outlier removal are listed.\n')
            f.write('_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_- _\n')
            f.write('ID\tFF (%) - PREFACE (%)\n')
            
            sorted_outlier_idxs = outlier_indices[np.argsort(-deviations[outlier_indices])]
            subset_ids = config.loc[train_indices_all, 'ID'].values[:n_used]
            subset_diffs = (predictions_corrected - y_used)
            
            for idx in sorted_outlier_idxs:
                f.write(f"{subset_ids[idx]}\t{subset_diffs[idx]:.4f}\n")
            f.write('_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_- _\n\n')
            
        elapsed_time = time.time() - start_time
        f.write(f"Training time: {elapsed_time:.0f} seconds\n")
        f.write(f"Overall correlation (r): {info_overall[4]:.4f}\n")
        f.write(f"Overall mean absolute error (MAE): {info_overall[2]:.4f} ± {info_overall[3]:.4f}\n")
        
        mask_10 = y_used < 10.0
        if np.any(mask_10):
            devs_10 = deviations[mask_10]
            f.write(f"FF < 10% mean absolute error (MAE): {np.mean(devs_10):.4f} ± {np.std(devs_10, ddof=1):.4f}\n")
        
        f.write("Correction for skew: \n")
        f.write(f"\tIntercept: {the_intercept}\n")
        f.write(f"\tSlope: {the_slope}\n\n")

    model_data = {
        'n_feat': n_feat,
        'mean_features': mean_features,
        'possible_features': possible_features,
        'pca': pca_final,
        'is_olm': olm,
        'the_intercept': the_intercept,
        'the_slope': the_slope,
        'the_intercept_X': the_intercept_X,
        'the_slope_X': the_slope_X,
    }
    
    joblib.dump(model_data, os.path.join(out_dir_path, 'model_meta.pkl'))
    
    if olm:
        joblib.dump(model_final, os.path.join(out_dir_path, 'model_weights.pkl'))
    else:
        model_final.save(os.path.join(out_dir_path, 'model_weights.keras'))
        
    typer.echo(f"Finished! Consult '{out_dir_path}training_statistics.txt' to analyse your model's performance.")

