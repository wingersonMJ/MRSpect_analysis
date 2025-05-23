
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ttest_rel
from statsmodels.stats.weightstats import ttest_ind
from tableone import TableOne
print('done')


# data pre-processing 
file_path = 'path'
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

categorical_cols = ['physactivity_yn', 'sex', 'race', 'ethnicity', 'sports_yn', 'conc_hx_yn']
df[categorical_cols] = df[categorical_cols].astype('category')

df['fitbit'] = (df['steps'] > 0).astype(int).astype(str) # create fitbit variable


# Define MRS variable subsets, labels, etc.
mrs1 = ['GSH_Cr_PCr1', 'Glu_Gln_Cr_PCr1', 'PCh_GPC_Cr_PCr1', 'mI_Cr_PCr1']
mrs2 = ['GSH_Cr_PCr2', 'Glu_Gln_Cr_PCr2', 'PCh_GPC_Cr_PCr2', 'mI_Cr_PCr2']

mrs_pairs = {
    'GSH_Cr_PCr1': 'GSH_Cr_PCr2',
    'Glu_Gln_Cr_PCr1': 'Glu_Gln_Cr_PCr2',
    'PCh_GPC_Cr_PCr1': 'PCh_GPC_Cr_PCr2',
    'mI_Cr_PCr1': 'mI_Cr_PCr2'
}

mrs_labels1 = {
    'GSH_Cr_PCr1': 'Glutathione::Creatine',
    'Glu_Gln_Cr_PCr1': 'Glutamate-Glutamine::Creatine',
    'PCh_GPC_Cr_PCr1': 'Total Choline::Creatine',
    'mI_Cr_PCr1': 'Myo-Inositol::Creatine'
}

mrs_labels2 = {
    'GSH_Cr_PCr2': 'Glutathione::Creatine',
    'Glu_Gln_Cr_PCr2': 'Glutamate-Glutamine::Creatine',
    'PCh_GPC_Cr_PCr2': 'Total Choline::Creatine',
    'mI_Cr_PCr2': 'Myo-Inositol::Creatine'
}


# Aim 1: Comparing MRS at time 1 vs time 2 
"""
Analysis: Paired t-test & KDE plots for each MRS variable
"""

def paired_ttest(df, var1, var2):
    paired_data = df[[var1, var2]].dropna()
    time1 = paired_data[var1].values
    time2 = paired_data[var2].values
    t_stat, p_value = ttest_rel(time1, time2)
    mean_diff = (time2 - time1).mean()
    hedges_g = pg.compute_effsize(time1, time2, paired=True, eftype='hedges')
    results = {
        't_stat': t_stat,
        'p_value': p_value,
        'mean_diff': mean_diff,
        'hedges_g': hedges_g,
        'n': len(paired_data),
        'time1': time1,
        'time2': time2
    }
    return results

def plot_kde(ax, results, var1, label):
    time1, time2 = results['time1'], results['time2']
    sns.kdeplot(time1, label='Visit 1', fill=True, alpha=0.5, color='darkgrey', ax=ax, common_norm=False)
    sns.kdeplot(time2, label='Visit 2', fill=True, alpha=0.2, color='steelblue', ax=ax, common_norm=False)
    ax.axvline(time1.mean(), color='gray', linestyle='--', linewidth=1.5, label='Mean Visit 1')
    ax.axvline(time2.mean(), color='steelblue', linestyle='--', linewidth=1.5, label='Mean Visit 2')
    ax.set_xlabel(f'{label} (mM)', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.text(
        0.95, 0.95,
        f"$\\bf{{P~Value}}$: {results['p_value']:.3f}\n$\\bf{{Mean~Diff}}$: {results['mean_diff']:.3f}\n$\\bf{{Hedge's~G}}$: {results['hedges_g']:.3f}",
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=16,
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray')
    )

# KDE plots
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
axes = axes.flatten()
subplot_labels = ['A.', 'B.', 'C.', 'D.']

# Iterate through MRS pairs
for i, var1 in enumerate(mrs1):
    var2 = mrs_pairs[var1]
    results = paired_ttest(df, var1, var2)

    # summary stats
    print(f'Variable: {var1}')
    print(f"T-statistic: {results['t_stat']:.3f}")
    print(f"P-value: {results['p_value']:.3f}")
    print(f"Mean Diff: {results['mean_diff']:.3f}")
    print(f"Hedge's G: {results['hedges_g']:.3f}")
    print(f"Valid pairs: {results['n']}")
    print('')

    # plot
    ax = axes[i]
    plot_kde(ax, results, var1, mrs_labels1[var1])

    # add subplot labels
    leftsubplotdistance = -0.08 if i != 1 else -0.10
    ax.text(leftsubplotdistance, 1.00, subplot_labels[i],
            transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

    if i == 0:
        ax.legend(fontsize=12)

# Save
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, bottom=0.08, left=0.05, wspace=0.15)
plt.savefig(
    'path', 
    dpi=300, bbox_inches='tight'
)
plt.show()



# Aim 2: Physical activity and MRS at visit 1
"""
Analysis:
1. Linear regression for each MRS variable at Visit 1
   Outcome: MRS at visit 1
   Predictor: Physical activity (binary)
   Covariate: Time to visit 1
2. For simplicity: Independent Samples t-tests and Violin Plots
"""

def mrs1_model(df, var):
    formula = f"{var} ~ C(physactivity_yn) + doi_to_v1"
    model = sm.OLS.from_formula(formula, data=df).fit()
    print(f"Linear Regression for {var}")
    print(model.summary())
    
    # Residual Plot
    plt.figure()
    plt.scatter(model.fittedvalues, model.resid)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot: {var}')
    plt.show()

    # QQ Plot
    sm.qqplot(model.resid, line='45')
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.title(f'QQ Plot: {var}')
    plt.show()

    return model

def violinplot_with_stats(ax, df, var, label, subplot_label, i):
    sns.violinplot(
        x='physactivity_yn', y=var, data=df,
        alpha=0.5, palette=['silver', 'steelblue'], ax=ax
    )

    # axes
    ax.set_ylabel(f'{label} (mM)', fontweight='light', fontsize=13)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['No', 'Yes'], fontsize=14)
    if i in [2, 3]:
        ax.set_xlabel('Physically active before Visit 1', fontsize=16, fontweight='bold', labelpad=15)
    else:
        ax.set_xlabel("")

    # subplot labels
    ax.text(-0.11, 1.08, subplot_label, transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')

    # Independent Samples t-test
    group1 = df[df['physactivity_yn'] == 0][var].dropna()
    group2 = df[df['physactivity_yn'] == 1][var].dropna()
    t_stat, p_value, _ = ttest_ind(group1, group2, usevar='unequal')
    mean_diff = group2.mean() - group1.mean()
    hedges_g = pg.compute_effsize(group1, group2, paired=False, eftype='hedges')

    # results
    locx, locy = (0.70, 0.95) if subplot_label == 'D.' else (0.05, 0.95)
    ax.text(
        locx, locy,
        f"$\\bf{{P~Value}}$: {p_value:.3f}\n$\\bf{{Mean~Diff}}$: {mean_diff:.3f}\n$\\bf{{Hedge's~G}}$: {hedges_g:.3f}",
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.3, edgecolor='gray')
    )

# Linear models
for var in mrs1:
    mrs1_model(df, var)

# Violin plots with independent samples t-tests
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
subplot_labels = ['A.', 'B.', 'C.', 'D.']

for i, var in enumerate(mrs1):
    violinplot_with_stats(
        ax=axes[i],
        df=df,
        var=var,
        label=mrs_labels1[var],
        subplot_label=subplot_labels[i],
        i=i
    )

# Save
plt.tight_layout()
plt.subplots_adjust(hspace=0.25, bottom=0.08, left=0.05)
plt.savefig(
    'path',
    dpi=300, bbox_inches='tight'
)
plt.show()



# Aim 3: MRS at Visit 1 predicting steps 
"""
Analysis: Linear regression
Outcome: Steps/day
Predictor: MRS at visit 1
Covariate: PCSI at visit 1
"""

def steps_model(df, var):
    formula = f"steps ~ {var} + pcsi_total_current"
    model = sm.OLS.from_formula(formula, data=df).fit()
    print(f"Linear Regression for {var} predicting steps/day")
    print(model.summary())
    print(f'R-squared: {model.rsquared:.3f}\n')

    # Residual plot
    plt.figure()
    plt.scatter(model.fittedvalues, model.resid)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot: {var}')
    plt.show()

    # QQ Plot
    sm.qqplot(model.resid, line='45')
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.title(f'QQ Plot: {var}')
    plt.show()

    return model.rsquared

def plot_mrs_vs_steps(ax, df, var, label, subplot_label, r_sq, i):

    # scatter plot
    sns.regplot(
        x='steps', y=var, data=df,
        scatter_kws={'alpha': 0.7, 'color': 'silver'},
        line_kws={'color': 'steelblue', 'linewidth': 2}, 
        ax=ax
    )
    if i in [2, 3]:
        ax.set_xlabel('Mean steps per day', labelpad=15, fontweight='bold', fontsize=16)
    else:
        ax.set_xlabel("")
    ax.set_ylabel(f'{label} (mM)', fontweight='light', fontsize=13)
    # subplot labels
    leftsubplotdistance = -0.10 if i not in [1, 3] else -0.15
    ysubplotdistance = 1.05 if i not in [1, 3] else 1.08
    ax.text(
        leftsubplotdistance, ysubplotdistance, subplot_label,  
        transform=ax.transAxes, fontsize=18, fontweight='bold', 
        va='top', ha='left'
    )

    # r-squared
    ax.text(
        0.75, 0.90,  
        f"$R^2$ = {r_sq:.2f}",  
        transform=ax.transAxes, fontsize=12, fontweight='light',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray')
    )

# Linear models
r_squared_values = []
for var in mrs1:
    r_sq = steps_model(df, var)
    r_squared_values.append(r_sq)

# Scater plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
subplot_labels = ['A.', 'B.', 'C.', 'D.']

for i, var in enumerate(mrs1):
    plot_mrs_vs_steps(
        ax=axes[i],
        df=df,
        var=var,
        label=mrs_labels1[var],
        subplot_label=subplot_labels[i],
        r_sq=r_squared_values[i],
        i=i
    )

# Save
plt.tight_layout()
plt.subplots_adjust(hspace=0.20, bottom=0.08, left=0.05, wspace=0.20)
plt.savefig(
    'path',
    dpi=300, bbox_inches='tight'
)
plt.show()



# Aim 4: Steps predicting MRS at visit 2
"""
Analysis: Linear regression
Outcome: MRS at visit 2
Predictor: steps/day
Covariate: time to visit 2
"""

def mrs2_model(df, var):
    formula = f"{var} ~ steps + doi_to_v2"
    model = sm.OLS.from_formula(formula, data=df).fit()
    print(f"Linear Regression for steps predicting {var} (Visit 2)")
    print(model.summary())
    print(f'R-squared: {model.rsquared:.3f}\n')

    # Residual plot
    plt.figure()
    plt.scatter(model.fittedvalues, model.resid)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot: {var}')
    plt.show()

    # QQ Plot
    sm.qqplot(model.resid, line='45')
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.title(f'QQ Plot: {var}')
    plt.show()

    return model.rsquared

def plot_steps_vs_mrs2(ax, df, var, label, subplot_label, r_sq, i):

    # scatter plot
    sns.regplot(
        x='steps', y=var, data=df,
        scatter_kws={'alpha': 0.7, 'color': 'silver'},
        line_kws={'color': 'steelblue', 'linewidth': 2}, 
        ax=ax
    )
    ax.set_ylabel(f'{label} (mM)', fontweight='light', fontsize=14)
    if i in [2, 3]:
        ax.set_xlabel('Mean steps per day', labelpad=15, fontweight='bold', fontsize=16)
    else:
        ax.set_xlabel("")

    # Subplot labels
    ax.text(
        -0.12, 1.05, subplot_label,  
        transform=ax.transAxes, fontsize=18, fontweight='bold', 
        va='top', ha='left'
    )

    # r-squared
    ax.text(
        0.75, 0.90,  
        f"$R^2$ = {r_sq:.2f}",  
        transform=ax.transAxes, fontsize=12, fontweight='light',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray')
    )

# Steps predicting MRS at visit 2
r_squared_values = []
for var in mrs2:
    r_sq = mrs2_model(df, var)
    r_squared_values.append(r_sq)

# scatter plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
subplot_labels = ['A.', 'B.', 'C.', 'D.']

for i, var in enumerate(mrs2):
    plot_steps_vs_mrs2(
        ax=axes[i],
        df=df,
        var=var,
        label=mrs_labels2[var],
        subplot_label=subplot_labels[i],
        r_sq=r_squared_values[i],
        i=i
    )

# save
plt.tight_layout()
plt.subplots_adjust(hspace=0.20, bottom=0.08, left=0.05, wspace=0.20)
plt.savefig(
    'path',
    dpi=300, bbox_inches='tight'
)
plt.show()
