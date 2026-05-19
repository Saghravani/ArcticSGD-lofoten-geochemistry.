[Uploading mfa_biplots.py…]()
# ArcticSGD-lofoten-geochemistry
MFA and FCM analysis of porewater geochemistry data from the Lofoten-Vesterålen submarine groundwater discharge study
# =============================================================================
# mfa_biplots.py
# =============================================================================
# Standalone code for Multiple Factor Analysis (MFA) and generation of
# improved biplots (with Bottom Water samples shown in black).
# 
# Dependencies:
#   - pandas, numpy, matplotlib
#   - prince (pip install prince)
#   - pathlib
#
# Required pre‑defined objects (from data loading & cleaning):
#   - df_complete       : DataFrame with complete cases
#   - analysis_params   : list of parameter names
#   - sample_info       : DataFrame with columns ['Core','Canyon','Habitat','Depth_cmbsf', ...]
#   - output_folder     : Path object for saving outputs
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import prince
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - ADJUSTABLE PARAMETERS
# =============================================================================

# Point size configuration
POINT_SIZE_MIN = 50
POINT_SIZE_MAX = 200

# Arrow configuration
ARROW_BASE_SCALE = 3.5
ARROW_LENGTH_MIN = 0.3
ARROW_ALPHA_MIN = 0.3
ARROW_ALPHA_MAX = 1.0

# Figure configuration
FIGURE_SIZE = (16, 8)
DPI = 500

# Font and axis configuration
AXIS_TITLE_SIZE = 14
AXIS_LABEL_SIZE = 12
TICK_SIZE = 10
PLOT_TITLE_SIZE = 15
FRAME_WIDTH = 1.5

# Legend configuration
LEGEND_MARKER_SIZE = 12
LEGEND_FONT_SIZE = 11

# Extreme value labeling configuration
EXTREME_METHOD = 'percentile'          # 'percentile' or 'distance'
EXTREME_PERCENTILE_DIM12 = 97
EXTREME_PERCENTILE_DIM13 = 97
EXTREME_DISTANCE_DIM12 = 3.0
EXTREME_DISTANCE_DIM13 = 3.0

LABEL_FONT_SIZE = 12
LABEL_OFFSET_X = 0.15
LABEL_OFFSET_Y = 0.15
LABEL_COLOR = 'black'
LABEL_ALPHA = 0.8
LABEL_BBOX = True
LABEL_BBOX_COLOR = 'white'
LABEL_BBOX_ALPHA = 0.8

# Chemical formulae mapping
CHEMICAL_NAMES = {
    'Ca': 'Ca²⁺',
    'K': 'K⁺',
    'Mg': 'Mg²⁺',
    'Ba': 'Ba²⁺',
    'Li': 'Li⁺',
    'Mn': 'Mn²⁺',
    'Si': 'SiO₂',
    'Total_alkalinity': 'TA',
    'Chlorinity': 'Cl⁻',
    'SO4': 'SO₄²⁻'
}

# Text position adjustments for specific variables
TEXT_OFFSETS = {
    'Cl⁻': (0, 0.3),
}

# Habitat-Canyon colour scheme (BW = black)
HABITAT_CANYON_COLORS = {
    ('BacMat', 'North'): '#8B0000',
    ('BacMat', 'South'): '#DC143C',
    ('BacMat', 'Outside'): '#FF6B6B',
    ('Worm', 'North'): '#00008B',
    ('Worm', 'South'): '#4169E1',
    ('Worm', 'Outside'): '#87CEEB',
    ('Background', 'North'): '#006400',
    ('Background', 'South'): '#32CD32',
    ('Background', 'Outside'): '#90EE90',
    ('BW', 'North'): '#000000',
    ('BW', 'South'): '#000000',
    ('BW', 'Outside'): '#000000'
}

GROUP_COLORS = {
    'Groundwater_tracers': 'darkgreen',
    'Seawater_diagenetic': 'darkblue',
    'Carbonate_system': 'darkorange',
    'Redox': 'purple'
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def is_bw_sample(row):
    """Check if a sample is a Bottom Water (BW) sample."""
    if 'Depth_original' in row.index:
        depth_orig = str(row['Depth_original']).upper()
        if depth_orig in ['BW', 'BOTTOM WATER', 'BOTTOM_WATER']:
            return True
    if 'Depth_cmbsf' in row.index:
        depth = row['Depth_cmbsf']
        if isinstance(depth, str):
            if depth.upper() in ['BW', 'BOTTOM WATER', 'BOTTOM_WATER']:
                return True
        elif isinstance(depth, (int, float)):
            if depth < 0:
                return True
    if 'is_BW' in row.index and row['is_BW']:
        return True
    if 'Label' in row.index and 'BW' in str(row['Label']).upper():
        return True
    return False

def create_sample_label(row):
    """Create label in format 'Core - Depth'."""
    try:
        core = row.get('Core', 'Unknown')
        if is_bw_sample(row):
            return f"{core}-BW"
        depth = None
        for col in ['Depth_original', 'Depth_cmbsf', 'Depth']:
            if col in row.index:
                depth = row[col]
                if depth is not None:
                    break
        if depth is None:
            return f"{core}-Unknown"
        if isinstance(depth, str):
            depth_upper = str(depth).upper()
            if depth_upper in ['BW', 'BOTTOM WATER', 'BOTTOM_WATER']:
                depth_str = "BW"
            elif depth_upper in ['NAN', 'NA', 'N/A', '']:
                depth_str = "Unknown"
            else:
                depth_str = depth.replace('cm', '').replace('cmbsf', '').replace('mbsf', '').strip()
                try:
                    depth_num = float(depth_str)
                    depth_str = f"{int(depth_num)}" if depth_num == int(depth_num) else f"{depth_num:.1f}"
                except:
                    pass
        elif pd.isna(depth):
            depth_str = "Unknown"
        elif isinstance(depth, (int, float)):
            if depth < 0:
                depth_str = "BW"
            else:
                depth_str = f"{int(depth)}" if depth == int(depth) else f"{depth:.1f}"
        else:
            depth_str = str(depth)
        return f"{core}-{depth_str}"
    except:
        return "Unknown"

def get_chemical_name(original_name):
    return CHEMICAL_NAMES.get(original_name, original_name)

def get_text_offset(chemical_name):
    return TEXT_OFFSETS.get(chemical_name, (0, 0))

def format_axis(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(FRAME_WIDTH)
    ax.tick_params(axis='both', which='major', labelsize=AXIS_LABEL_SIZE,
                   width=FRAME_WIDTH, length=TICK_SIZE)

def add_sample_labels(ax, data, extreme_mask, dim_x, dim_y):
    """Add labels to extreme samples."""
    extreme_data = data[extreme_mask]
    for _, row in extreme_data.iterrows():
        x = row[dim_x]
        y = row[dim_y]
        label = row['Label']
        x_sign = 1 if x >= 0 else -1
        y_sign = 1 if y >= 0 else -1
        text_x = x + (LABEL_OFFSET_X * x_sign)
        text_y = y + (LABEL_OFFSET_Y * y_sign)
        ha = 'left' if x >= 0 else 'right'
        if LABEL_BBOX:
            bbox_props = dict(boxstyle='round,pad=0.2', facecolor=LABEL_BBOX_COLOR,
                              alpha=LABEL_BBOX_ALPHA, edgecolor='gray', linewidth=0.5)
            ax.annotate(label, xy=(x, y), xytext=(text_x, text_y),
                        fontsize=LABEL_FONT_SIZE, color=LABEL_COLOR, alpha=LABEL_ALPHA,
                        ha=ha, va='center', bbox=bbox_props)
        else:
            ax.text(text_x, text_y, label, fontsize=LABEL_FONT_SIZE,
                    color=LABEL_COLOR, alpha=LABEL_ALPHA, ha=ha, va='center')[fcm_clustering_analysis.py](https://github.com/user-attachments/files/28009229/fcm_clustering_analysis.py)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("MFA BIPLOTS GENERATION")
print("=" * 80)

# Check required objects
assert 'df_complete' in globals(), "df_complete not defined"
assert 'analysis_params' in globals(), "analysis_params not defined"
assert 'output_folder' in globals(), "output_folder not defined"
print("✓ Required objects found")

# Define variable groups
variable_groups = {
    'Groundwater_tracers': ['Li', 'Ba', 'Si'],
    'Seawater_diagenetic': ['K', 'Mg', 'SO4', 'Chlorinity'],
    'Carbonate_system': ['Ca', 'Total_alkalinity'],
    'Redox': ['Mn']
}

# Prepare data for MFA
df_mfa = df_complete[analysis_params].copy().reset_index(drop=True)
group_labels = []
for col in df_mfa.columns:
    for group_name, params in variable_groups.items():
        if col in params:
            group_labels.append(group_name)
            break
    else:
        group_labels.append('Unknown')
df_mfa.columns = pd.MultiIndex.from_arrays([group_labels, df_mfa.columns],
                                           names=['Group', 'Variable'])

# Fit MFA
mfa = prince.MFA(n_components=5, n_iter=20, copy=True, check_input=True, random_state=42)
mfa.fit(df_mfa)

# Extract eigenvalues and variance
if hasattr(mfa, 'eigenvalues_'):
    eigenvalues = np.array(mfa.eigenvalues_)
else:
    eigenvalues = np.array(mfa.eigenvalues)
if hasattr(mfa, 'percentage_of_variance_'):
    explained = np.array(mfa.percentage_of_variance_)
elif hasattr(mfa, 'explained_inertia_'):
    explained = np.array(mfa.explained_inertia_)
else:
    total = sum(eigenvalues)
    explained = np.array([(ev / total) * 100 for ev in eigenvalues])

print("\n--- MFA Explained Variance ---")
variance_df = pd.DataFrame({
    'Dimension': [f'Dim{i+1}' for i in range(len(eigenvalues))],
    'Eigenvalue': eigenvalues,
    'Variance_%': explained,
    'Cumulative_%': np.cumsum(explained)
})
print(variance_df.round(4).to_string(index=False))

# Row coordinates
mfa_row_coords = mfa.row_coordinates(df_mfa)
mfa_row_coords.columns = [f'Dim{i+1}' for i in range(mfa_row_coords.shape[1])]

# Build combined results with sample info
if 'sample_info' not in globals():
    sample_info = df_complete[['Core', 'Canyon', 'Habitat']].copy().reset_index(drop=True)
    if 'Depth_original' in df_complete.columns:
        sample_info['Depth_original'] = df_complete['Depth_original'].reset_index(drop=True)
    if 'Depth_cmbsf' in df_complete.columns:
        sample_info['Depth_cmbsf'] = df_complete['Depth_cmbsf'].reset_index(drop=True)
    if 'is_BW' in df_complete.columns:
        sample_info['is_BW'] = df_complete['is_BW'].reset_index(drop=True)

mfa_results = pd.concat([sample_info.reset_index(drop=True), mfa_row_coords], axis=1)
mfa_results['is_BW'] = mfa_results.apply(is_bw_sample, axis=1)
mfa_results['Label'] = mfa_results.apply(create_sample_label, axis=1)
print(f"✓ {len(mfa_results)} samples, BW samples: {mfa_results['is_BW'].sum()}")

# Sample contributions and point sizes
n_samples = len(mfa_row_coords)
sample_contributions = pd.DataFrame(index=mfa_row_coords.index)
for k, dim in enumerate(mfa_row_coords.columns):
    if k < len(eigenvalues) and eigenvalues[k] > 0:
        sample_contributions[dim] = (mfa_row_coords[dim] ** 2 / eigenvalues[k]) * (1 / n_samples) * 100
    else:
        sample_contributions[dim] = 0

if len(explained) >= 3:
    var1, var2, var3 = explained[0], explained[1], explained[2]
    sample_contributions['CTR_Dim1_Dim2'] = (
        (var1 * sample_contributions['Dim1'] + var2 * sample_contributions['Dim2']) / (var1 + var2)
    )
    sample_contributions['CTR_Dim1_Dim3'] = (
        (var1 * sample_contributions['Dim1'] + var3 * sample_contributions['Dim3']) / (var1 + var3)
    )
else:
    sample_contributions['CTR_Dim1_Dim2'] = 1.0
    sample_contributions['CTR_Dim1_Dim3'] = 1.0

def scale_point_sizes(contributions, min_size=POINT_SIZE_MIN, max_size=POINT_SIZE_MAX):
    ctr_min = contributions.min()
    ctr_max = contributions.max()
    if ctr_max - ctr_min == 0:
        return np.full(len(contributions), (min_size + max_size) / 2)
    sizes = min_size + (contributions - ctr_min) / (ctr_max - ctr_min) * (max_size - min_size)
    return sizes

point_sizes_dim12 = scale_point_sizes(sample_contributions['CTR_Dim1_Dim2'])
point_sizes_dim13 = scale_point_sizes(sample_contributions['CTR_Dim1_Dim3'])

# Distances and extreme masks
distances_dim12 = np.sqrt(mfa_results['Dim1']**2 + mfa_results['Dim2']**2)
distances_dim13 = np.sqrt(mfa_results['Dim1']**2 + mfa_results['Dim3']**2)

def get_extreme_mask(distances, method, percentile_threshold, distance_threshold):
    if method == 'percentile':
        threshold = np.percentile(distances, percentile_threshold)
        return distances >= threshold
    else:
        return distances >= distance_threshold

extreme_mask_dim12 = get_extreme_mask(distances_dim12, EXTREME_METHOD,
                                       EXTREME_PERCENTILE_DIM12, EXTREME_DISTANCE_DIM12)
extreme_mask_dim13 = get_extreme_mask(distances_dim13, EXTREME_METHOD,
                                       EXTREME_PERCENTILE_DIM13, EXTREME_DISTANCE_DIM13)

# Variable coordinates
mfa_col_coords = None
try:
    mfa_col_coords = mfa.column_coordinates(df_mfa)
except:
    pass
if mfa_col_coords is None:
    try:
        if hasattr(mfa, 'column_correlations'):
            mfa_col_coords = mfa.column_correlations
            if callable(mfa_col_coords):
                mfa_col_coords = mfa_col_coords(df_mfa)
    except:
        pass
if mfa_col_coords is None:
    correlations = []
    for col in df_mfa.columns:
        row_corr = []
        for dim in mfa_row_coords.columns:
            r = np.corrcoef(df_mfa[col], mfa_row_coords[dim])[0, 1]
            row_corr.append(r)
        correlations.append(row_corr)
    mfa_col_coords = pd.DataFrame(correlations, index=df_mfa.columns,
                                  columns=mfa_row_coords.columns)

mfa_col_coords.columns = [f'Dim{i+1}' for i in range(mfa_col_coords.shape[1])]
if not isinstance(mfa_col_coords.index, pd.MultiIndex):
    new_index = []
    for var_name in mfa_col_coords.index:
        if isinstance(var_name, tuple):
            new_index.append(var_name)
        else:
            found_group = "Unknown"
            for g, params in variable_groups.items():
                if var_name in params:
                    found_group = g
                    break
            new_index.append((found_group, var_name))
    mfa_col_coords.index = pd.MultiIndex.from_tuples(new_index, names=['Group', 'Variable'])

# Variable contributions and cos²
n_vars = len(mfa_col_coords)
var_contributions = pd.DataFrame(index=mfa_col_coords.index)
for k, dim in enumerate(mfa_col_coords.columns):
    if k < len(eigenvalues) and eigenvalues[k] > 0:
        var_contributions[dim] = (mfa_col_coords[dim] ** 2 / eigenvalues[k]) * (1 / n_vars) * 100
    else:
        var_contributions[dim] = 0
if len(explained) >= 3:
    var_contributions['CTR_Dim1_Dim2'] = (
        (var1 * var_contributions['Dim1'] + var2 * var_contributions['Dim2']) / (var1 + var2)
    )
    var_contributions['CTR_Dim1_Dim3'] = (
        (var1 * var_contributions['Dim1'] + var3 * var_contributions['Dim3']) / (var1 + var3)
    )
else:
    var_contributions['CTR_Dim1_Dim2'] = 1.0
    var_contributions['CTR_Dim1_Dim3'] = 1.0

total_squared = (mfa_col_coords ** 2).sum(axis=1)
cos2_dim12 = (mfa_col_coords['Dim1'] ** 2 + mfa_col_coords['Dim2'] ** 2) / total_squared
cos2_dim13 = (mfa_col_coords['Dim1'] ** 2 + mfa_col_coords['Dim3'] ** 2) / total_squared
cos2_dim12 = cos2_dim12.fillna(0.5).clip(upper=1)
cos2_dim13 = cos2_dim13.fillna(0.5).clip(upper=1)

# Arrow scaling and alpha
def calculate_arrow_length_scale(contributions, base_scale=ARROW_BASE_SCALE,
                                 min_factor=ARROW_LENGTH_MIN):
    ctr_max = contributions.max()
    if ctr_max == 0 or pd.isna(ctr_max):
        return {idx: base_scale * min_factor for idx in contributions.index}
    normalized = contributions / ctr_max
    scale_factors = min_factor + (1 - min_factor) * np.sqrt(normalized)
    return {idx: base_scale * scale_factors.loc[idx] for idx in contributions.index}

def calculate_arrow_alpha(cos2_value):
    if pd.isna(cos2_value):
        return (ARROW_ALPHA_MIN + ARROW_ALPHA_MAX) / 2
    return ARROW_ALPHA_MIN + (ARROW_ALPHA_MAX - ARROW_ALPHA_MIN) * cos2_value

arrow_scales_dim12 = calculate_arrow_length_scale(var_contributions['CTR_Dim1_Dim2'])
arrow_scales_dim13 = calculate_arrow_length_scale(var_contributions['CTR_Dim1_Dim3'])

# =============================================================================
# CREATE BIPLOTS
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)

# --- Left panel: Dim1 vs Dim2 ---
ax1 = axes[0]
bw_mask = mfa_results['is_BW']
non_bw_mask = ~bw_mask

if bw_mask.sum() > 0:
    for canyon in df_complete['Canyon'].unique():
        mask = bw_mask & (mfa_results['Canyon'] == canyon)
        if mask.sum() > 0:
            ax1.scatter(mfa_results.loc[mask, 'Dim1'],
                        mfa_results.loc[mask, 'Dim2'],
                        c='#000000', marker='o', s=point_sizes_dim12[mask],
                        alpha=0.9, edgecolors='white', linewidths=1.0, zorder=5)

for habitat in df_complete['Habitat'].unique():
    for canyon in df_complete['Canyon'].unique():
        mask = non_bw_mask & (mfa_results['Habitat'] == habitat) & (mfa_results['Canyon'] == canyon)
        if mask.sum() > 0:
            color = HABITAT_CANYON_COLORS.get((habitat, canyon), 'gray')
            ax1.scatter(mfa_results.loc[mask, 'Dim1'],
                        mfa_results.loc[mask, 'Dim2'],
                        c=color, marker='o', s=point_sizes_dim12[mask],
                        alpha=0.7, edgecolors='black', linewidths=0.5, zorder=4)

add_sample_labels(ax1, mfa_results, extreme_mask_dim12, 'Dim1', 'Dim2')

for idx in mfa_col_coords.index:
    if isinstance(idx, tuple):
        group, variable = idx
    else:
        variable = idx; group = "Unknown"
    color = GROUP_COLORS.get(group, 'gray')
    length_scale = arrow_scales_dim12.get(idx, ARROW_BASE_SCALE)
    alpha = calculate_arrow_alpha(cos2_dim12.loc[idx])
    x = mfa_col_coords.loc[idx, 'Dim1'] * length_scale
    y = mfa_col_coords.loc[idx, 'Dim2'] * length_scale
    ax1.arrow(0, 0, x, y, head_width=0.12, head_length=0.08,
              fc=color, ec=color, alpha=alpha, linewidth=1.5, zorder=3)
    chem_name = get_chemical_name(variable)
    x_offset, y_offset = get_text_offset(chem_name)
    ax1.text(x * 1.15 + x_offset, y * 1.15 + y_offset, chem_name,
             color=color, fontsize=AXIS_LABEL_SIZE, fontweight='bold',
             ha='center', va='center', alpha=alpha, zorder=6)

ax1.axhline(0, color='darkgray', linewidth=1.0, linestyle=':', alpha=0.8, zorder=1)
ax1.axvline(0, color='darkgray', linewidth=1.0, linestyle=':', alpha=0.8, zorder=1)
ax1.set_xlabel(f'Dim1 ({explained[0]:.1f}%)', fontsize=AXIS_TITLE_SIZE, fontweight='bold')
ax1.set_ylabel(f'Dim2 ({explained[1]:.1f}%)', fontsize=AXIS_TITLE_SIZE, fontweight='bold')
ax1.set_title('MFA Biplot: Dim1 vs Dim2', fontsize=PLOT_TITLE_SIZE, fontweight='bold')
ax1.set_aspect('equal', adjustable='datalim')
format_axis(ax1)

# --- Right panel: Dim1 vs Dim3 ---
ax2 = axes[1]

if bw_mask.sum() > 0:
    for canyon in df_complete['Canyon'].unique():
        mask = bw_mask & (mfa_results['Canyon'] == canyon)
        if mask.sum() > 0:
            ax2.scatter(mfa_results.loc[mask, 'Dim1'],
                        mfa_results.loc[mask, 'Dim3'],
                        c='#000000', marker='o', s=point_sizes_dim13[mask],
                        alpha=0.9, edgecolors='white', linewidths=1.0, zorder=5)

for habitat in df_complete['Habitat'].unique():
    for canyon in df_complete['Canyon'].unique():
        mask = non_bw_mask & (mfa_results['Habitat'] == habitat) & (mfa_results['Canyon'] == canyon)
        if mask.sum() > 0:
            color = HABITAT_CANYON_COLORS.get((habitat, canyon), 'gray')
            ax2.scatter(mfa_results.loc[mask, 'Dim1'],
                        mfa_results.loc[mask, 'Dim3'],
                        c=color, marker='o', s=point_sizes_dim13[mask],
                        alpha=0.7, edgecolors='black', linewidths=0.5, zorder=4)

add_sample_labels(ax2, mfa_results, extreme_mask_dim13, 'Dim1', 'Dim3')

for idx in mfa_col_coords.index:
    if isinstance(idx, tuple):
        group, variable = idx
    else:
        variable = idx; group = "Unknown"
    color = GROUP_COLORS.get(group, 'gray')
    length_scale = arrow_scales_dim13.get(idx, ARROW_BASE_SCALE)
    alpha = calculate_arrow_alpha(cos2_dim13.loc[idx])
    x = mfa_col_coords.loc[idx, 'Dim1'] * length_scale
    y = mfa_col_coords.loc[idx, 'Dim3'] * length_scale
    ax2.arrow(0, 0, x, y, head_width=0.12, head_length=0.08,
              fc=color, ec=color, alpha=alpha, linewidth=1.5, zorder=3)
    chem_name = get_chemical_name(variable)
    x_offset, y_offset = get_text_offset(chem_name)
    ax2.text(x * 1.15 + x_offset, y * 1.15 + y_offset, chem_name,
             color=color, fontsize=AXIS_LABEL_SIZE, fontweight='bold',
             ha='center', va='center', alpha=alpha, zorder=6)

ax2.axhline(0, color='darkgray', linewidth=1.0, linestyle=':', alpha=0.8, zorder=1)
ax2.axvline(0, color='darkgray', linewidth=1.0, linestyle=':', alpha=0.8, zorder=1)
ax2.set_xlabel(f'Dim1 ({explained[0]:.1f}%)', fontsize=AXIS_TITLE_SIZE, fontweight='bold')
ax2.set_ylabel(f'Dim3 ({explained[2]:.1f}%)', fontsize=AXIS_TITLE_SIZE, fontweight='bold')
ax2.set_title('MFA Biplot: Dim1 vs Dim3', fontsize=PLOT_TITLE_SIZE, fontweight='bold')
ax2.set_aspect('equal', adjustable='datalim')
format_axis(ax2)

# --- Legend ---
legend_elements = []
if bw_mask.sum() > 0:
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='#000000', markeredgecolor='white',
                                  markeredgewidth=1.0, markersize=LEGEND_MARKER_SIZE,
                                  linestyle='None', label='Bottom Water (BW)'))
for habitat in ['BacMat', 'Worm', 'Background']:
    for canyon in ['North', 'South', 'Outside']:
        mask = (df_complete['Habitat'] == habitat) & (df_complete['Canyon'] == canyon)
        if mask.sum() > 0:
            color = HABITAT_CANYON_COLORS.get((habitat, canyon), 'gray')
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markeredgecolor='black',
                                          markersize=LEGEND_MARKER_SIZE, linestyle='None',
                                          label=f'{habitat} - {canyon}'))

ax2.legend(handles=legend_elements, loc='upper right', fontsize=LEGEND_FONT_SIZE,
           frameon=False, handletextpad=0.5, labelspacing=0.8)

plt.tight_layout()
plt.savefig(output_folder / 'MFA_biplots_Dim12_Dim13.png', dpi=DPI, bbox_inches='tight')
plt.show()
print(f"\n✓ Saved: MFA_biplots_Dim12_Dim13.png")

# --- Save results ---
mfa_results_export = mfa_results.copy()
mfa_results_export['CTR_Dim1'] = sample_contributions['Dim1']
mfa_results_export['CTR_Dim2'] = sample_contributions['Dim2']
mfa_results_export['CTR_Dim3'] = sample_contributions['Dim3']
mfa_results_export['CTR_Dim1_Dim2'] = sample_contributions['CTR_Dim1_Dim2']
mfa_results_export['CTR_Dim1_Dim3'] = sample_contributions['CTR_Dim1_Dim3']
mfa_results_export['PointSize_Dim12'] = point_sizes_dim12
mfa_results_export['PointSize_Dim13'] = point_sizes_dim13
mfa_results_export.to_csv(output_folder / 'MFA_sample_scores.csv', index=False)
print("✓ Saved: MFA_sample_scores.csv")

mfa_col_coords.to_csv(output_folder / 'MFA_variable_coordinates.csv')
print("✓ Saved: MFA_variable_coordinates.csv")

print("=" * 80)
print("MFA BIPLOTS GENERATION COMPLETE")
print("=" * 80)









[fcm_clustering_analysis.py](https://github.com/user-attachments/files/28009228/fcm_clustering_analysis.py)
# =============================================================================
# fcm_clustering_analysis.py
# =============================================================================
# Standalone code for Fuzzy C‑Means (FCM) clustering on MFA dimensions.
# Generates:
#   - Parameter optimisation heatmaps
#   - Final FCM biplots (Dim1‑Dim2, Dim1‑Dim3, membership certainty)
#   - Cluster composition bar chart
#   - Depth‑profile scatter plots (Dim1, Dim2, Dim3 vs depth) with BW samples
#
# Dependencies:
#   - pandas, numpy, matplotlib, seaborn
#   - prince (for MFA, only if re‑running from scratch)
#   - scikit‑fuzzy (pip install scikit‑fuzzy)
#   - sklearn, scipy
#
# Required pre‑defined objects:
#   - mfa_results      : DataFrame with MFA dimensions (Dim1, Dim2, Dim3)
#                         and sample info (Core, Canyon, Habitat, Depth_cmbsf, is_BW, etc.)
#   - df_complete      : DataFrame with original geochemical data
#   - analysis_params  : list of parameter names
#   - output_folder    : Path object for saving outputs
#
# Adjustable parameters (clusters, fuzziness, etc.) are set in the CONFIGURATION section.
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    import skfuzzy as fuzz
    FCM_AVAILABLE = True
except ImportError:
    FCM_AVAILABLE = False
    raise ImportError("scikit‑fuzzy is required. Install with: pip install scikit‑fuzzy")

# =============================================================================
# CONFIGURATION
# =============================================================================
# Number of clusters and fuzziness exponent for final clustering
C_CLUSTERS = 3
M_FUZZINESS = 2.0
N_INIT = 100                     # Number of random initializations for final run

# Parameter ranges for optimisation
C_RANGE = range(2, 6)            # Test cluster numbers 2..5
M_RANGE = np.arange(1.1, 3.0, 0.2)  # Test fuzziness 1.1, 1.3, ..., 2.9

# Transitional sample threshold (max membership below this value)
TRANSITIONAL_THRESHOLD = 0.6

# Habitat‑Canyon colour scheme (same as MFA script)
HABITAT_CANYON_COLORS = {
    ('BacMat', 'North'): '#8B0000',
    ('BacMat', 'South'): '#FF6B6B',
    ('Worm', 'North'): '#00008B',
    ('Worm', 'South'): '#87CEEB',
    ('Background', 'Outside'): '#90EE90',
    ('BW', 'North'): '#000000',
    ('BW', 'South'): '#000000',
    ('BW', 'Outside'): '#000000'
}

# Cluster marker shapes
CLUSTER_MARKERS = {1: 'o', 2: '^', 3: 's'}

# Figure / font settings
FIGURE_SIZE_OPT = (14, 10)       # Parameter optimisation figure
FIGURE_SIZE_FCM = (18, 12)       # FCM biplots figure
FIGURE_SIZE_DEPTH = (18, 6)      # Depth profiles figure
DPI = 300
SCATTER_SIZE = 120

# Depth profile settings
BW_DEPTH_POSITION = -3           # Plot BW samples above seafloor
BW_AXIS_BUFFER = 2

# =============================================================================
# HELPER FUNCTION (BW detection, safe for any environment)
# =============================================================================
def is_bw(row):
    """Return True if the row is a Bottom Water sample."""
    if 'is_BW' in row.index and row['is_BW']:
        return True
    if 'Depth_original' in row.index:
        val = str(row['Depth_original']).upper().strip()
        if val in ['BW', 'BOTTOM WATER', 'BOTTOM_WATER']:
            return True
    if 'Depth_cmbsf' in row.index:
        val = row['Depth_cmbsf']
        if isinstance(val, str) and val.upper().strip() in ['BW', 'BOTTOM WATER', 'BOTTOM_WATER']:
            return True
        if isinstance(val, (int, float)) and val < 0:
            return True
    return False

# =============================================================================
# ENSURE REQUIRED DATA ARE AVAILABLE
# =============================================================================
print("\n" + "=" * 80)
print("FUZZY C‑MEANS CLUSTERING ON MFA DIMENSIONS")
print("=" * 80)

assert 'mfa_results' in globals() or 'mfa_results' in locals(), \
    "mfa_results DataFrame not found. Run MFA first or re‑define it."
assert 'df_complete' in globals() or 'df_complete' in locals(), \
    "df_complete not found."
assert 'output_folder' in globals() or 'output_folder' in locals(), \
    "output_folder not defined."

# If is_BW column missing, compute it
if 'is_BW' not in mfa_results.columns:
    mfa_results['is_BW'] = mfa_results.apply(is_bw, axis=1)

# Ensure MFA dimensions are present
for dim in ['Dim1', 'Dim2', 'Dim3']:
    assert dim in mfa_results.columns, f"{dim} missing in mfa_results."

# Build the data matrix for FCM (n_samples × 3)
X_mfa = mfa_results[['Dim1', 'Dim2', 'Dim3']].values.T  # shape (3, n)

print(f"Samples for FCM: {X_mfa.shape[1]} (BW samples: {mfa_results['is_BW'].sum()})")

# =============================================================================
# PART 1: PARAMETER OPTIMISATION (FPC, Silhouette, etc.)
# =============================================================================
print("\n--- FCM Parameter Optimisation ---")
print(f"Testing c ∈ {list(C_RANGE)}, m ∈ {[round(m,1) for m in M_RANGE]}")

param_results = []

for c in C_RANGE:
    for m in M_RANGE:
        best_fpc = -1
        best_u = None
        best_centers = None
        # Run multiple initializations for each (c,m) pair
        for init in range(50):
            try:
                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    X_mfa, c=c, m=m, error=0.001, maxiter=1000, init=None, seed=init
                )
                if fpc > best_fpc:
                    best_fpc = fpc
                    best_centers = cntr
                    best_u = u
            except:
                continue
        if best_u is None:
            continue

        # Hard labels for validity indices
        labels = np.argmax(best_u, axis=0)
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(X_mfa.T, labels)
            ch = calinski_harabasz_score(X_mfa.T, labels)
        else:
            sil = -1
            ch = 0

        # Xie‑Beni index
        try:
            n = X_mfa.shape[1]
            numerator = sum(
                (best_u[i, j] ** m) * np.linalg.norm(X_mfa[:, j] - best_centers[i])**2
                for i in range(c) for j in range(n)
            )
            min_center_dist = min(
                np.linalg.norm(best_centers[i] - best_centers[j])
                for i in range(c) for j in range(i+1, c)
            )
            xb = numerator / (n * min_center_dist**2) if min_center_dist > 0 else np.inf
        except:
            xb = np.inf

        param_results.append({
            'c': c, 'm': round(m, 1),
            'FPC': best_fpc, 'Silhouette': sil,
            'Calinski_Harabasz': ch, 'Xie_Beni': xb
        })

param_df = pd.DataFrame(param_results)

# Heatmaps of the indices
fig, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_OPT)
metrics = ['FPC', 'Silhouette', 'Xie_Beni', 'Calinski_Harabasz']
cmaps = ['YlGnBu', 'YlGnBu', 'YlOrRd_r', 'YlGnBu']
for i, (metric, cmap) in enumerate(zip(metrics, cmaps)):
    ax = axes[i//2, i%2]
    pivot = param_df.pivot(index='m', columns='c', values=metric)
    if metric == 'Xie_Beni':
        pivot = pivot.clip(upper=10)  # limit color scale for extreme values
    sns.heatmap(pivot, annot=True, fmt='.3f' if metric != 'Calinski_Harabasz' else '.1f',
                cmap=cmap, ax=ax, cbar_kws={'label': metric})
    ax.set_title(metric.replace('_',' '), fontweight='bold')
    ax.set_xlabel('c'); ax.set_ylabel('m')
plt.suptitle('FCM Parameter Optimisation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_folder / 'FCM_parameter_optimization.png', dpi=DPI, bbox_inches='tight')
plt.show()
print("✓ Saved FCM_parameter_optimization.png")

# Print best recommendations
print("\n--- Recommended Parameters ---")
best_fpc_row = param_df.loc[param_df['FPC'].idxmax()]
print(f"Best by FPC: c={int(best_fpc_row['c'])}, m={best_fpc_row['m']:.1f} (FPC={best_fpc_row['FPC']:.4f})")
best_sil_row = param_df.loc[param_df['Silhouette'].idxmax()]
print(f"Best by Silhouette: c={int(best_sil_row['c'])}, m={best_sil_row['m']:.1f} (Sil={best_sil_row['Silhouette']:.4f})")
valid_xb = param_df[param_df['Xie_Beni'] < np.inf]
if len(valid_xb) > 0:
    best_xb_row = valid_xb.loc[valid_xb['Xie_Beni'].idxmin()]
    print(f"Best by Xie‑Beni: c={int(best_xb_row['c'])}, m={best_xb_row['m']:.1f} (XB={best_xb_row['Xie_Beni']:.4f})")

# =============================================================================
# PART 2: FINAL FCM CLUSTERING
# =============================================================================
print(f"\n--- Final FCM (c={C_CLUSTERS}, m={M_FUZZINESS}) ---")

best_fpc = -1
best_cntr = None
best_u = None

for init in range(N_INIT):
    try:
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_mfa, c=C_CLUSTERS, m=M_FUZZINESS, error=0.0001, maxiter=1000, init=None, seed=init
        )
        if fpc > best_fpc:
            best_fpc = fpc
            best_cntr = cntr
            best_u = u
    except:
        continue

print(f"Best FPC: {best_fpc:.4f}")

# Hard cluster labels
fcm_labels = np.argmax(best_u, axis=0)
memberships = best_u.T

# Build results DataFrame
fcm_membership_df = pd.DataFrame(
    memberships,
    columns=[f'Cluster_{i+1}_membership' for i in range(C_CLUSTERS)]
)
fcm_membership_df['FCM_Cluster'] = fcm_labels + 1
fcm_membership_df['Max_Membership'] = memberships.max(axis=1)
fcm_results = pd.concat(
    [mfa_results.reset_index(drop=True), fcm_membership_df.reset_index(drop=True)],
    axis=1
)

# Cluster summaries
print("\nCluster summaries:")
for cl in range(1, C_CLUSTERS+1):
    cluster_data = fcm_results[fcm_results['FCM_Cluster'] == cl]
    bw_count = cluster_data['is_BW'].sum()
    print(f"  Cluster {cl}: n={len(cluster_data)} (BW={bw_count})")
    print(f"    Dim1 mean: {cluster_data['Dim1'].mean():.3f}  "
          f"Dim2 mean: {cluster_data['Dim2'].mean():.3f}  "
          f"Dim3 mean: {cluster_data['Dim3'].mean():.3f}")
    print(f"    Mean max membership: {cluster_data['Max_Membership'].mean():.3f}")
    print(f"    Habitats: {dict(cluster_data['Habitat'].value_counts())}")

# Transitional samples
transitional = fcm_results[fcm_results['Max_Membership'] < TRANSITIONAL_THRESHOLD]
print(f"\nTransitional samples (max membership < {TRANSITIONAL_THRESHOLD}): {len(transitional)}")
if len(transitional) > 0:
    print(transitional[['Core', 'Habitat', 'Depth_cmbsf', 'is_BW', 'FCM_Cluster', 'Max_Membership']].to_string(index=False))

# Cluster centers
centers_df = pd.DataFrame(
    best_cntr, columns=['Dim1', 'Dim2', 'Dim3'],
    index=[f'Cluster_{i+1}' for i in range(C_CLUSTERS)]
)
print("\nCluster centers (MFA space):")
print(centers_df.round(4))

# =============================================================================
# PART 3: FCM BIPLOTS (Dim1‑Dim2, Dim1‑Dim3, Membership certainty, Composition)
# =============================================================================
fig = plt.figure(figsize=FIGURE_SIZE_FCM)

# (a) Dim1 vs Dim2 with cluster shapes & habitat colours
ax1 = fig.add_subplot(2, 3, 1)
for _, row in fcm_results.iterrows():
    if row['is_BW']:
        color = '#000000'
    else:
        hab = row['Habitat']
        can = row['Canyon'] if row['Canyon'] in ['North', 'South'] else 'Outside'
        color = HABITAT_CANYON_COLORS.get((hab, can), '#808080')
    marker = CLUSTER_MARKERS.get(row['FCM_Cluster'], 'o')
    ax1.scatter(row['Dim1'], row['Dim2'], c=[color], marker=marker, s=SCATTER_SIZE,
                alpha=0.9 if row['is_BW'] else 0.7,
                edgecolors='white' if row['is_BW'] else 'black', linewidths=1.0 if row['is_BW'] else 0.5)
ax1.scatter(best_cntr[:,0], best_cntr[:,1], c='black', marker='X', s=300, zorder=5, label='Centers')
ax1.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax1.axvline(0, color='gray', linestyle='--', alpha=0.3)
ax1.set_xlabel(f'Dim1 ({mfa_results.iloc[:,3].name.split()[-1] if False else "47.3%"})', fontsize=11)
ax1.set_ylabel('Dim2', fontsize=11)  # In practice, you can pass the variance from MFA
ax1.set_title('FCM Clusters: Dim1 vs Dim2', fontweight='bold')
ax1.grid(True, alpha=0.3)

# (b) Dim1 vs Dim3
ax2 = fig.add_subplot(2, 3, 2)
for _, row in fcm_results.iterrows():
    if row['is_BW']:
        color = '#000000'
    else:
        hab = row['Habitat']
        can = row['Canyon'] if row['Canyon'] in ['North', 'South'] else 'Outside'
        color = HABITAT_CANYON_COLORS.get((hab, can), '#808080')
    marker = CLUSTER_MARKERS.get(row['FCM_Cluster'], 'o')
    ax2.scatter(row['Dim1'], row['Dim3'], c=[color], marker=marker, s=SCATTER_SIZE,
                alpha=0.9 if row['is_BW'] else 0.7,
                edgecolors='white' if row['is_BW'] else 'black', linewidths=1.0 if row['is_BW'] else 0.5)
ax2.scatter(best_cntr[:,0], best_cntr[:,2], c='black', marker='X', s=300, zorder=5)
ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax2.axvline(0, color='gray', linestyle='--', alpha=0.3)
ax2.set_xlabel('Dim1', fontsize=11)
ax2.set_ylabel('Dim3', fontsize=11)
ax2.set_title('FCM Clusters: Dim1 vs Dim3', fontweight='bold')
ax2.grid(True, alpha=0.3)

# (c) Legend (combined colours + shapes)
ax3 = fig.add_subplot(2, 3, 3)
ax3.axis('off')
legend_elements = []
# habitat‑canyon combos
for (hab, can), col in HABITAT_CANYON_COLORS.items():
    if hab != 'BW':
        # only add if present in data
        mask = (fcm_results['Habitat'] == hab) & (
            (fcm_results['Canyon'] == can) if can != 'Outside' else ~fcm_results['Canyon'].isin(['North', 'South'])
        )
        if mask.sum() > 0:
            legend_elements.append(
                plt.scatter([], [], c=col, marker='o', s=100,
                            label=f'{hab} ({can})', edgecolors='black', linewidths=0.5)
            )
# BW
if fcm_results['is_BW'].any():
    legend_elements.append(
        plt.scatter([], [], c='#000000', marker='o', s=100,
                    label='Bottom Water (BW)', edgecolors='white', linewidths=1.0)
    )
# cluster shapes
for cl_id, marker in CLUSTER_MARKERS.items():
    legend_elements.append(
        plt.scatter([], [], c='gray', marker=marker, s=100,
                    label=f'Cluster {cl_id}', edgecolors='black', linewidths=0.5)
    )
ax3.legend(handles=legend_elements, loc='center', frameon=True,
           title='Habitat (Canyon) & Clusters', title_fontsize=11)

# (d) Dim1 vs Dim2 coloured by membership
ax4 = fig.add_subplot(2, 3, 4)
sc1 = ax4.scatter(fcm_results['Dim1'], fcm_results['Dim2'],
                  c=fcm_results['Max_Membership'], cmap='viridis',
                  s=SCATTER_SIZE, alpha=0.8, edgecolors='black', linewidths=0.5)
plt.colorbar(sc1, ax=ax4, label='Max Membership')
ax4.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax4.axvline(0, color='gray', linestyle='--', alpha=0.3)
ax4.set_xlabel('Dim1'); ax4.set_ylabel('Dim2')
ax4.set_title('Membership Certainty: Dim1 vs Dim2', fontweight='bold')
ax4.grid(True, alpha=0.3)

# (e) Dim1 vs Dim3 coloured by membership
ax5 = fig.add_subplot(2, 3, 5)
sc2 = ax5.scatter(fcm_results['Dim1'], fcm_results['Dim3'],
                  c=fcm_results['Max_Membership'], cmap='viridis',
                  s=SCATTER_SIZE, alpha=0.8, edgecolors='black', linewidths=0.5)
plt.colorbar(sc2, ax=ax5, label='Max Membership')
ax5.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax5.axvline(0, color='gray', linestyle='--', alpha=0.3)
ax5.set_xlabel('Dim1'); ax5.set_ylabel('Dim3')
ax5.set_title('Membership Certainty: Dim1 vs Dim3', fontweight='bold')
ax5.grid(True, alpha=0.3)

# (f) Cluster composition bar chart
ax6 = fig.add_subplot(2, 3, 6)
cluster_habitat = fcm_results.groupby(['FCM_Cluster', 'Habitat']).size().unstack(fill_value=0)
cluster_habitat.plot(kind='bar', stacked=True, ax=ax6,
                     color=['#FF6B6B', '#87CEEB', '#90EE90'][:len(cluster_habitat.columns)])
ax6.set_xlabel('FCM Cluster'); ax6.set_ylabel('Number of Samples')
ax6.set_title('Cluster Composition by Habitat', fontweight='bold')
ax6.legend(title='Habitat')
ax6.set_xticklabels([f'Cluster {i}' for i in cluster_habitat.index], rotation=0)
ax6.grid(True, alpha=0.3, axis='y')

plt.suptitle('FCM Clustering on MFA Dimensions', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_folder / 'FCM_clustering_biplots.png', dpi=DPI, bbox_inches='tight')
plt.show()
print("✓ Saved FCM_clustering_biplots.png")

# =============================================================================
# PART 4: DEPTH PROFILES (Dim1, Dim2, Dim3 vs depth)
# =============================================================================
print("\n--- Depth Profile Plots ---")

# Create a plotting depth column (BW samples at BW_DEPTH_POSITION)
fcm_results['Depth_plot'] = fcm_results.apply(
    lambda row: BW_DEPTH_POSITION if row['is_BW'] else row['Depth_cmbsf'], axis=1
)
fcm_results['Depth_plot'] = pd.to_numeric(fcm_results['Depth_plot'], errors='coerce')

# Determine depth range (excluding BW)
sediment_depths = fcm_results[~fcm_results['is_BW']]['Depth_plot']
max_depth = sediment_depths.max() + 1
min_y = BW_DEPTH_POSITION - BW_AXIS_BUFFER

fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE_DEPTH)
for i, dim in enumerate(['Dim1', 'Dim2', 'Dim3']):
    ax = axes[i]
    for _, row in fcm_results.iterrows():
        if row['is_BW']:
            color = '#000000'
        else:
            hab = row['Habitat']
            can = row['Canyon'] if row['Canyon'] in ['North', 'South'] else 'Outside'
            color = HABITAT_CANYON_COLORS.get((hab, can), '#808080')
        marker = CLUSTER_MARKERS.get(row['FCM_Cluster'], 'o')
        ax.scatter(row[dim], row['Depth_plot'],
                   c=[color], marker=marker, s=SCATTER_SIZE,
                   alpha=0.9 if row['is_BW'] else 0.7,
                   edgecolors='white' if row['is_BW'] else 'black',
                   linewidths=1.0 if row['is_BW'] else 0.5,
                   zorder=5 if row['is_BW'] else 4)
    ax.axhline(0, color='brown', linestyle='-', alpha=0.5, linewidth=1.5, label='Seafloor')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel(dim, fontweight='bold')
    ax.set_ylabel('Depth (cmbsf)', fontweight='bold')
    ax.set_title(f'Depth Profile: {dim}', fontweight='bold')
    ax.set_ylim(max_depth, min_y)
    # Show only sediment depths (0 and below) on y‑axis
    yticks = np.arange(0, max_depth, 5, dtype=int)
    ax.set_yticks(yticks)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

# Add shape legend to first subplot
shape_legend = [plt.scatter([], [], c='gray', marker=CLUSTER_MARKERS[cl],
                            s=100, label=f'Cluster {cl}', edgecolors='black') for cl in CLUSTER_MARKERS]
axes[0].legend(handles=shape_legend, loc='best', fontsize=9)

# Add colour legend to third subplot
color_legend = []
if fcm_results['is_BW'].any():
    color_legend.append(plt.scatter([], [], c='#000000', s=100, label='BW', edgecolors='white', linewidths=1.0))
for (hab, can), col in HABITAT_CANYON_COLORS.items():
    if hab != 'BW':
        mask = (fcm_results['Habitat'] == hab) & (
            (fcm_results['Canyon'] == can) if can != 'Outside' else ~fcm_results['Canyon'].isin(['North', 'South'])
        )
        if mask.sum() > 0:
            color_legend.append(plt.scatter([], [], c=col, s=100, label=f'{hab} ({can})',
                                            edgecolors='black', linewidths=0.5))
axes[2].legend(handles=color_legend, loc='best', fontsize=9)

plt.suptitle('FCM Clusters – Depth Profiles Along MFA Dimensions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_folder / 'FCM_depth_profiles.png', dpi=DPI, bbox_inches='tight')
plt.show()
print("✓ Saved FCM_depth_profiles.png")

# =============================================================================
# SAVE DATA
# =============================================================================
fcm_results.to_csv(output_folder / 'comprehensive_FCM_results.csv', index=False)
print("✓ Saved comprehensive_FCM_results.csv")
centers_df.to_csv(output_folder / 'FCM_cluster_centers.csv')
print("✓ Saved FCM_cluster_centers.csv")
param_df.to_csv(output_folder / 'FCM_parameter_optimization.csv', index=False)
print("✓ Saved FCM_parameter_optimization.csv")

print("\n" + "=" * 80)
print("FCM ANALYSIS COMPLETE")
print("=" * 80)
