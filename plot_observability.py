import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# --- 1. DEFINE MAPPINGS ONCE (HARDCODED LOOKUPS) ---
JOINT_SEGMENT_DICT = {
    'Lumbar': ('torso_imu', 'pelvis_imu'),
    'R_Hip': ('pelvis_imu', 'femur_r_imu'),
    'R_Knee': ('femur_r_imu', 'tibia_r_imu'),
    'R_Ankle': ('tibia_r_imu', 'calcn_r_imu'),
    'L_Hip': ('pelvis_imu', 'femur_l_imu'),
    'L_Knee': ('femur_l_imu', 'tibia_l_imu'),
    'L_Ankle': ('tibia_l_imu', 'calcn_l_imu'),
}

def normalize_imu_name(imu_name):
    if 'pelvis' in imu_name: return 'Pelvis'
    if 'torso' in imu_name: return 'Torso'
    if 'femur' in imu_name: return 'Femur'
    if 'tibia' in imu_name: return 'Tibia'
    if 'calcn' in imu_name: return 'Foot'
    return 'Unknown'

# Pre-build lookup dictionaries
GENERIC_JOINT_MAP = {}
PARENT_SEG_MAP = {}
CHILD_SEG_MAP = {}

for specific_joint, (p_imu, c_imu) in JOINT_SEGMENT_DICT.items():
    if 'Lumbar' in specific_joint:
        GENERIC_JOINT_MAP[specific_joint] = 'Lumbar'
    else:
        GENERIC_JOINT_MAP[specific_joint] = specific_joint.split('_')[-1]
    
    PARENT_SEG_MAP[specific_joint] = normalize_imu_name(p_imu)
    CHILD_SEG_MAP[specific_joint] = normalize_imu_name(c_imu)

# ---------------------------------------------------

def prepare_data(df):
    """
    Optimized data preparation.
    Calculates Parent, Child, and Min(Parent, Child) statistics.
    """
    df['Generic_Joint'] = df['Joint'].map(GENERIC_JOINT_MAP)
    df['Parent_Segment'] = df['Joint'].map(PARENT_SEG_MAP)
    df['Child_Segment'] = df['Joint'].map(CHILD_SEG_MAP)

    # 1. Parent Data
    df_parent = df[['Generic_Joint', 'Parent_Segment', 'O_Parent']].copy()
    df_parent.rename(columns={'Generic_Joint': 'Joint', 'Parent_Segment': 'Segment', 'O_Parent': 'Observability'}, inplace=True)
    df_parent['Role'] = 'Parent'
    
    # 2. Child Data
    df_child = df[['Generic_Joint', 'Child_Segment', 'O_Child']].copy()
    df_child.rename(columns={'Generic_Joint': 'Joint', 'Child_Segment': 'Segment', 'O_Child': 'Observability'}, inplace=True)
    df_child['Role'] = 'Child'
    
    # 3. Min Data (Intersection)
    # Calculate row-wise minimum of parent and child
    df['O_Min'] = df[['O_Parent', 'O_Child']].min(axis=1)
    df_min = df[['Generic_Joint', 'O_Min']].copy()
    df_min.rename(columns={'Generic_Joint': 'Joint', 'O_Min': 'Observability'}, inplace=True)
    df_min['Role'] = 'Min'
    df_min['Segment'] = 'Intersection' # Placeholder name
    
    return pd.concat([df_parent, df_child, df_min], ignore_index=True)


def plot_median_iqr(df_long):
    """
    Figure 2: Custom Point Plot (Median + IQR)
    Includes Parent, Child, and the 'Min' intersection in grey.
    """
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")
    
    # 1. Calculate Statistics
    stats = df_long.groupby(['Joint', 'Role'])['Observability'].agg(
        median='median', 
        q25=lambda x: np.percentile(x, 25), 
        q75=lambda x: np.percentile(x, 75)
    ).reset_index()

    # Get Segment names for coloring
    seg_map = df_long.groupby(['Joint', 'Role'])['Segment'].first().reset_index()
    stats = pd.merge(stats, seg_map, on=['Joint', 'Role'])
    
    # --- PRINT STATS ---
    print("\n=== Observability Statistics (Median & IQR) ===")
    stats['IQR'] = stats['q75'] - stats['q25']
    print_cols = ['Joint', 'Role', 'Segment', 'median', 'IQR']
    
    # Custom sort for display
    sorter = {'Lumbar': 0, 'Hip': 1, 'Knee': 2, 'Ankle': 3}
    stats['sort_key'] = stats['Joint'].map(sorter)
    stats_sorted = stats.sort_values(['sort_key', 'Role']).drop('sort_key', axis=1)
    
    print(stats_sorted[print_cols].to_string(index=False, float_format="%.2f"))
    print("============================================\n")
    # -------------------

    # 2. Setup Plot Scaffolding
    unique_segments = sorted([s for s in df_long['Segment'].unique() if s != 'Intersection'])
    colors = sns.color_palette("Set2", n_colors=len(unique_segments))
    segment_palette = dict(zip(unique_segments, colors))
    
    joint_order = ['Lumbar', 'Hip', 'Knee', 'Ankle']
    joint_order = [j for j in joint_order if j in stats['Joint'].unique()]
    x_map = {j: i for i, j in enumerate(joint_order)}
    
    offset = 0.2

    # 3. Plot Manually
    for _, row in stats.iterrows():
        if row['Joint'] not in x_map: continue
        
        x_base = x_map[row['Joint']]
        
        if row['Role'] == 'Parent':
            x_pos = x_base - offset
            color = segment_palette.get(row['Segment'], 'grey')
            alpha_val = 1.0
        elif row['Role'] == 'Child':
            x_pos = x_base + offset
            color = segment_palette.get(row['Segment'], 'grey')
            alpha_val = 1.0
        elif row['Role'] == 'Min':
            continue
            x_pos = x_base
            color = 'grey'
            alpha_val = 1.0
        
        lower_err = row['median'] - row['q25']
        upper_err = row['q75'] - row['median']
        
        plt.errorbar(
            x=x_pos, 
            y=row['median'], 
            yerr=[[lower_err], [upper_err]], 
            fmt='o',            
            markerfacecolor=color,  
            markeredgecolor=color,  
            ecolor=color,      
            elinewidth=2.5,     
            capsize=5,          
            markersize=8,
            alpha=alpha_val
        )

    # 4. Formatting
    plt.xticks(range(len(joint_order)), joint_order, fontsize=18)
    plt.yticks(fontsize=14)
    plt.xlim(-0.5, len(joint_order) - 0.5)
    
    legend_patches = [mpatches.Patch(color=segment_palette[s], label=s) for s in unique_segments]
    legend_patches.append(mpatches.Patch(color='grey', label='Minimum'))
    
    plt.legend(handles=legend_patches, title='Segment', loc='upper left', fontsize=14, title_fontsize=14)

    plt.ylabel(r'Observability ($\text{m}^2\text{/s}^5$)', fontsize=18)
    plt.axhline(y=150.0, color='r', linestyle='--', alpha=0.5)

    sns.despine(bottom=True)
    ax = plt.gca()
    ax.spines['left'].set_linewidth(0.5)
    
    plt.tight_layout()
    # Ensure plots directory exists
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    plt.savefig('plots/observability_median_iqr.pdf', dpi=400)
    plt.show()

if __name__ == "__main__":
    if os.path.exists('observability_components.csv'):
        print("Loading existing CSV...")
        df = pd.read_csv('observability_components.csv')
        
        df_long = prepare_data(df)

        print("Plotting Median/IQR with Min Intersection...")
        plot_median_iqr(df_long)
    else:
        print("CSV not found. Please run the main analysis script first.")