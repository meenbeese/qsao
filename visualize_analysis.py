import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
import numpy as np
warnings.filterwarnings('ignore')

# Create visuals directory if it doesn't exist
os.makedirs('visuals', exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load trained models and data
try:
    rf_model = joblib.load("./models/baseball_rf_model.pkl")
    feature_columns = joblib.load("./models/feature_columns.pkl")
    print("✓ Models loaded successfully")
except:
    print("Error: Models not found. Run analyze_baseball_data.py first.")
    exit()

# Load data
batter_data = pd.read_csv("./Savant Batter 2021-2025.csv")
pitcher_data = pd.read_csv("./Savant Pitcher 2021-2025.csv")

# Load KC Royals data
try:
    kc_data = pd.read_csv('./kc_royals.csv')
    print("✓ KC Royals data loaded successfully")
except:
    kc_data = None
    print("Warning: KC Royals data not found")

# ============================================================================
# VISUALIZATION 1: Feature Importance
# ============================================================================
def plot_feature_importance():
    """Plot top 15 most important features"""
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Top 15 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig('visuals/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Feature importance plot saved: visuals/feature_importance.png")
    plt.close()

# ============================================================================
# VISUALIZATION 2: Performance Metrics Over Time
# ============================================================================
def plot_performance_trends():
    """Plot batting performance trends by year"""
    yearly_stats = batter_data.groupby('year').agg({
        'batting_avg': 'mean',
        'on_base_percent': 'mean',
        'slg_percent': 'mean',
        'home_run': 'sum'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Batting Average
    axes[0, 0].plot(yearly_stats['year'], yearly_stats['batting_avg'], marker='o', linewidth=2, color='#1f77b4')
    axes[0, 0].set_title('Avg Batting Average by Year', fontweight='bold')
    axes[0, 0].set_ylabel('Batting Average')
    
    # On-Base Percentage
    axes[0, 1].plot(yearly_stats['year'], yearly_stats['on_base_percent'], marker='s', linewidth=2, color='#ff7f0e')
    axes[0, 1].set_title('Avg On-Base % by Year', fontweight='bold')
    axes[0, 1].set_ylabel('OBP')
    
    # Slugging Percentage
    axes[1, 0].plot(yearly_stats['year'], yearly_stats['slg_percent'], marker='^', linewidth=2, color='#2ca02c')
    axes[1, 0].set_title('Avg Slugging % by Year', fontweight='bold')
    axes[1, 0].set_ylabel('SLG')
    
    # Home Runs
    axes[1, 1].bar(yearly_stats['year'], yearly_stats['home_run'], color='#d62728', alpha=0.7)
    axes[1, 1].set_title('Total Home Runs by Year', fontweight='bold')
    axes[1, 1].set_ylabel('Home Runs')
    
    for ax in axes.flat:
        ax.set_xlabel('Year')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visuals/performance_trends.png', dpi=300, bbox_inches='tight')
    print("✓ Performance trends plot saved: visuals/performance_trends.png")
    plt.close()

# ============================================================================
# VISUALIZATION 3: Player Archetype Distribution
# ============================================================================
def plot_player_archetypes():
    """Categorize players by archetype"""
    current_year = batter_data['year'].max()
    recent = batter_data[batter_data['year'] == current_year].copy()
    
    # Define archetypes
    recent['archetype'] = 'Other'
    recent.loc[(recent['slg_percent'] > 0.480) & (recent['isolated_power'] > 0.200), 'archetype'] = 'Power Hitter'
    recent.loc[(recent['on_base_percent'] > 0.360) & (recent['batting_avg'] > 0.270), 'archetype'] = 'High OBP'
    recent.loc[(recent['batting_avg'] > 0.260) & (recent['on_base_percent'] > 0.330) & (recent['slg_percent'] > 0.420), 'archetype'] = 'Balanced'
    recent.loc[(recent['r_total_stolen_base'] > 15) & (recent['batting_avg'] > 0.250), 'archetype'] = 'Speed Threat'
    
    archetype_counts = recent['archetype'].value_counts()
    
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    plt.pie(archetype_counts.values, labels=archetype_counts.index, autopct='%1.1f%%', 
            colors=colors[:len(archetype_counts)], startangle=90)
    plt.title(f'Player Archetypes Distribution ({current_year})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visuals/player_archetypes.png', dpi=300, bbox_inches='tight')
    print("✓ Player archetypes plot saved: visuals/player_archetypes.png")
    plt.close()

# ============================================================================
# VISUALIZATION 4: Age vs Performance
# ============================================================================
def plot_age_performance():
    """Analyze relationship between player age and performance"""
    current_year = batter_data['year'].max()
    recent = batter_data[batter_data['year'] == current_year].copy()
    recent = recent.dropna(subset=['player_age', 'batting_avg', 'slg_percent'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Age vs Batting Average
    axes[0].scatter(recent['player_age'], recent['batting_avg'], alpha=0.5, s=50)
    z = np.polyfit(recent['player_age'].dropna(), recent['batting_avg'].dropna(), 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(recent['player_age'].min(), recent['player_age'].max(), 100)
    axes[0].plot(x_smooth, p(x_smooth), "r--", linewidth=2, label='Trend')
    axes[0].set_xlabel('Player Age')
    axes[0].set_ylabel('Batting Average')
    axes[0].set_title('Age vs Batting Average', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Age vs Slugging Percentage
    axes[1].scatter(recent['player_age'], recent['slg_percent'], alpha=0.5, s=50, color='orange')
    z = np.polyfit(recent['player_age'].dropna(), recent['slg_percent'].dropna(), 2)
    p = np.poly1d(z)
    axes[1].plot(x_smooth, p(x_smooth), "r--", linewidth=2, label='Trend')
    axes[1].set_xlabel('Player Age')
    axes[1].set_ylabel('Slugging Percentage')
    axes[1].set_title('Age vs Slugging Percentage', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visuals/age_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Age vs performance plot saved: visuals/age_performance.png")
    plt.close()

# ============================================================================
# VISUALIZATION 5: Correlation Heatmap
# ============================================================================
def plot_correlation_heatmap():
    """Plot correlation matrix of key batting metrics"""
    current_year = batter_data['year'].max()
    recent = batter_data[batter_data['year'] == current_year].copy()
    
    key_cols = ['batting_avg', 'on_base_percent', 'slg_percent', 'isolated_power', 
                'home_run', 'strikeout', 'walk', 'r_total_stolen_base']
    corr_data = recent[key_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix: Batting Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('visuals/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Correlation heatmap saved: visuals/correlation_heatmap.png")
    plt.close()

# ============================================================================
# VISUALIZATION 6: KC ROYALS ARCHETYPE DISTRIBUTION
# ============================================================================
def plot_kc_archetype_distribution():
    """Visualize KC Royals pitcher archetypes"""
    if kc_data is None:
        print("⚠ Skipping KC archetype visualization (data not available)")
        return
    
    from kc_royals_analysis import PlayerEvaluationModel
    
    eval_model = PlayerEvaluationModel(kc_data)
    archetypes = eval_model.create_pitcher_archetypes()
    kc_roster = eval_model.assign_player_archetypes(archetypes)
    
    archetype_counts = kc_roster['Archetype'].value_counts()
    
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = plt.bar(archetype_counts.index, archetype_counts.values, color=colors[:len(archetype_counts)], alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.title('KC Royals Pitcher Archetype Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Archetype', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visuals/kc_archetype_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ KC archetype distribution plot saved: visuals/kc_archetype_distribution.png")
    plt.close()

# ============================================================================
# VISUALIZATION 7: KC ROYALS WAR vs ERA SCATTER
# ============================================================================
def plot_kc_war_vs_era():
    """Scatter plot of KC Royals players by WAR and ERA"""
    if kc_data is None:
        print("⚠ Skipping KC WAR vs ERA visualization (data not available)")
        return
    
    from kc_royals_analysis import PlayerEvaluationModel
    
    eval_model = PlayerEvaluationModel(kc_data)
    archetypes = eval_model.create_pitcher_archetypes()
    kc_roster = eval_model.assign_player_archetypes(archetypes)
    
    # Filter to pitchers with 20+ IP
    qualified = kc_roster[kc_roster['IP'] >= 20].copy()
    
    # Define color map for archetypes
    color_map = {
        'Ace': '#d62728',
        'Reliable_Starter': '#ff7f0e',
        'High_Volume_Reliever': '#2ca02c',
        'Closer': '#1f77b4',
        'Young_Prospect': '#9467bd',
        'Depth Piece': '#7f7f7f'
    }
    colors = [color_map.get(arch, '#7f7f7f') for arch in qualified['Archetype']]
    
    plt.figure(figsize=(12, 7))
    scatter = plt.scatter(qualified['ERA'], qualified['WAR'], s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Add player labels for top performers
    for idx, row in qualified.nlargest(5, 'WAR').iterrows():
        plt.annotate(row['Player'], (row['ERA'], row['WAR']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    # Add reference lines for league average
    plt.axvline(x=4.0, color='gray', linestyle='--', alpha=0.5, label='League Avg ERA (4.0)')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero WAR')
    
    plt.xlabel('ERA', fontsize=12, fontweight='bold')
    plt.ylabel('WAR', fontsize=12, fontweight='bold')
    plt.title('KC Royals: WAR vs ERA (Min 20 IP)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Create custom legend for archetypes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[arch], edgecolor='black', label=arch) 
                      for arch in color_map.keys()]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('visuals/kc_war_vs_era.png', dpi=300, bbox_inches='tight')
    print("✓ KC WAR vs ERA plot saved: visuals/kc_war_vs_era.png")
    plt.close()

# ============================================================================
# VISUALIZATION 8: KC ROYALS AGE CURVE & PERFORMANCE
# ============================================================================
def plot_kc_age_performance():
    """Analyze KC Royals age vs performance"""
    if kc_data is None:
        print("⚠ Skipping KC age performance visualization (data not available)")
        return
    
    qualified = kc_data[kc_data['IP'] >= 20].dropna(subset=['Age', 'ERA'])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Age vs ERA
    axes[0].scatter(qualified['Age'], qualified['ERA'], alpha=0.6, s=100, color='#ff7f0e', edgecolors='black')
    z = np.polyfit(qualified['Age'].dropna(), qualified['ERA'].dropna(), 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(qualified['Age'].min(), qualified['Age'].max(), 100)
    axes[0].plot(x_smooth, p(x_smooth), "r--", linewidth=2.5, label='Age Curve')
    axes[0].axhline(y=4.0, color='gray', linestyle='--', alpha=0.5, label='League Avg')
    axes[0].set_xlabel('Age', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('ERA', fontsize=12, fontweight='bold')
    axes[0].set_title('KC Royals: Age vs ERA', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Age vs WAR
    qualified_war = kc_data[kc_data['IP'] >= 20].dropna(subset=['Age', 'WAR'])
    axes[1].scatter(qualified_war['Age'], qualified_war['WAR'], alpha=0.6, s=100, color='#2ca02c', edgecolors='black')
    z = np.polyfit(qualified_war['Age'].dropna(), qualified_war['WAR'].dropna(), 2)
    p = np.poly1d(z)
    axes[1].plot(x_smooth, p(x_smooth), "r--", linewidth=2.5, label='Age Curve')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Zero WAR')
    axes[1].set_xlabel('Age', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('WAR', fontsize=12, fontweight='bold')
    axes[1].set_title('KC Royals: Age vs WAR', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visuals/kc_age_performance.png', dpi=300, bbox_inches='tight')
    print("✓ KC age performance plot saved: visuals/kc_age_performance.png")
    plt.close()

# ============================================================================
# VISUALIZATION 9: KC ROYALS ARBITRATION CANDIDATES
# ============================================================================
def plot_kc_arbitration_analysis():
    """Visualize arbitration candidates by age and WAR"""
    if kc_data is None:
        print("⚠ Skipping KC arbitration analysis (data not available)")
        return
    
    from kc_royals_analysis import RosterOptimizationDecisions
    
    roster_opt = RosterOptimizationDecisions(kc_data)
    arb_analysis = roster_opt.identify_arbitration_candidates()
    
    arb_candidates = kc_data[
        (kc_data['Age'] >= 27) & 
        (kc_data['Age'] <= 30) &
        (kc_data['WAR'] > 0)
    ]
    
    if len(arb_candidates) == 0:
        print("⚠ No arbitration candidates found")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Categorize by recommendation
    early_ext = arb_analysis['early_extension']
    peak_trade = arb_analysis['peak_trade']
    
    # Plot all arb candidates
    for idx, row in arb_candidates.iterrows():
        if idx in early_ext.index:
            color = '#2ca02c'
            label = 'Extend Early'
        elif idx in peak_trade.index:
            color = '#d62728'
            label = 'Trade at Peak'
        else:
            color = '#7f7f7f'
            label = 'Monitor'
        
        ax.scatter(row['Age'], row['WAR'], s=300, color=color, alpha=0.7, edgecolors='black', linewidth=2)
        ax.annotate(row['Player'], (row['Age'], row['WAR']), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', edgecolor='black', label=f"Extend Early ({len(early_ext)})"),
        Patch(facecolor='#d62728', edgecolor='black', label=f"Trade at Peak ({len(peak_trade)})"),
        Patch(facecolor='#7f7f7f', edgecolor='black', label='Monitor')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.set_xlabel('Age', fontsize=12, fontweight='bold')
    ax.set_ylabel('WAR', fontsize=12, fontweight='bold')
    ax.set_title('KC Royals: Arbitration Candidate Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visuals/kc_arbitration_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ KC arbitration analysis plot saved: visuals/kc_arbitration_analysis.png")
    plt.close()

# ============================================================================
# VISUALIZATION 10: KC ROYALS ROSTER COMPOSITION
# ============================================================================
def plot_kc_roster_composition():
    """Visualize KC Royals roster breakdown by role"""
    if kc_data is None:
        print("⚠ Skipping KC roster composition visualization (data not available)")
        return
    
    from kc_royals_analysis import KCRoyalsTeamAnalysis
    
    team = KCRoyalsTeamAnalysis(kc_data)
    composition = team.analyze_roster_composition()
    
    # Categorize by role
    starters = kc_data[kc_data['GS'] > 20]
    relievers = kc_data[(kc_data['G'] > kc_data['GS']) & (kc_data['IP'] > 50)]
    closers = kc_data[kc_data['SV'] > 5]
    depth = kc_data[~kc_data.index.isin(pd.concat([starters, relievers, closers]).index)]
    
    roles = ['Starters', 'Relievers', 'Closers', 'Depth']
    counts = [len(starters), len(relievers), len(closers), len(depth)]
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#7f7f7f']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    bars = ax1.bar(roles, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('KC Royals: Roster Composition by Role', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(counts, labels=roles, autopct='%1.1f%%', colors=colors, startangle=90, 
           wedgeprops=dict(edgecolor='black', linewidth=2))
    ax2.set_title('KC Royals: Roster Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visuals/kc_roster_composition.png', dpi=300, bbox_inches='tight')
    print("✓ KC roster composition plot saved: visuals/kc_roster_composition.png")
    plt.close()

# ============================================================================
# EXECUTE ALL VISUALIZATIONS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Savant data visualizations
    plot_feature_importance()
    plot_performance_trends()
    plot_player_archetypes()
    plot_age_performance()
    plot_correlation_heatmap()
    
    # KC Royals visualizations
    print("\n" + "-"*70)
    print("KC ROYALS ANALYSIS VISUALIZATIONS")
    print("-"*70 + "\n")
    plot_kc_archetype_distribution()
    plot_kc_war_vs_era()
    plot_kc_age_performance()
    plot_kc_arbitration_analysis()
    plot_kc_roster_composition()
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*70)
    print("\nGenerated files in 'visuals/' directory:")
    print("  SAVANT DATA:")
    print("    • visuals/feature_importance.png")
    print("    • visuals/performance_trends.png")
    print("    • visuals/player_archetypes.png")
    print("    • visuals/age_performance.png")
    print("    • visuals/correlation_heatmap.png")
    print("\n  KC ROYALS ANALYSIS:")
    print("    • visuals/kc_archetype_distribution.png")
    print("    • visuals/kc_war_vs_era.png")
    print("    • visuals/kc_age_performance.png")
    print("    • visuals/kc_arbitration_analysis.png")
    print("    • visuals/kc_roster_composition.png")
