import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Create visuals directory if it doesn't exist
os.makedirs('visuals', exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load trained models and data
try:
    rf_model = joblib.load("baseball_rf_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    print("✓ Models loaded successfully")
except:
    print("Error: Models not found. Run analyze_baseball_data.py first.")
    exit()

# Load data
batter_data = pd.read_csv("./Savant Batter 2021-2025.csv")
pitcher_data = pd.read_csv("./Savant Pitcher 2021-2025.csv")

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
# EXECUTE ALL VISUALIZATIONS
# ============================================================================
import numpy as np

if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    plot_feature_importance()
    plot_performance_trends()
    plot_player_archetypes()
    plot_age_performance()
    plot_correlation_heatmap()
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE")
    print("="*70)
    print("\nGenerated files in 'visuals/' directory:")
    print("  • visuals/feature_importance.png")
    print("  • visuals/performance_trends.png")
    print("  • visuals/player_archetypes.png")
    print("  • visuals/age_performance.png")
    print("  • visuals/correlation_heatmap.png")
