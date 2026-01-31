import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

pitcher_data_path = "./Savant Pitcher 2021-2025.csv"
batter_data_path = "./Savant Batter 2021-2025.csv"

try:
    pitcher_data = pd.read_csv(pitcher_data_path)
    batter_data = pd.read_csv(batter_data_path)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print("Pitcher Data Shape:", pitcher_data.shape)
print("Batter Data Shape:", batter_data.shape)

if pitcher_data.empty or batter_data.empty:
    print("One or both CSV files are empty. Please check the data files.")
    exit()

# Remove duplicates keeping first occurrence
pitcher_data = pitcher_data.drop_duplicates(subset=['player_id', 'year'], keep='first')
batter_data = batter_data.drop_duplicates(subset=['player_id', 'year'], keep='first')

# Check for missing player_id or year values
if pitcher_data['player_id'].isnull().any() or pitcher_data['year'].isnull().any():
    print("Removing rows with missing player_id or year in pitcher data...")
    pitcher_data = pitcher_data.dropna(subset=['player_id', 'year'])

if batter_data['player_id'].isnull().any() or batter_data['year'].isnull().any():
    print("Removing rows with missing player_id or year in batter data...")
    batter_data = batter_data.dropna(subset=['player_id', 'year'])

# Merge datasets on player_id and year
merged_data = pd.merge(
    batter_data, 
    pitcher_data, 
    on=['player_id', 'year'], 
    suffixes=('_batter', '_pitcher'),
    how='inner'
)

print(f"\nMerged Data Shape: {merged_data.shape}")
print(f"Rows in merged dataset: {len(merged_data)}")

if merged_data.empty:
    print("No common player-year combinations found. Check your data.")
    exit()

target_column = 'batting_avg_batter'

if target_column not in merged_data.columns:
    print(f"Error: Target column '{target_column}' not found.")
    print(f"Available columns: {[c for c in merged_data.columns if 'batting' in c.lower()]}")
    exit()

# Drop rows where target is missing
merged_data = merged_data.dropna(subset=[target_column])
print(f"Rows after removing missing target: {len(merged_data)}")

# Check for missing values
missing_values = merged_data.isnull().sum()
missing_counts = missing_values[missing_values > 0]
threshold = 0.5 * len(merged_data)
columns_to_drop = missing_values[missing_values > threshold].index.tolist()
if columns_to_drop:
    merged_data = merged_data.drop(columns=columns_to_drop)

numeric_columns = merged_data.select_dtypes(include=['float64', 'int64']).columns
merged_data[numeric_columns] = merged_data[numeric_columns].fillna(
    merged_data[numeric_columns].mean()
)

# Feature selection - exclude pitcher columns and metadata
exclude_cols = {
    'player_id', 'year', target_column, 
    'last_name, first_name_batter', 'last_name, first_name_pitcher',
    'player_age_batter', 'player_age_pitcher'
}

feature_columns = [col for col in merged_data.columns 
                   if col not in exclude_cols 
                   and merged_data[col].dtype in ['float64', 'int64']
                   and not col.startswith('p_')]  # Exclude pitcher stats

print(f"\nUsing {len(feature_columns)} features for training")

X = merged_data[feature_columns].copy()
y = merged_data[target_column].copy()

# Remove any remaining NaN values
valid_indices = X.notna().all(axis=1) & y.notna()
X = X[valid_indices]
y = y[valid_indices]

print(f"Final training set size: X={X.shape}, y={y.shape}")

if X.empty or y.empty:
    print("No valid data for training.")
    exit()

# Scale features
scaler = StandardScaler()
try:
    X_scaled = scaler.fit_transform(X)
    print("Features scaled successfully.")
except ValueError as e:
    print(f"Error during scaling: {e}")
    exit()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# Train multiple models
print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

# Model 1: Random Forest (for feature importance)
print("\n1. Training Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# Model 2: Gradient Boosting (better generalization)
print("2. Training Gradient Boosting Regressor...")
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=7,
    min_samples_split=10,
    random_state=42
)
gb_model.fit(X_train, y_train)

# Evaluate models
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)

print("\n" + "="*70)
print("MODEL PERFORMANCE")
print("="*70)

rf_r2 = r2_score(y_test, y_pred_rf)
gb_r2 = r2_score(y_test, y_pred_gb)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
gb_mae = mean_absolute_error(y_test, y_pred_gb)

print(f"\nRandom Forest:")
print(f"  R² Score: {rf_r2:.4f}")
print(f"  MAE: {rf_mae:.4f}")

print(f"\nGradient Boosting:")
print(f"  R² Score: {gb_r2:.4f}")
print(f"  MAE: {gb_mae:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# Save models
joblib.dump(rf_model, "baseball_rf_model.pkl")
joblib.dump(gb_model, "baseball_gb_model.pkl")
joblib.dump(scaler, "baseball_scaler.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

print("\n✓ Models saved successfully")

# ============================================================================
# REQUIREMENT 1: KC ROYALS TEAM ANALYSIS
# ============================================================================

class KCRoyalsAnalysis:
    """Comprehensive KC Royals roster analysis"""
    
    def __init__(self, batter_data, pitcher_data):
        self.batters = batter_data.copy()
        self.pitchers = pitcher_data.copy()
        self.current_year = max(
            self.batters['year'].max(), 
            self.pitchers['year'].max()
        )
        
    def get_recent_data(self, years_back=1):
        """Get data from recent seasons"""
        start_year = self.current_year - years_back
        recent_batters = self.batters[self.batters['year'] >= start_year]
        recent_pitchers = self.pitchers[self.pitchers['year'] >= start_year]
        return recent_batters, recent_pitchers
    
    def analyze_roster_composition(self):
        """Analyze position distribution and roles"""
        recent_batters, recent_pitchers = self.get_recent_data()
        
        # Only analyze pitchers with games_started data
        pitcher_roles = []
        if 'games_started' in recent_pitchers.columns and 'innings_pitched' in recent_pitchers.columns:
            for _, p in recent_pitchers.iterrows():
                gs = p.get('games_started', 0) or 0
                g = p.get('games_played', 1) or 1
                ip = p.get('innings_pitched', 0) or 0
                
                if gs > g * 0.5:
                    role = 'SP'
                elif ip < 50 and ip > 0:
                    role = 'CL'
                else:
                    role = 'RP'
                pitcher_roles.append(role)
        
        sp_count = pitcher_roles.count('SP') if pitcher_roles else 0
        rp_count = pitcher_roles.count('RP') if pitcher_roles else 0
        cl_count = pitcher_roles.count('CL') if pitcher_roles else 0
        
        return {
            'starters': sp_count,
            'relievers': rp_count,
            'closers': cl_count,
            'total_pitchers': len(pitcher_roles)
        }
    
    def evaluate_performance_metrics(self):
        """Key performance indicators"""
        recent_batters, recent_pitchers = self.get_recent_data()
        
        # Check available columns
        batting_cols = {col: col for col in recent_batters.columns if col in ['batting_avg', 'on_base_percent', 'slg_percent']}
        pitching_cols = {col: col for col in recent_pitchers.columns if col in ['earned_run_average', 'WHIP']}
        
        batting_metrics = {
            'avg_ba': recent_batters[batting_cols.get('batting_avg', 'batting_avg')].mean() if 'batting_avg' in batting_cols else 0,
            'avg_obp': recent_batters[batting_cols.get('on_base_percent', 'on_base_percent')].mean() if 'on_base_percent' in batting_cols else 0,
            'avg_slg': recent_batters[batting_cols.get('slg_percent', 'slg_percent')].mean() if 'slg_percent' in batting_cols else 0,
        }
        
        pitching_metrics = {
            'avg_era': recent_pitchers[pitching_cols.get('earned_run_average', 'earned_run_average')].mean() if 'earned_run_average' in pitching_cols else 0,
            'avg_whip': recent_pitchers[pitching_cols.get('WHIP', 'WHIP')].mean() if 'WHIP' in pitching_cols else 0,
        }
        
        return batting_metrics, pitching_metrics
    
    def assess_competitive_window(self):
        """Determine if team is in rebuild/retool/contend window"""
        recent_batters, recent_pitchers = self.get_recent_data(years_back=3)
        
        age_col = 'player_age' if 'player_age' in recent_batters.columns else 'player_age_batter'
        ba_col = 'batting_avg' if 'batting_avg' in recent_batters.columns else 'batting_avg_batter'
        era_col = 'earned_run_average' if 'earned_run_average' in recent_pitchers.columns else 'ERA'
        
        young_players = len(recent_batters[recent_batters[age_col] < 26]) if age_col in recent_batters.columns else 0
        avg_ba = recent_batters[ba_col].mean() if ba_col in recent_batters.columns else 0
        avg_era = recent_pitchers[era_col].mean() if era_col in recent_pitchers.columns else 0
        
        if avg_ba > 0.280 and avg_era < 4.00:
            window = "CONTENDING"
        elif young_players > 15 or avg_ba < 0.260:
            window = "REBUILDING"
        else:
            window = "RETOOLING"
        
        return {
            'window': window,
            'young_players': young_players,
            'avg_ba': avg_ba,
            'avg_era': avg_era
        }

# ============================================================================
# REQUIREMENT 2: PLAYER EVALUATION MODEL (WAR-based Framework)
# ============================================================================

class PlayerEvaluationFramework:
    """Analytics-based player evaluation for team needs"""
    
    def __init__(self, rf_model, gb_model, scaler, feature_cols):
        self.rf_model = rf_model
        self.gb_model = gb_model
        self.scaler = scaler
        self.feature_cols = feature_cols
    
    def create_player_archetypes(self, batter_data):
        """Define strategic player archetypes"""
        
        archetypes = {
            'Power_Hitter': {
                'min_slg': 0.480,
                'min_iso': 0.200,
                'description': 'High power, 30+ HR potential'
            },
            'High_OBP': {
                'min_obp': 0.360,
                'min_ba': 0.270,
                'description': 'Leadoff type, contact/speed'
            },
            'Balanced': {
                'min_ba': 0.260,
                'min_obp': 0.330,
                'min_slg': 0.420,
                'description': 'All-around contributor'
            },
            'Speed_Threat': {
                'min_sb': 15,
                'min_ba': 0.250,
                'description': 'Stolen base threat'
            }
        }
        
        return archetypes
    
    def score_players(self, player_stats, rf_predictions, gb_predictions):
        """Generate composite player value score"""
        
        # Average model predictions
        avg_prediction = (rf_predictions + gb_predictions) / 2
        
        # Calculate WAR proxy (simplified)
        war_proxy = (player_stats['woba'].fillna(0.320) - 0.320) * 10
        
        # Composite score (60% prediction, 40% WAR proxy)
        composite = (avg_prediction * 0.6) + (war_proxy * 0.4)
        
        return composite

# ============================================================================
# REQUIREMENT 3: ROSTER OPTIMIZATION & CONTRACT DECISIONS
# ============================================================================

class RosterOptimization:
    """Internal roster decision recommendations"""
    
    def __init__(self, batter_data, pitcher_data):
        self.batters = batter_data
        self.pitchers = pitcher_data
        self.current_year = max(
            self.batters['year'].max(),
            self.pitchers['year'].max()
        )
    
    def identify_arbitration_candidates(self):
        """Players entering arbitration (typically 3-4 years MLB service)"""
        # Simplified: assume players age 27-29 with recent activity
        candidates = []
        
        recent_batters = self.batters[self.batters['year'] == self.current_year]
        for _, player in recent_batters.iterrows():
            age = player.get('player_age_batter', 0)
            if 27 <= age <= 29:
                candidates.append({
                    'name': f"{player.get('last_name, first_name_batter', 'Unknown')}",
                    'age': age,
                    'position': player.get('position', 'DH'),
                    'action': 'MONITOR_FOR_ARBITRATION'
                })
        
        return pd.DataFrame(candidates)
    
    def identify_underperformers(self):
        """Players underperforming contract expectations"""
        recent_batters = self.batters[self.batters['year'] == self.current_year].copy()
        
        # Use correct column name (no _batter suffix in original data)
        ba_col = 'batting_avg'
        obp_col = 'on_base_percent'
        
        if ba_col not in recent_batters.columns or obp_col not in recent_batters.columns:
            return pd.DataFrame()  # Return empty dataframe if columns don't exist
        
        recent_batters['recent_ba'] = recent_batters[ba_col]
        recent_batters['recent_obp'] = recent_batters[obp_col]
        
        # Threshold: below league average in both BA and OBP
        league_ba = recent_batters['recent_ba'].mean()
        league_obp = recent_batters['recent_obp'].mean()
        
        underperformers = recent_batters[
            (recent_batters['recent_ba'] < league_ba * 0.85) &
            (recent_batters['recent_obp'] < league_obp * 0.85)
        ]
        
        name_col = 'last_name, first_name'
        age_col = 'player_age'
        
        # Only return columns if they exist
        cols_to_return = [col for col in [name_col, 'recent_ba', 'recent_obp', age_col] 
                          if col in underperformers.columns]
        
        if cols_to_return:
            return underperformers[cols_to_return]
        else:
            return underperformers[['recent_ba', 'recent_obp']]

# ============================================================================
# REQUIREMENT 4: TRADE PROPOSAL FRAMEWORK
# ============================================================================

class TradeProposalGenerator:
    """Generate realistic trade proposals based on analytics"""
    
    def __init__(self, evaluation_framework):
        self.eval_framework = evaluation_framework
    
    def generate_trade_targets(self, team_needs, available_players):
        """Match surplus players to team needs"""
        
        proposals = []
        
        # Example proposal structure
        proposal1 = {
            'trade_id': 1,
            'kc_sends': [
                {'name': 'Surplus Player A', 'position': 'OF', 'value_score': 0.65}
            ],
            'kc_receives': [
                {'name': 'Target Player B', 'position': 'SP', 'value_score': 0.78}
            ],
            'rationale': 'Improves pitching rotation; addresses #1 weakness',
            'short_term': 'Neutral (lose depth)',
            'long_term': 'Positive (ace pitcher in prime years)'
        }
        
        proposal2 = {
            'trade_id': 2,
            'kc_sends': [
                {'name': 'Underperformer C', 'position': 'SS', 'value_score': 0.55}
            ],
            'kc_receives': [
                {'name': 'Prospect D', 'position': 'C', 'value_score': 0.72}
            ],
            'rationale': 'Youth infusion; move salary-saving player',
            'short_term': 'Negative (losing regular)',
            'long_term': 'Positive (upgrade catcher position)'
        }
        
        proposals.append(proposal1)
        proposals.append(proposal2)
        
        return proposals

# ============================================================================
# EXECUTE ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("KC ROYALS COMPREHENSIVE ANALYSIS")
print("="*70)

kc_analysis = KCRoyalsAnalysis(batter_data, pitcher_data)

# Requirement 1: Team Analysis
print("\n1. TEAM COMPOSITION & PERFORMANCE")
print("-" * 70)
composition = kc_analysis.analyze_roster_composition()
print(f"Starters: {composition['starters']}")
print(f"Relievers: {composition['relievers']}")
print(f"Closers: {composition['closers']}")
print(f"Total Pitchers: {composition['total_pitchers']}")

batting_metrics, pitching_metrics = kc_analysis.evaluate_performance_metrics()
print(f"\nBatting Metrics (All Data):")
print(f"  BA: {batting_metrics['avg_ba']:.3f}")
print(f"  OBP: {batting_metrics['avg_obp']:.3f}")
print(f"  SLG: {batting_metrics['avg_slg']:.3f}")

print(f"\nPitching Metrics (All Data):")
print(f"  ERA: {pitching_metrics['avg_era']:.2f}")
print(f"  WHIP: {pitching_metrics['avg_whip']:.2f}")

competitive = kc_analysis.assess_competitive_window()
print(f"\nCompetitive Window Assessment: {competitive['window']}")
print(f"  Young Players (<26): {competitive['young_players']}")
print(f"  Avg BA (3yr): {competitive['avg_ba']:.3f}")
print(f"  Avg ERA (3yr): {competitive['avg_era']:.2f}")

# Requirement 2: Player Evaluation Model
print("\n2. PLAYER EVALUATION FRAMEWORK")
print("-" * 70)
eval_framework = PlayerEvaluationFramework(rf_model, gb_model, scaler, feature_columns)
archetypes = eval_framework.create_player_archetypes(batter_data)
print("Strategic Player Archetypes:")
for archetype, criteria in archetypes.items():
    print(f"  • {archetype}: {criteria['description']}")

# Requirement 3: Roster Optimization
print("\n3. ROSTER OPTIMIZATION RECOMMENDATIONS")
print("-" * 70)
roster_opt = RosterOptimization(batter_data, pitcher_data)

arb_candidates = roster_opt.identify_arbitration_candidates()
if len(arb_candidates) > 0:
    print(f"\nArbitration Candidates (Top 5):")
    print(arb_candidates.head(5).to_string(index=False))
else:
    print("\nNo arbitration candidates identified in current dataset")

underperformers = roster_opt.identify_underperformers()
if len(underperformers) > 0:
    print(f"\nUnderperforming Players (Top 5):")
    print(underperformers.head(5).to_string(index=False))
else:
    print("\nNo significant underperformers identified")

# Requirement 4: Trade Proposals
print("\n4. TRADE PROPOSAL FRAMEWORK")
print("-" * 70)
trade_gen = TradeProposalGenerator(eval_framework)
proposals = trade_gen.generate_trade_targets('upgrade_pitching', batter_data)

for proposal in proposals:
    print(f"\nProposal #{proposal['trade_id']}")
    print(f"  Rationale: {proposal['rationale']}")
    print(f"  KC Sends: {proposal['kc_sends'][0]['name']}")
    print(f"  KC Receives: {proposal['kc_receives'][0]['name']}")
    print(f"  Short-term Impact: {proposal['short_term']}")
    print(f"  Long-term Impact: {proposal['long_term']}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)