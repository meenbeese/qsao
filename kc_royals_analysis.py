import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

kc_data = pd.read_csv('./kc_royals.csv')

external_pitchers = pd.read_csv('./Savant Pitcher 2021-2025.csv')

try:
    rf_model = joblib.load('./models/baseball_rf_model.pkl')
    gb_model = joblib.load('./models/baseball_gb_model.pkl')
    scaler = joblib.load('./models/baseball_scaler.pkl')
    feature_columns = joblib.load('./models/feature_columns.pkl')
    print("✓ Pre-trained models loaded successfully")
except Exception as e:
    print(f"Warning: Could not load pre-trained models: {e}")
    print("Proceeding with analysis without ML predictions")
    rf_model = gb_model = scaler = feature_columns = None

print("\n" + "="*70)
print("KC ROYALS 2025 ROSTER ANALYSIS WITH REALISTIC TRADE PROPOSALS")
print("="*70)

# ============================================================================
# DATA PREPARATION: Map external data to KC roster format
# ============================================================================

def prepare_external_pitcher_data(savant_df):
    """Convert Savant pitcher data to KC roster format for comparison"""
    
    # Create player full name
    savant_df['Player'] = savant_df['last_name, first_name'].str.split(', ').apply(
        lambda x: f"{x[1]} {x[0]}" if isinstance(x, list) and len(x) == 2 else ""
    )
    
    # Map Savant columns to KC roster equivalents
    pitcher_roster = pd.DataFrame({
        'Player': savant_df['Player'],
        'player_id': savant_df['player_id'],
        'Age': savant_df['player_age'],
        'Year': savant_df['year'],
        'G': savant_df['p_game'],
        'GS': 0,  # Not directly available, will estimate
        'IP': savant_df['p_formatted_ip'].astype(str).str.replace('.', '', regex=False).astype(float) / 10,  # Convert formatted IP
        'W': savant_df['p_win'],
        'L': savant_df['p_loss'],
        'SV': savant_df['p_save'],
        'BS': savant_df['p_blown_save'],
        'ERA': savant_df['p_era'],
        'WHIP': np.nan,  # Calculate if possible
        'SO': savant_df['strikeout'],
        'BB': savant_df['walk'],
        'K_percent': savant_df['k_percent'],
        'BB_percent': savant_df['bb_percent'],
        'AVG': savant_df['p_opp_batting_avg'],
        'OBP': savant_df['p_opp_on_base_avg'],
        'xBA': savant_df['xba'],
        'xSLG': savant_df['xslg'],
        'wOBA': savant_df['woba'],
        'xwOBA': savant_df['xwoba'],
        'exit_velo': savant_df['exit_velocity_avg'],
        'barrel_rate': savant_df['barrel_batted_rate'],
        'hard_hit_percent': savant_df['hard_hit_percent'],
        'whiff_percent': savant_df['whiff_percent']
    })
    
    # Estimate GS based on games and saves (starters don't get saves)
    pitcher_roster['GS'] = np.where(
        pitcher_roster['SV'] == 0,
        (pitcher_roster['G'] * 0.7).astype(int),  # Estimate starters
        0  # Relievers
    )
    
    # Estimate WAR based on ERA, IP, and K% (simplified formula)
    # WAR approximation: ((League ERA - Player ERA) / 9) * IP/9 + K bonus
    league_era = 4.00
    pitcher_roster['WAR'] = (
        ((league_era - pitcher_roster['ERA']) / 9) * (pitcher_roster['IP'] / 9) +
        (pitcher_roster['K_percent'] - 20) / 100
    ).fillna(0)
    
    # Calculate ERA+ (100 = league average)
    pitcher_roster['ERA+'] = ((league_era / pitcher_roster['ERA']) * 100).fillna(100)
    
    # Filter to most recent year (2024) for current value
    pitcher_roster = pitcher_roster[pitcher_roster['Year'] == pitcher_roster['Year'].max()]
    
    # Remove duplicates (keep best season if multiple entries)
    pitcher_roster = pitcher_roster.sort_values('WAR', ascending=False).drop_duplicates('player_id', keep='first')
    
    return pitcher_roster

print("\nPreparing external pitcher dataset...")
external_roster = prepare_external_pitcher_data(external_pitchers)
print(f"✓ Loaded {len(external_roster)} external pitchers from Savant data")

# Get KC player IDs to filter them out from external candidates
kc_player_names = set(kc_data['Player'].str.lower().str.strip())

# Filter external pitchers (exclude anyone already on KC)
available_pitchers = external_roster[
    ~external_roster['Player'].str.lower().str.strip().isin(kc_player_names)
].copy()

print(f"✓ {len(available_pitchers)} external pitchers available (excluding current KC roster)")

# ============================================================================
# REQUIREMENT 1: TEAM ANALYSIS
# ============================================================================

class KCRoyalsTeamAnalysis:
    """Comprehensive analysis of KC Royals 40-man and 26-man roster"""
    
    def __init__(self, roster_df):
        self.roster = roster_df
        self.current_year = 2025
        
    def analyze_roster_composition(self):
        """Analyze position distribution and pitcher roles"""
        print("\n1. ROSTER COMPOSITION ANALYSIS")
        print("-" * 70)
        
        # Categorize pitchers by role
        starters = self.roster[self.roster['GS'] > 20]
        relievers = self.roster[(self.roster['G'] > self.roster['GS']) & (self.roster['IP'] > 50)]
        closers = self.roster[self.roster['SV'] > 5]
        
        print(f"Starting Pitchers (GS > 20): {len(starters)}")
        print(f"Relief Pitchers (G > GS, IP > 50): {len(relievers)}")
        print(f"Closers (SV > 5): {len(closers)}")
        
        # Calculate team performance metrics
        avg_era = self.roster['ERA'].mean()
        avg_whip = self.roster['WHIP'].mean()
        total_war = self.roster['WAR'].sum()
        
        print(f"\nTeam Pitching Metrics:")
        print(f"  Average ERA: {avg_era:.2f}")
        print(f"  Average WHIP: {avg_whip:.2f}")
        print(f"  Total WAR (Pitching): {total_war:.1f}")
        
        return {
            'starters': len(starters),
            'relievers': len(relievers),
            'closers': len(closers),
            'avg_era': avg_era,
            'avg_whip': avg_whip,
            'total_war': total_war
        }
    
    def identify_core_strengths(self):
        """Identify core roster strengths and competitive advantages"""
        print("\n2. CORE STRENGTHS & COMPETITIVE ADVANTAGES")
        print("-" * 70)
        
        # Top WAR contributors
        top_war = self.roster.nlargest(5, 'WAR')[['Player', 'WAR', 'ERA', 'IP']]
        print("\nTop 5 WAR Contributors:")
        print(top_war.to_string(index=False))
        
        # Best ERA pitchers (min 50 IP)
        qualified = self.roster[self.roster['IP'] > 50]
        best_era = qualified.nsmallest(5, 'ERA')[['Player', 'ERA', 'IP', 'WHIP']]
        print("\nBest ERA Pitchers (min 50 IP):")
        print(best_era.to_string(index=False))
        
        # Efficient pitchers (good ERA+ >100)
        efficient = self.roster[self.roster['ERA+'] > 110]
        print(f"\nEfficient Pitchers (ERA+ > 110): {len(efficient)} pitchers")
        
        return {
            'top_contributors': top_war,
            'best_era_pitchers': best_era,
            'efficient_count': len(efficient)
        }
    
    def identify_weaknesses(self):
        """Identify roster weaknesses and inefficiencies"""
        print("\n3. ROSTER WEAKNESSES & INEFFICIENCIES")
        print("-" * 70)
        
        # Underperforming pitchers (ERA > 5.00, IP > 30)
        underperformers = self.roster[(self.roster['ERA'] > 5.0) & (self.roster['IP'] > 30)]
        print(f"\nUnderperforming Pitchers (ERA > 5.0, IP > 30): {len(underperformers)}")
        if len(underperformers) > 0:
            print(underperformers[['Player', 'ERA', 'IP', 'WAR']].to_string(index=False))
        
        # High ERA+ pitchers (ERA+ < 80)
        inefficient = self.roster[self.roster['ERA+'] < 80]
        print(f"\nInefficient Pitchers (ERA+ < 80): {len(inefficient)}")
        
        # Depth concerns - few relief arms
        relievers = self.roster[(self.roster['G'] > 20) & (self.roster['GS'] < 5)]
        print(f"\nRelief Depth: {len(relievers)} arms with 20+ appearances")
        
        return {
            'underperformers': underperformers,
            'inefficient_count': len(inefficient),
            'relief_depth': len(relievers)
        }
    
    def assess_competitive_window(self):
        """Assess competitive window vs league average"""
        print("\n4. COMPETITIVE WINDOW ASSESSMENT")
        print("-" * 70)
        
        avg_era = self.roster['ERA'].mean()
        avg_war = self.roster['WAR'].mean()
        
        # League average ERA ~ 4.00, WAR per pitcher ~ 1.5
        league_avg_era = 4.00
        league_avg_war = 1.5
        
        era_vs_league = "Better" if avg_era < league_avg_era else "Worse"
        war_vs_league = "Better" if avg_war > league_avg_war else "Worse"
        
        print(f"\nTeam ERA: {avg_era:.2f} ({era_vs_league} than league avg {league_avg_era})")
        print(f"Team Avg WAR: {avg_war:.2f} ({war_vs_league} than league avg {league_avg_war})")
        
        # Determine window
        total_war = self.roster['WAR'].sum()
        if total_war > 20 and avg_era < 3.80:
            window = "CONTENDING"
        elif avg_era > 4.30 or self.roster[self.roster['IP'] > 50].shape[0] < 3:
            window = "REBUILDING"
        else:
            window = "RETOOLING"
        
        print(f"\nCompetitive Window: {window}")
        print(f"Total Pitching WAR: {total_war:.1f}")
        
        return {
            'window': window,
            'total_war': total_war,
            'avg_era': avg_era,
            'avg_war': avg_war
        }

# ============================================================================
# REQUIREMENT 2: PLAYER EVALUATION FRAMEWORK
# ============================================================================

class PlayerEvaluationModel:
    """Analytics-based framework for player evaluation"""
    
    def __init__(self, roster_df, rf_model=None, gb_model=None, scaler=None, feature_cols=None):
        self.roster = roster_df
        self.rf_model = rf_model
        self.gb_model = gb_model
        self.scaler = scaler
        self.feature_cols = feature_cols
        
    def create_pitcher_archetypes(self):
        """Define pitcher archetypes aligned with team needs"""
        print("\n5. PITCHER ARCHETYPES & EVALUATION FRAMEWORK")
        print("-" * 70)
        
        archetypes = {
            'Ace': {
                'min_war': 4.0,
                'max_era': 3.00,
                'min_ip': 150,
                'description': 'Elite starter, 150+ IP, ERA < 3.00'
            },
            'Reliable_Starter': {
                'min_war': 2.0,
                'max_era': 3.80,
                'min_ip': 100,
                'description': 'Quality starter, 100-150 IP, ERA < 3.80'
            },
            'High_Volume_Reliever': {
                'min_war': 1.5,
                'max_era': 3.50,
                'min_ip': 50,
                'description': 'Workhorse reliever, 50+ IP'
            },
            'Closer': {
                'min_sv': 5,
                'max_era': 3.30,
                'description': 'Saves leader, sub-3.30 ERA'
            },
            'Young_Prospect': {
                'max_age': 26,
                'description': 'Prospect with upside (age <= 26)'
            }
        }
        
        print("\nPitcher Archetypes:")
        for archetype, criteria in archetypes.items():
            print(f"\n{archetype}:")
            for key, val in criteria.items():
                if key != 'description':
                    print(f"  {key}: {val}")
            print(f"  {criteria['description']}")
        
        return archetypes
    
    def score_players_with_ml(self):
        """Use pre-trained models to generate player value scores"""
        if self.rf_model is None or self.scaler is None:
            print("\nML models not available - skipping ML-based scoring")
            return self.roster
        
        print("\n6A. ML-BASED PLAYER SCORING")
        print("-" * 70)
        
        try:
            # Filter to available features
            available_cols = [col for col in self.feature_cols if col in self.roster.columns]
            
            if len(available_cols) == 0:
                print("Warning: No matching feature columns found in roster data")
                return self.roster
            
            X_roster = self.roster[available_cols].fillna(self.roster[available_cols].mean())
            
            # Scale features
            X_scaled = self.scaler.transform(X_roster)
            
            # Get predictions from both models
            rf_scores = self.rf_model.predict(X_scaled)
            gb_scores = self.gb_model.predict(X_scaled)
            
            # Ensemble prediction (average of both models)
            ensemble_scores = (rf_scores + gb_scores) / 2
            
            # Add scores to roster
            self.roster['ML_RF_Score'] = rf_scores
            self.roster['ML_GB_Score'] = gb_scores
            self.roster['ML_Ensemble_Score'] = ensemble_scores
            
            # Display top performers by ML score
            print("\nTop 10 Players by ML Ensemble Score:")
            top_ml = self.roster.nlargest(10, 'ML_Ensemble_Score')[['Player', 'Age', 'WAR', 'ERA', 'ML_Ensemble_Score']]
            print(top_ml.to_string(index=False))
            
            print("\nBottom 5 Players by ML Ensemble Score:")
            bottom_ml = self.roster.nsmallest(5, 'ML_Ensemble_Score')[['Player', 'Age', 'WAR', 'ERA', 'ML_Ensemble_Score']]
            print(bottom_ml.to_string(index=False))
            
            return self.roster
            
        except Exception as e:
            print(f"Warning: Error during ML scoring: {e}")
            print("Proceeding with analysis without ML scores")
            return self.roster
    
    def assign_player_archetypes(self, archetypes):
        """Assign players to archetypes"""
        print("\n6. PLAYER ARCHETYPE ASSIGNMENTS")
        print("-" * 70)
        
        self.roster['Archetype'] = 'Depth Piece'
        
        # Ace classification
        aces = self.roster[
            (self.roster['WAR'] >= archetypes['Ace']['min_war']) &
            (self.roster['ERA'] <= archetypes['Ace']['max_era']) &
            (self.roster['IP'] >= archetypes['Ace']['min_ip'])
        ]
        self.roster.loc[aces.index, 'Archetype'] = 'Ace'
        
        # Reliable Starter
        starters = self.roster[
            (self.roster['Archetype'] == 'Depth Piece') &
            (self.roster['WAR'] >= archetypes['Reliable_Starter']['min_war']) &
            (self.roster['ERA'] <= archetypes['Reliable_Starter']['max_era']) &
            (self.roster['IP'] >= archetypes['Reliable_Starter']['min_ip'])
        ]
        self.roster.loc[starters.index, 'Archetype'] = 'Reliable_Starter'
        
        # Closer
        closers = self.roster[
            (self.roster['Archetype'] == 'Depth Piece') &
            (self.roster['SV'] >= archetypes['Closer']['min_sv']) &
            (self.roster['ERA'] <= archetypes['Closer']['max_era'])
        ]
        self.roster.loc[closers.index, 'Archetype'] = 'Closer'
        
        # High Volume Reliever
        relievers = self.roster[
            (self.roster['Archetype'] == 'Depth Piece') &
            (self.roster['WAR'] >= archetypes['High_Volume_Reliever']['min_war']) &
            (self.roster['ERA'] <= archetypes['High_Volume_Reliever']['max_era']) &
            (self.roster['IP'] >= archetypes['High_Volume_Reliever']['min_ip'])
        ]
        self.roster.loc[relievers.index, 'Archetype'] = 'High_Volume_Reliever'
        
        # Young Prospect
        prospects = self.roster[
            (self.roster['Archetype'] == 'Depth Piece') &
            (self.roster['Age'] <= archetypes['Young_Prospect']['max_age'])
        ]
        self.roster.loc[prospects.index, 'Archetype'] = 'Young_Prospect'
        
        # Display assignments
        for archetype in self.roster['Archetype'].unique():
            count = len(self.roster[self.roster['Archetype'] == archetype])
            print(f"\n{archetype}: {count} pitchers")
            subset = self.roster[self.roster['Archetype'] == archetype][['Player', 'Age', 'WAR', 'ERA', 'IP']]
            print(subset.to_string(index=False))
        
        return self.roster

# ============================================================================
# REQUIREMENT 3: ROSTER OPTIMIZATION & CONTRACT DECISIONS
# ============================================================================

class RosterOptimizationDecisions:
    """Internal roster optimization recommendations"""
    
    def __init__(self, roster_df):
        self.roster = roster_df
        
    def identify_arbitration_candidates(self):
        """Identify players approaching arbitration eligibility"""
        print("\n7. ARBITRATION CANDIDATE ANALYSIS")
        print("-" * 70)
        
        # Typically 3-4 years service time = ages 27-30 with established value
        # Look for high WAR, peak performance window
        arb_candidates = self.roster[
            (self.roster['Age'] >= 27) & 
            (self.roster['Age'] <= 30) &
            (self.roster['WAR'] > 0)
        ].sort_values('WAR', ascending=False)
        
        print(f"\nArbitration-Eligible Window (Age 27-30, WAR > 0): {len(arb_candidates)} players")
        print("\nRecommendations:")
        
        # Early extension candidates (high WAR, younger)
        early_ext = arb_candidates[(arb_candidates['Age'] <= 28) & (arb_candidates['WAR'] > 2)]
        print(f"\n  EXTEND EARLY (high upside, avoid arb): {len(early_ext)}")
        if len(early_ext) > 0:
            print(early_ext[['Player', 'Age', 'WAR', 'W', 'L']].to_string(index=False))
        
        # Trade at peak value (older, high WAR)
        peak_trade = arb_candidates[(arb_candidates['Age'] >= 28) & (arb_candidates['WAR'] > 2)]
        print(f"\n  TRADE AT PEAK (age/cost curve): {len(peak_trade)}")
        if len(peak_trade) > 0:
            print(peak_trade[['Player', 'Age', 'WAR', 'ERA']].to_string(index=False))
        
        return {
            'arb_eligible': len(arb_candidates),
            'early_extension': early_ext,
            'peak_trade': peak_trade
        }
    
    def evaluate_underperformers(self):
        """Evaluate underperforming players for roster decisions"""
        print("\n8. UNDERPERFORMING PLAYER ANALYSIS")
        print("-" * 70)
        
        # Negative WAR, high ERA, or both
        underperformers = self.roster[
            (self.roster['WAR'] < 0) | 
            ((self.roster['ERA'] > 5.0) & (self.roster['IP'] > 30))
        ]
        
        print(f"\nUnderperformers (WAR < 0 OR ERA > 5.0 with 30+ IP): {len(underperformers)}")
        
        dfa = pd.DataFrame()
        demote = pd.DataFrame()
        
        if len(underperformers) > 0:
            print("\nRecommendations:")
            
            # DFA candidates
            dfa = underperformers[(underperformers['WAR'] < -0.5) & (underperformers['IP'] < 50)]
            print(f"\n  LIKELY DFA (negative WAR, limited IP): {len(dfa)}")
            if len(dfa) > 0:
                print(dfa[['Player', 'WAR', 'ERA', 'IP']].to_string(index=False))
            
            # Demote to minors
            demote = underperformers[
                (underperformers['WAR'] < 0) & 
                (underperformers['IP'] > 30)
            ]
            print(f"\n  DEMOTE TO MINORS (development needed): {len(demote)}")
            if len(demote) > 0:
                print(demote[['Player', 'WAR', 'ERA', 'IP']].to_string(index=False))
        
        return {
            'underperformers': underperformers,
            'dfa_candidates': dfa,
            'demote_candidates': demote,
            'dfa_count': len(dfa)
        }
    
    def assess_prospect_promotion(self):
        """Assess top prospects for promotion"""
        print("\n9. PROSPECT PROMOTION ANALYSIS")
        print("-" * 70)
        
        # Young prospects with positive WAR
        prospects = self.roster[
            (self.roster['Age'] <= 26) &
            (self.roster['WAR'] > 1) &
            (self.roster['IP'] >= 50)
        ].sort_values('WAR', ascending=False)
        
        print(f"\nPromotable Prospects (Age <= 26, WAR > 1, IP >= 50): {len(prospects)}")
        
        if len(prospects) > 0:
            print("\nPromote to MLB (immediate impact):")
            promote = prospects[prospects['WAR'] > 2.5]
            print(promote[['Player', 'Age', 'WAR', 'ERA']].to_string(index=False))
            
            print("\nContinue development (AAA/AA):")
            develop = prospects[prospects['WAR'] <= 2.5]
            print(develop[['Player', 'Age', 'WAR', 'ERA']].to_string(index=False))
        
        return {
            'promotable': len(prospects),
            'promote_mlb': len(prospects[prospects['WAR'] > 2.5]) if len(prospects) > 0 else 0
        }

# ============================================================================
# REQUIREMENT 4: REALISTIC TRADE VALUATION SYSTEM
# ============================================================================

class RealisticTradeValuation:
    """Calculate fair market value for trades with realism scoring"""
    
    def __init__(self, kc_roster_df, external_roster_df):
        self.kc_roster = kc_roster_df
        self.external_roster = external_roster_df
        
    def calculate_trade_value(self, player_row):
        """Calculate a player's trade value based on multiple factors"""
        
        # Base value from WAR
        war_value = player_row['WAR'] * 10
        
        # Age adjustment (prime years 25-29 are most valuable)
        age = player_row['Age']
        if 25 <= age <= 29:
            age_multiplier = 1.2
        elif age < 25:
            age_multiplier = 1.1  # Upside potential
        elif 30 <= age <= 32:
            age_multiplier = 0.9
        else:
            age_multiplier = 0.7  # Declining years
        
        # Performance quality adjustment
        era = player_row['ERA']
        if era < 3.00:
            perf_multiplier = 1.3
        elif era < 3.50:
            perf_multiplier = 1.1
        elif era < 4.00:
            perf_multiplier = 1.0
        elif era < 5.00:
            perf_multiplier = 0.8
        else:
            perf_multiplier = 0.5
        
        # Innings pitched (durability)
        ip = player_row['IP']
        if ip >= 180:
            ip_multiplier = 1.2
        elif ip >= 150:
            ip_multiplier = 1.1
        elif ip >= 100:
            ip_multiplier = 1.0
        elif ip >= 50:
            ip_multiplier = 0.9
        else:
            ip_multiplier = 0.7
        
        # Calculate total trade value
        trade_value = war_value * age_multiplier * perf_multiplier * ip_multiplier
        
        # Floor at 0 (can't have negative trade value)
        trade_value = max(0, trade_value)
        
        return trade_value
    
    def calculate_realism_score(self, kc_package_value, external_package_value):
        """
        Calculate how realistic a trade is based on value balance
        Returns: (realism_percentage, evaluation_string)
        """
        
        # Calculate value difference
        value_diff = abs(kc_package_value - external_package_value)
        avg_value = (kc_package_value + external_package_value) / 2
        
        if avg_value == 0:
            return 0, "UNREALISTIC - No value"
        
        # Calculate percentage difference
        pct_diff = (value_diff / avg_value) * 100
        
        # Realism scoring
        if pct_diff <= 10:
            realism = 95
            evaluation = "HIGHLY REALISTIC - Near-perfect value match"
        elif pct_diff <= 20:
            realism = 85
            evaluation = "VERY REALISTIC - Balanced trade"
        elif pct_diff <= 30:
            realism = 70
            evaluation = "REALISTIC - Slightly unbalanced but acceptable"
        elif pct_diff <= 40:
            realism = 55
            evaluation = "SOMEWHAT REALISTIC - One side gets better value"
        elif pct_diff <= 50:
            realism = 40
            evaluation = "QUESTIONABLE - Significant imbalance"
        else:
            realism = 20
            evaluation = "UNREALISTIC - Major value mismatch"
        
        return realism, evaluation
    
    def find_balanced_trade_package(self, target_player, kc_available_players, max_players=3):
        """
        Find the best combination of KC players to match target player's value
        """
        target_value = self.calculate_trade_value(target_player)
        
        best_package = []
        best_realism = 0
        best_value_diff = float('inf')
        
        # Try single player trades first
        for _, kc_player in kc_available_players.iterrows():
            kc_value = self.calculate_trade_value(kc_player)
            realism, _ = self.calculate_realism_score(kc_value, target_value)
            value_diff = abs(kc_value - target_value)
            
            if realism > best_realism or (realism == best_realism and value_diff < best_value_diff):
                best_realism = realism
                best_package = [kc_player]
                best_value_diff = value_diff
        
        # Try two-player packages if single player isn't realistic enough
        if best_realism < 70 and len(kc_available_players) >= 2:
            for i in range(len(kc_available_players)):
                for j in range(i+1, min(i+10, len(kc_available_players))):  # Limit combinations
                    player1 = kc_available_players.iloc[i]
                    player2 = kc_available_players.iloc[j]
                    
                    package_value = (self.calculate_trade_value(player1) + 
                                   self.calculate_trade_value(player2))
                    realism, _ = self.calculate_realism_score(package_value, target_value)
                    value_diff = abs(package_value - target_value)
                    
                    if realism > best_realism or (realism == best_realism and value_diff < best_value_diff):
                        best_realism = realism
                        best_package = [player1, player2]
                        best_value_diff = value_diff
        
        # Try three-player packages if still not realistic
        if best_realism < 70 and len(kc_available_players) >= 3 and max_players >= 3:
            for i in range(len(kc_available_players)):
                for j in range(i+1, min(i+8, len(kc_available_players))):
                    for k in range(j+1, min(j+5, len(kc_available_players))):
                        player1 = kc_available_players.iloc[i]
                        player2 = kc_available_players.iloc[j]
                        player3 = kc_available_players.iloc[k]
                        
                        package_value = (self.calculate_trade_value(player1) + 
                                       self.calculate_trade_value(player2) +
                                       self.calculate_trade_value(player3))
                        realism, _ = self.calculate_realism_score(package_value, target_value)
                        value_diff = abs(package_value - target_value)
                        
                        if realism > best_realism or (realism == best_realism and value_diff < best_value_diff):
                            best_realism = realism
                            best_package = [player1, player2, player3]
                            best_value_diff = value_diff
        
        return best_package, best_realism

# ============================================================================
# REALISTIC EXTERNAL TRADE PROPOSALS
# ============================================================================

class RealisticTradeProposals:
    """Generate realistic, value-balanced trade proposals"""
    
    def __init__(self, kc_roster_df, external_roster_df, valuation_system):
        self.kc_roster = kc_roster_df
        self.external_roster = external_roster_df
        self.valuation = valuation_system
        
    def find_trade_targets_by_need(self):
        """Identify external trade targets based on KC needs"""
        print("\n10. EXTERNAL TRADE TARGET IDENTIFICATION")
        print("-" * 70)
        
        # Find external ace candidates
        ace_candidates = self.external_roster[
            (self.external_roster['ERA'] < 3.00) &
            (self.external_roster['IP'] >= 150) &
            (self.external_roster['WAR'] >= 4.0)
        ].sort_values('WAR', ascending=False).head(15)
        
        # Calculate trade values
        if len(ace_candidates) > 0:
            ace_candidates['Trade_Value'] = ace_candidates.apply(
                self.valuation.calculate_trade_value, axis=1
            )
        
        print(f"\n  ACE CANDIDATES FROM EXTERNAL POOL: {len(ace_candidates)}")
        if len(ace_candidates) > 0:
            print(ace_candidates[['Player', 'Age', 'WAR', 'ERA', 'IP', 'Trade_Value']].to_string(index=False))
        
        # Find high-leverage relievers
        reliever_candidates = self.external_roster[
            (self.external_roster['ERA'] < 3.00) &
            (self.external_roster['IP'] >= 50) &
            (self.external_roster['IP'] < 100) &
            (self.external_roster['K_percent'] > 25)
        ].sort_values('WAR', ascending=False).head(15)
        
        if len(reliever_candidates) > 0:
            reliever_candidates['Trade_Value'] = reliever_candidates.apply(
                self.valuation.calculate_trade_value, axis=1
            )
        
        print(f"\n  HIGH-LEVERAGE RELIEVER CANDIDATES: {len(reliever_candidates)}")
        if len(reliever_candidates) > 0:
            print(reliever_candidates[['Player', 'Age', 'WAR', 'ERA', 'Trade_Value']].to_string(index=False))
        
        # Find young starters with upside
        young_starter_candidates = self.external_roster[
            (self.external_roster['Age'] < 27) &
            (self.external_roster['IP'] >= 100) &
            (self.external_roster['ERA'] < 4.00) &
            (self.external_roster['WAR'] > 1.5)
        ].sort_values('WAR', ascending=False).head(15)
        
        if len(young_starter_candidates) > 0:
            young_starter_candidates['Trade_Value'] = young_starter_candidates.apply(
                self.valuation.calculate_trade_value, axis=1
            )
        
        print(f"\n  YOUNG STARTER CANDIDATES (Upside): {len(young_starter_candidates)}")
        if len(young_starter_candidates) > 0:
            print(young_starter_candidates[['Player', 'Age', 'WAR', 'ERA', 'Trade_Value']].to_string(index=False))
        
        return {
            'ace_candidates': ace_candidates,
            'reliever_candidates': reliever_candidates,
            'young_starter_candidates': young_starter_candidates
        }
    
    def identify_kc_trade_chips(self):
        """Identify KC players who could be traded with their values"""
        print("\n11. KC ROYALS AVAILABLE TRADE CHIPS (WITH VALUES)")
        print("-" * 70)
        
        # Calculate trade values for all KC players
        self.kc_roster['Trade_Value'] = self.kc_roster.apply(
            self.valuation.calculate_trade_value, axis=1
        )
        
        # Categorize available players
        tradeable_players = self.kc_roster[
            (self.kc_roster['Archetype'].isin(['Depth Piece', 'Young_Prospect', 'High_Volume_Reliever'])) |
            (self.kc_roster['WAR'] < 1.5)
        ].sort_values('Trade_Value', ascending=False)
        
        print(f"\n  TRADEABLE ASSETS: {len(tradeable_players)} players")
        if len(tradeable_players) > 0:
            print(tradeable_players[['Player', 'Age', 'WAR', 'ERA', 'Archetype', 'Trade_Value']].to_string(index=False))
        
        return tradeable_players
    
    def generate_realistic_trade_proposals(self, trade_targets, kc_tradeable):
        """Generate realistic external trade proposals with realism scoring"""
        print("\n12. REALISTIC TRADE PROPOSALS WITH VALUE ANALYSIS")
        print("="*70)
        
        ace_candidates = trade_targets.get('ace_candidates', pd.DataFrame())
        young_starters = trade_targets.get('young_starter_candidates', pd.DataFrame())
        relievers = trade_targets.get('reliever_candidates', pd.DataFrame())
        
        # ----------------------------
        # PROPOSAL #1: Acquire Ace Starter
        # ----------------------------
        print("\n╔═══════════════════════════════════════════════════════════════════════╗")
        print("║ TRADE PROPOSAL #1: ACQUIRE ACE STARTER (VALUE-BALANCED)              ║")
        print("╚═══════════════════════════════════════════════════════════════════════╝")
        
        if len(ace_candidates) > 0 and len(kc_tradeable) > 0:
            # Get best ace target
            target_ace = ace_candidates.iloc[0]
            target_value = target_ace['Trade_Value']
            
            # Find best KC package to match
            kc_package, realism = self.valuation.find_balanced_trade_package(
                target_ace, kc_tradeable, max_players=3
            )
            
            if len(kc_package) > 0:
                print("\n  KC ROYALS SEND OUT:")
                kc_total_value = 0
                for player in kc_package:
                    player_value = self.valuation.calculate_trade_value(player)
                    kc_total_value += player_value
                    print(f"    ❌ {player['Player']:<25} Age: {player['Age']:<3} WAR: {player['WAR']:>5.1f}  "
                          f"ERA: {player['ERA']:>5.2f}  Value: {player_value:>6.1f}")
                print(f"    {'TOTAL KC PACKAGE VALUE:':<44} {kc_total_value:>6.1f}")
                
                print("\n  KC ROYALS RECEIVE:")
                print(f"    ✅ {target_ace['Player']:<25} Age: {target_ace['Age']:<3} WAR: {target_ace['WAR']:>5.1f}  "
                      f"ERA: {target_ace['ERA']:>5.2f}  Value: {target_value:>6.1f}")
                
                # Realism analysis
                realism_pct, realism_eval = self.valuation.calculate_realism_score(
                    kc_total_value, target_value
                )
                
                print(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
                print(f"  ║ REALISM SCORE: {realism_pct}% - {realism_eval:<43}║")
                print(f"  ╚══════════════════════════════════════════════════════════════════╝")
                
                print(f"\n  VALUE ANALYSIS:")
                print(f"    KC Sends:     {kc_total_value:>6.1f} trade value ({len(kc_package)} player{'s' if len(kc_package) > 1 else ''})")
                print(f"    KC Receives:  {target_value:>6.1f} trade value (1 player)")
                print(f"    Difference:   {abs(kc_total_value - target_value):>6.1f} ({abs(kc_total_value - target_value) / target_value * 100:.1f}% gap)")
                
                # Calculate WAR impact
                war_out = sum([p['WAR'] for p in kc_package])
                war_in = target_ace['WAR']
                
                print(f"\n  IMPACT ANALYSIS:")
                print(f"    WAR Change: {war_in - war_out:+.1f} ({war_out:.1f} out → {war_in:.1f} in)")
                print(f"    Roster Spots: {len(kc_package)} → 1 (clears {len(kc_package)-1} spot{'s' if len(kc_package) > 1 else ''})")
                
                print(f"\n  STRATEGIC RATIONALE:")
                if realism_pct >= 70:
                    print(f"    ✓ Fair value trade - both teams benefit")
                    print(f"    ✓ KC addresses rotation need with proven ace")
                    print(f"    ✓ Sending team gets {len(kc_package)} MLB-ready arms")
                else:
                    print(f"    ⚠ Value imbalance may require additional prospect")
                    print(f"    ⚠ Consider adding cash or PTBNL to balance")
        
        print("\n" + "="*70)
        
        # ----------------------------
        # PROPOSAL #2: Youth Infusion
        # ----------------------------
        print("\n╔═══════════════════════════════════════════════════════════════════════╗")
        print("║ TRADE PROPOSAL #2: YOUNG CONTROLLABLE STARTER                        ║")
        print("╚═══════════════════════════════════════════════════════════════════════╝")
        
        if len(young_starters) > 0 and len(kc_tradeable) > 1:
            # Get best young starter
            target_young = young_starters.iloc[0]
            target_value = target_young['Trade_Value']
            
            # Find KC package
            kc_package, realism = self.valuation.find_balanced_trade_package(
                target_young, kc_tradeable, max_players=2
            )
            
            if len(kc_package) > 0:
                print("\n  KC ROYALS SEND OUT:")
                kc_total_value = 0
                for player in kc_package:
                    player_value = self.valuation.calculate_trade_value(player)
                    kc_total_value += player_value
                    print(f"    ❌ {player['Player']:<25} Age: {player['Age']:<3} WAR: {player['WAR']:>5.1f}  "
                          f"ERA: {player['ERA']:>5.2f}  Value: {player_value:>6.1f}")
                print(f"    {'TOTAL KC PACKAGE VALUE:':<44} {kc_total_value:>6.1f}")
                
                print("\n  KC ROYALS RECEIVE:")
                print(f"    ✅ {target_young['Player']:<25} Age: {target_young['Age']:<3} WAR: {target_young['WAR']:>5.1f}  "
                      f"ERA: {target_young['ERA']:>5.2f}  Value: {target_value:>6.1f}")
                
                # Realism analysis
                realism_pct, realism_eval = self.valuation.calculate_realism_score(
                    kc_total_value, target_value
                )
                
                print(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
                print(f"  ║ REALISM SCORE: {realism_pct}% - {realism_eval:<43}║")
                print(f"  ╚══════════════════════════════════════════════════════════════════╝")
                
                print(f"\n  VALUE ANALYSIS:")
                print(f"    KC Sends:     {kc_total_value:>6.1f} trade value")
                print(f"    KC Receives:  {target_value:>6.1f} trade value")
                print(f"    Difference:   {abs(kc_total_value - target_value):>6.1f}")
                
                war_out = sum([p['WAR'] for p in kc_package])
                war_in = target_young['WAR']
                
                print(f"\n  IMPACT ANALYSIS:")
                print(f"    WAR Change: {war_in - war_out:+.1f}")
                print(f"    Age Profile: Avg {sum([p['Age'] for p in kc_package])/len(kc_package):.1f} → {target_young['Age']} years")
                print(f"    Control: Limited → 4-6 years team control")
        
        print("\n" + "="*70)
        
        # ----------------------------
        # PROPOSAL #3: Bullpen Upgrade
        # ----------------------------
        print("\n╔═══════════════════════════════════════════════════════════════════════╗")
        print("║ TRADE PROPOSAL #3: HIGH-LEVERAGE RELIEVER                            ║")
        print("╚═══════════════════════════════════════════════════════════════════════╝")
        
        if len(relievers) > 0 and len(kc_tradeable) > 2:
            # Get best reliever
            target_reliever = relievers.iloc[0]
            target_value = target_reliever['Trade_Value']
            
            # Find KC package
            kc_package, realism = self.valuation.find_balanced_trade_package(
                target_reliever, kc_tradeable, max_players=2
            )
            
            if len(kc_package) > 0:
                print("\n  KC ROYALS SEND OUT:")
                kc_total_value = 0
                for player in kc_package:
                    player_value = self.valuation.calculate_trade_value(player)
                    kc_total_value += player_value
                    print(f"    ❌ {player['Player']:<25} Age: {player['Age']:<3} WAR: {player['WAR']:>5.1f}  "
                          f"ERA: {player['ERA']:>5.2f}  Value: {player_value:>6.1f}")
                print(f"    {'TOTAL KC PACKAGE VALUE:':<44} {kc_total_value:>6.1f}")
                
                print("\n  KC ROYALS RECEIVE:")
                print(f"    ✅ {target_reliever['Player']:<25} Age: {target_reliever['Age']:<3} WAR: {target_reliever['WAR']:>5.1f}  "
                      f"ERA: {target_reliever['ERA']:>5.2f}  Value: {target_value:>6.1f}")
                
                # Realism analysis
                realism_pct, realism_eval = self.valuation.calculate_realism_score(
                    kc_total_value, target_value
                )
                
                print(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
                print(f"  ║ REALISM SCORE: {realism_pct}% - {realism_eval:<43}║")
                print(f"  ╚══════════════════════════════════════════════════════════════════╝")
                
                print(f"\n  VALUE ANALYSIS:")
                print(f"    KC Sends:     {kc_total_value:>6.1f} trade value")
                print(f"    KC Receives:  {target_value:>6.1f} trade value")
                print(f"    Difference:   {abs(kc_total_value - target_value):>6.1f}")

# ============================================================================
# EXECUTE FULL ANALYSIS
# ============================================================================

# Run team analysis
team = KCRoyalsTeamAnalysis(kc_data)
composition = team.analyze_roster_composition()
strengths = team.identify_core_strengths()
weaknesses = team.identify_weaknesses()
window = team.assess_competitive_window()

# Run player evaluation
eval_model = PlayerEvaluationModel(kc_data, rf_model, gb_model, scaler, feature_columns)
archetypes = eval_model.create_pitcher_archetypes()
kc_data_assigned = eval_model.assign_player_archetypes(archetypes)

# Score players with ML models
if rf_model is not None:
    kc_data_assigned = eval_model.score_players_with_ml()
    if kc_data_assigned is None:
        kc_data_assigned = eval_model.roster

# Run roster optimization
roster_opt = RosterOptimizationDecisions(kc_data_assigned)
arb_analysis = roster_opt.identify_arbitration_candidates()
underperf_analysis = roster_opt.evaluate_underperformers()
prospect_analysis = roster_opt.assess_prospect_promotion()

# Run REALISTIC trade analysis with valuation system
valuation_system = RealisticTradeValuation(kc_data_assigned, available_pitchers)
realistic_trades = RealisticTradeProposals(kc_data_assigned, available_pitchers, valuation_system)
trade_targets = realistic_trades.find_trade_targets_by_need()
kc_tradeable = realistic_trades.identify_kc_trade_chips()
realistic_trades.generate_realistic_trade_proposals(trade_targets, kc_tradeable)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("KC ROYALS ANALYSIS COMPLETE - REALISTIC TRADE EDITION")
print("="*70)
print("\nEXECUTIVE SUMMARY:")
print(f"  Competitive Window: {window['window']}")
print(f"  Team ERA: {window['avg_era']:.2f}")
print(f"  Total Pitching WAR: {window['total_war']:.1f}")
print(f"\n  External Ace Targets: {len(trade_targets.get('ace_candidates', []))}")
print(f"  External Young Starters: {len(trade_targets.get('young_starter_candidates', []))}")
print(f"  External Relievers: {len(trade_targets.get('reliever_candidates', []))}")
print(f"\n  KC Tradeable Assets: {len(kc_tradeable)}")

print("\n" + "="*70)
print("TRADE VALUE SYSTEM FEATURES:")
print("  ✓ WAR-based valuation")
print("  ✓ Age curve adjustments")
print("  ✓ Performance quality multipliers")
print("  ✓ Durability (IP) considerations")
print("  ✓ Automatic package balancing (1-3 players)")
print("  ✓ Realism scoring (0-100%)")
print("="*70)