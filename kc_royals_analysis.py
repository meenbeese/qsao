import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load KC Royals roster data
kc_data = pd.read_csv('./kc_royals.csv')

# Load pre-trained models from analyze_baseball_data.py
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
print("KC ROYALS 2025 ROSTER ANALYSIS")
print("="*70)

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
            'dfa_count': len(dfa) if len(underperformers) > 0 else 0
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
# REQUIREMENT 4: TRADE PROPOSAL FRAMEWORK
# ============================================================================

class TradeProposalAnalysis:
    """Generate realistic, analytics-backed trade proposals"""
    
    def __init__(self, roster_df):
        self.roster = roster_df
        
    def generate_trade_proposals(self):
        """Generate two realistic trade proposals"""
        print("\n10. TRADE PROPOSAL FRAMEWORK")
        print("-" * 70)
        
        print("\nPROPOSAL #1: ACQUIRE ACE STARTER")
        print("  Rationale: Address rotation weakness (WAR/ERA gap)")
        print("\n  KC SENDS OUT:")
        print("    • Underperforming reliever (negative WAR, expendable)")
        print("    • Young prospect with limited upside")
        
        underperf = self.roster[(self.roster['WAR'] < 0) & (self.roster['IP'] < 50)]
        if len(underperf) > 0:
            print(f"\n    Example: {underperf.iloc[0]['Player']} (WAR: {underperf.iloc[0]['WAR']:.1f})")
        
        print("\n  KC RECEIVES:")
        print("    • Established SP (2.5+ WAR, sub-4.00 ERA)")
        print("    • Fills immediate rotation need")
        
        print("\n  IMPACT:")
        print("    Short-term: +1.5-2.0 WAR improvement, better ERA leadership")
        print("    Long-term: Competitive rotation; premium salary obligation")
        
        print("\n" + "="*70)
        print("\nPROPOSAL #2: YOUTH INFUSION + SALARY RELIEF")
        print("  Rationale: Move veteran reliever at peak value, acquire young SP")
        print("\n  KC SENDS OUT:")
        print("    • High-value reliever (2+ WAR, sub-3.30 ERA)")
        
        good_relievers = self.roster[
            (self.roster['G'] > 20) & 
            (self.roster['GS'] < 5) &
            (self.roster['WAR'] > 1.5)
        ]
        if len(good_relievers) > 0:
            print(f"    Example: {good_relievers.iloc[0]['Player']} (WAR: {good_relievers.iloc[0]['WAR']:.1f})")
        
        print("\n  KC RECEIVES:")
        print("    • Young starter (age 25-27, high upside)")
        print("    • Cost-controlled for 2-3 years")
        
        print("\n  IMPACT:")
        print("    Short-term: -0.5 WAR (lose established arm)")
        print("    Long-term: +1.5-2.0 WAR (prospect development), $5M+ savings")

# ============================================================================
# EXECUTE FULL ANALYSIS
# ============================================================================

# Run all analyses
team = KCRoyalsTeamAnalysis(kc_data)
composition = team.analyze_roster_composition()
strengths = team.identify_core_strengths()
weaknesses = team.identify_weaknesses()
window = team.assess_competitive_window()

eval_model = PlayerEvaluationModel(kc_data, rf_model, gb_model, scaler, feature_columns)
archetypes = eval_model.create_pitcher_archetypes()
kc_data_assigned = eval_model.assign_player_archetypes(archetypes)

# Score players with ML models - ensure roster is preserved
if rf_model is not None:
    kc_data_assigned = eval_model.score_players_with_ml()
    if kc_data_assigned is None:
        kc_data_assigned = eval_model.roster

roster_opt = RosterOptimizationDecisions(kc_data_assigned)
arb_analysis = roster_opt.identify_arbitration_candidates()
underperf_analysis = roster_opt.evaluate_underperformers()
prospect_analysis = roster_opt.assess_prospect_promotion()

trades = TradeProposalAnalysis(kc_data_assigned)
trades.generate_trade_proposals()

print("\n" + "="*70)
print("KC ROYALS ANALYSIS COMPLETE")
print("="*70)
print("\nSUMMARY:")
print(f"  Competitive Window: {window['window']}")
print(f"  Team ERA: {window['avg_era']:.2f}")
print(f"  Total Pitching WAR: {window['total_war']:.1f}")
print(f"  Arb Candidates: {arb_analysis['arb_eligible']}")
print(f"  Underperformers: {len(underperf_analysis['underperformers'])}")
print(f"  Promotable Prospects: {prospect_analysis['promotable']}")
