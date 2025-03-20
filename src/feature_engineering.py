import numpy as np
import pandas as pd
import re

def get_gender_from_teamid(team_id):
    """
    Determine the gender of the team based on TeamID
    
    Args:
        team_id: Team ID
        
    Returns:
        'M' for men's teams (ID 1000-1999)
        'W' for women's teams (ID 3000-3999)
    """
    try:
        team_id = int(team_id) if team_id is not None else None
        
        if team_id is None:
            return None
        elif 1000 <= team_id < 2000:
            return 'M'
        elif 3000 <= team_id < 4000:
            return 'W'
        else:
            # Handle unexpected team ID ranges
            print(f"Warning: Team ID {team_id} outside expected ranges (1000-1999 for men, 3000-3999 for women)")
            # Default to men's team for IDs below 3000, women's for IDs 3000+
            return 'M' if team_id < 3000 else 'W'
    except (ValueError, TypeError) as e:
        # Handle conversion errors for non-integer team_id
        print(f"Error determining gender for team_id '{team_id}': {e}")
        return None

def feature_engineering(games, seed_dict):
    """
    Basic feature engineering:
    - Generate game IDs and order teams.
    - Map seed values and calculate seed differences and strengths.
    - Generate target variable.
    
    Args:
        games: DataFrame with game data
        seed_dict: Dictionary mapping Season_TeamID to seed value
        
    Returns:
        DataFrame with engineered features
    """
    print("Performing basic feature engineering...")
    
    # Generate game identifier and team order (sorted by TeamID)
    games['ID'] = games.apply(lambda r: '_'.join([str(r['Season'])] +
                                                 list(map(str, sorted([r['WTeamID'], r['LTeamID']])))),
                              axis=1)
    games['IDTeams'] = games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'], r['LTeamID']]))),
                                   axis=1)
    games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'], r['LTeamID']])[0], axis=1)
    games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'], r['LTeamID']])[1], axis=1)
    games['IDTeam1'] = games.apply(lambda r: '_'.join([str(r['Season']), str(r['Team1'])]), axis=1)
    games['IDTeam2'] = games.apply(lambda r: '_'.join([str(r['Season']), str(r['Team2'])]), axis=1)
    
    # Add gender features based on TeamID
    games['Team1Gender'] = games['Team1'].apply(get_gender_from_teamid)
    games['Team2Gender'] = games['Team2'].apply(get_gender_from_teamid)
    
    # Ensure the gender is the same for both teams (validation check)
    gender_mismatch = games[games['Team1Gender'] != games['Team2Gender']]
    if not gender_mismatch.empty:
        print(f"Warning: {len(gender_mismatch)} games have mismatched genders between teams.")
        # For debugging purposes, show some examples
        if len(gender_mismatch) > 0:
            print("Sample of gender mismatches:")
            print(gender_mismatch[['Season', 'Team1', 'Team2', 'Team1Gender', 'Team2Gender']].head(3))
    
    # Add a single gender column for the game (since both teams should have the same gender)
    games['Gender'] = games['Team1Gender']
    
    # Convert string gender to numeric values (1 for men, 0 for women)
    games['GenderCode'] = games['Gender'].map({'M': 1, 'W': 0}).fillna(0.5)  # Default 0.5 if unknown
    
    # Drop the string gender columns as they can't be used by the model directly
    games = games.drop(columns=['Team1Gender', 'Team2Gender', 'Gender'])
    
    # Map seed values and calculate seed differences
    games['Team1Seed'] = games['IDTeam1'].map(seed_dict).fillna(16)
    games['Team2Seed'] = games['IDTeam2'].map(seed_dict).fillna(16)
    games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed']
    
    # Seed strength features (exponentially decaying)
    games['Team1SeedStrength'] = np.exp(-games['Team1Seed'] / 4)
    games['Team2SeedStrength'] = np.exp(-games['Team2Seed'] / 4)
    games['SeedStrengthDiff'] = games['Team1SeedStrength'] - games['Team2SeedStrength']
    
    # Generate target variable WinA (1 if Team1 wins, else 0)
    games['WinA'] = games.apply(lambda r: 1 if sorted([r['WTeamID'], r['LTeamID']])[0] == r['WTeamID'] else 0, axis=1)
    
    if 'WScore' in games.columns and 'LScore' in games.columns:
        games['ScoreDiff'] = games['WScore'] - games['LScore']
        games['ScoreDiffNorm'] = games.apply(lambda r: r['ScoreDiff'] if r['WinA'] == 1 else -r['ScoreDiff'], axis=1)
    
    return games

def add_team_features(games, current_season=None, num_prev_tournaments=3):
    """
    Add team statistics based on regular season games only, using Season_TeamID as key.
    This avoids data leakage by ensuring tournament data is not used for feature generation.
    
    Args:
        games: DataFrame with game data. Must contain 'Season', 'DayNum', and 'GameType' columns.
        current_season: Current season year (if None, use max Season in games)
        num_prev_tournaments: Not used here; kept for interface consistency.
    
    Returns:
        Tuple: (DataFrame with added team features, team_stats_cum dictionary)
    """
    print("Calculating team statistics features (regular season data only, to avoid data leakage)...")
    if current_season is None:
        current_season = games['Season'].max()
    
    # Initialize team statistics with default values
    games['Team1_WinRate'] = 0.5
    games['Team1_AvgScore'] = 60
    games['Team1_AvgAllowed'] = 60
    games['Team1_PointDiff'] = 0
    games['Team1_GamesPlayed'] = 0
    games['Team1_HomeWinRate'] = 0.5
    games['Team1_AwayWinRate'] = 0.5
    games['Team1_FGPct'] = 0.45    # 投篮命中率
    games['Team1_3PPct'] = 0.33    # 三分命中率
    games['Team1_FTPct'] = 0.7     # 罚球命中率
    
    games['Team2_WinRate'] = 0.5
    games['Team2_AvgScore'] = 60
    games['Team2_AvgAllowed'] = 60
    games['Team2_PointDiff'] = 0
    games['Team2_GamesPlayed'] = 0
    games['Team2_HomeWinRate'] = 0.5
    games['Team2_AwayWinRate'] = 0.5
    games['Team2_FGPct'] = 0.45
    games['Team2_3PPct'] = 0.33
    games['Team2_FTPct'] = 0.7
    
    if 'GameType' not in games.columns:
        print("GameType column not found, cannot distinguish between regular season and tournament. Adding default GameType column...")
        games['GameType'] = 'Regular'  # Default to regular season
    
    # Create a copy with only regular season data
    regular_games = games[games['GameType'] == 'Regular'].copy()
    
    if regular_games.empty:
        print("No regular season data found, using all data for team statistics...")
        regular_games = games.copy()
    
    # Sort by Season and DayNum to ensure correct order
    regular_games = regular_games.sort_values(by=['Season', 'DayNum']).reset_index(drop=True)
    
    # Create team statistics dictionary using Season_TeamID as key
    team_stats = {}
    
    # Calculate statistics for each team
    for season in regular_games['Season'].unique():
        season_games = regular_games[regular_games['Season'] == season]
        
        # Process winning team statistics
        for _, row in season_games.iterrows():
            team_id = row['WTeamID']
            opponent_id = row['LTeamID']
            key = f"{season}_{team_id}"
            
            if key not in team_stats:
                team_stats[key] = {
                    'wins': 0, 'losses': 0, 'points_for': 0, 'points_against': 0,
                    'home_games': 0, 'home_wins': 0, 'away_games': 0, 'away_wins': 0,
                    'fg_made': 0, 'fg_att': 0, 'fg3_made': 0, 'fg3_att': 0, 
                    'ft_made': 0, 'ft_att': 0
                }
            
            team_stats[key]['wins'] += 1
            team_stats[key]['points_for'] += row['WScore'] if 'WScore' in row else 0
            team_stats[key]['points_against'] += row['LScore'] if 'LScore' in row else 0
            
            # Add home/away stats if available
            if 'WLoc' in row:
                if row['WLoc'] == 'H':
                    team_stats[key]['home_games'] += 1
                    team_stats[key]['home_wins'] += 1
                elif row['WLoc'] == 'A':
                    team_stats[key]['away_games'] += 1
                    team_stats[key]['away_wins'] += 1
            
            # Add shooting stats if available
            for stat, col in [('fg_made', 'WFGM'), ('fg_att', 'WFGA'), 
                             ('fg3_made', 'WFGM3'), ('fg3_att', 'WFGA3'),
                             ('ft_made', 'WFTM'), ('ft_att', 'WFTA')]:
                if col in row:
                    team_stats[key][stat] += row[col]
            
            # Process losing team statistics
            opp_key = f"{season}_{opponent_id}"
            if opp_key not in team_stats:
                team_stats[opp_key] = {
                    'wins': 0, 'losses': 0, 'points_for': 0, 'points_against': 0,
                    'home_games': 0, 'home_wins': 0, 'away_games': 0, 'away_wins': 0,
                    'fg_made': 0, 'fg_att': 0, 'fg3_made': 0, 'fg3_att': 0, 
                    'ft_made': 0, 'ft_att': 0
                }
            
            team_stats[opp_key]['losses'] += 1
            team_stats[opp_key]['points_for'] += row['LScore'] if 'LScore' in row else 0
            team_stats[opp_key]['points_against'] += row['WScore'] if 'WScore' in row else 0
            
            # Add home/away stats if available
            if 'WLoc' in row:
                if row['WLoc'] == 'A':
                    team_stats[opp_key]['home_games'] += 1
                elif row['WLoc'] == 'H':
                    team_stats[opp_key]['away_games'] += 1
            
            # Add shooting stats if available
            for stat, col in [('fg_made', 'LFGM'), ('fg_att', 'LFGA'), 
                             ('fg3_made', 'LFGM3'), ('fg3_att', 'LFGA3'),
                             ('ft_made', 'LFTM'), ('ft_att', 'LFTA')]:
                if col in row:
                    team_stats[opp_key][stat] += row[col]
    
    # Calculate derived statistics
    for key in team_stats:
        stats = team_stats[key]
        games_played = stats['wins'] + stats['losses']
        if games_played > 0:
            stats['win_rate'] = stats['wins'] / games_played
            stats['avg_score'] = stats['points_for'] / games_played
            stats['avg_allowed'] = stats['points_against'] / games_played
            stats['point_diff'] = stats['avg_score'] - stats['avg_allowed']
            stats['games_played'] = games_played
            
            # Home/Away win rates
            stats['home_win_rate'] = stats['home_wins'] / stats['home_games'] if stats['home_games'] > 0 else 0.5
            stats['away_win_rate'] = stats['away_wins'] / stats['away_games'] if stats['away_games'] > 0 else 0.5
            
            # Shooting percentages
            stats['fg_pct'] = stats['fg_made'] / stats['fg_att'] if stats['fg_att'] > 0 else 0.45
            stats['fg3_pct'] = stats['fg3_made'] / stats['fg3_att'] if stats['fg3_att'] > 0 else 0.33
            stats['ft_pct'] = stats['ft_made'] / stats['ft_att'] if stats['ft_att'] > 0 else 0.7
        else:
            stats['win_rate'] = 0.5
            stats['avg_score'] = 60
            stats['avg_allowed'] = 60
            stats['point_diff'] = 0
            stats['games_played'] = 0
            stats['home_win_rate'] = 0.5
            stats['away_win_rate'] = 0.5
            stats['fg_pct'] = 0.45
            stats['fg3_pct'] = 0.33
            stats['ft_pct'] = 0.7
    
    # Map the team statistics back to the original dataframe
    games['Team1_WinRate'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('win_rate', 0.5))
    games['Team1_AvgScore'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('avg_score', 60))
    games['Team1_AvgAllowed'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('avg_allowed', 60))
    games['Team1_PointDiff'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('point_diff', 0))
    games['Team1_GamesPlayed'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('games_played', 0))
    games['Team1_HomeWinRate'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('home_win_rate', 0.5))
    games['Team1_AwayWinRate'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('away_win_rate', 0.5))
    games['Team1_FGPct'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('fg_pct', 0.45))
    games['Team1_3PPct'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('fg3_pct', 0.33))
    games['Team1_FTPct'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('ft_pct', 0.7))
    
    games['Team2_WinRate'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('win_rate', 0.5))
    games['Team2_AvgScore'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('avg_score', 60))
    games['Team2_AvgAllowed'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('avg_allowed', 60))
    games['Team2_PointDiff'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('point_diff', 0))
    games['Team2_GamesPlayed'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('games_played', 0))
    games['Team2_HomeWinRate'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('home_win_rate', 0.5))
    games['Team2_AwayWinRate'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('away_win_rate', 0.5))
    games['Team2_FGPct'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('fg_pct', 0.45))
    games['Team2_3PPct'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('fg3_pct', 0.33))
    games['Team2_FTPct'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('ft_pct', 0.7))
    
    # Compute difference features
    games['WinRateDiff'] = games['Team1_WinRate'] - games['Team2_WinRate']
    games['AvgScoreDiff'] = games['Team1_AvgScore'] - games['Team2_AvgScore']
    games['AvgAllowedDiff'] = games['Team1_AvgAllowed'] - games['Team2_AvgAllowed']
    games['PointDiffDiff'] = games['Team1_PointDiff'] - games['Team2_PointDiff']
    games['GamesPlayedDiff'] = games['Team1_GamesPlayed'] - games['Team2_GamesPlayed']
    games['HomeWinRateDiff'] = games['Team1_HomeWinRate'] - games['Team2_HomeWinRate']
    games['AwayWinRateDiff'] = games['Team1_AwayWinRate'] - games['Team2_AwayWinRate']
    games['FGPctDiff'] = games['Team1_FGPct'] - games['Team2_FGPct']
    games['3PPctDiff'] = games['Team1_3PPct'] - games['Team2_3PPct']
    games['FTPctDiff'] = games['Team1_FTPct'] - games['Team2_FTPct']
    
    print(f"Team statistics features calculation complete, processed {len(team_stats)} team-season combinations")
    
    return games, team_stats

def add_head_to_head_features(games):
    """
    Add head-to-head features to the games DataFrame
    
    Args:
        games: DataFrame with game data
        
    Returns:
        DataFrame with head-to-head features added
    """
    print("Calculating head-to-head features...")
    
    def compute_h2h(row, df):
        current_season = row['Season']
        team1 = row['Team1']
        team2 = row['Team2']
        
        # Get all past games between these teams
        past_games = df[(df['Season'] < current_season) &
                        (((df['Team1'] == team1) & (df['Team2'] == team2)) |
                         ((df['Team1'] == team2) & (df['Team2'] == team1)))]
        
        if past_games.empty:
            return {
                'win_rate': 0.5,
                'games_count': 0,
                'avg_score_diff': 0.0,
                'last_3_win_rate': 0.5,
                'last_5_win_rate': 0.5
            }
        
        # Count total games
        total_games = len(past_games)
        
        # Calculate base win rate
        team1_win_count = len(past_games[(past_games['Team1'] == team1) & (past_games['WinA'] == 1)]) + \
                          len(past_games[(past_games['Team2'] == team1) & (past_games['WinA'] == 0)])
        win_rate = team1_win_count / total_games if total_games > 0 else 0.5
        
        # Calculate average score difference
        if 'ScoreDiffNorm' in past_games.columns:
            # Need to adjust based on which team is Team1 in each game
            score_diffs = []
            for _, game in past_games.iterrows():
                if game['Team1'] == team1:
                    score_diffs.append(game['ScoreDiffNorm'])
                else:
                    score_diffs.append(-game['ScoreDiffNorm'])
            avg_score_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0.0
        else:
            avg_score_diff = 0.0
        
        # Calculate win rates for last 3 and last 5 games
        past_games_sorted = past_games.sort_values('Season', ascending=False)
        
        last_3_games = past_games_sorted.head(3)
        last_3_team1_wins = len(last_3_games[(last_3_games['Team1'] == team1) & (last_3_games['WinA'] == 1)]) + \
                           len(last_3_games[(last_3_games['Team2'] == team1) & (last_3_games['WinA'] == 0)])
        last_3_win_rate = last_3_team1_wins / len(last_3_games) if len(last_3_games) > 0 else 0.5
        
        last_5_games = past_games_sorted.head(5)
        last_5_team1_wins = len(last_5_games[(last_5_games['Team1'] == team1) & (last_5_games['WinA'] == 1)]) + \
                           len(last_5_games[(last_5_games['Team2'] == team1) & (last_5_games['WinA'] == 0)])
        last_5_win_rate = last_5_team1_wins / len(last_5_games) if len(last_5_games) > 0 else 0.5
        
        return {
            'win_rate': win_rate,
            'games_count': total_games,
            'avg_score_diff': avg_score_diff,
            'last_3_win_rate': last_3_win_rate,
            'last_5_win_rate': last_5_win_rate
        }
    
    # Apply the function to each row
    h2h_stats = games.apply(lambda r: compute_h2h(r, games), axis=1)
    
    # Extract the values and set them as new columns
    games['H2H_WinRate'] = h2h_stats.apply(lambda x: x['win_rate'])
    games['H2H_GamesCount'] = h2h_stats.apply(lambda x: x['games_count'])
    games['H2H_AvgScoreDiff'] = h2h_stats.apply(lambda x: x['avg_score_diff'])
    games['H2H_Last3WinRate'] = h2h_stats.apply(lambda x: x['last_3_win_rate'])
    games['H2H_Last5WinRate'] = h2h_stats.apply(lambda x: x['last_5_win_rate'])
    
    print("Head-to-head features calculation complete")
    return games

def add_recent_performance_features(games, window=5):
    """
    Add features based on teams' recent performance (last N games)
    
    Args:
        games: DataFrame with game data
        window: Number of recent games to consider
        
    Returns:
        DataFrame with added recent performance features
    """
    print(f"Adding recent performance features (window = {window})...")
    
    # Sort games by DayNum within each Season
    games = games.sort_values(['Season', 'DayNum'])
    
    # Make a copy to avoid modifying the original DataFrame
    result = games.copy()
    
    # Create a dictionary to store recent game results by team
    recent_games = {}
    
    def recent_stats(row, team, df):
        """Get recent performance stats for a team"""
        season = row['Season']
        day = row['DayNum']
        
        # Get previous games in the same season
        prev_games = df[(df['Season'] == season) & (df['DayNum'] < day) & 
                         ((df['Team1'] == team) | (df['Team2'] == team))].sort_values('DayNum', ascending=False).head(window)
        
        if len(prev_games) == 0:
            return {
                'win_rate': 0.5,
                'score_diff': 0,
                'momentum': 0,
                'reb_avg': 0,
                'reb_std': 0,
                'three_pt_pct_avg': 0.33,
                'three_pt_pct_std': 0.05
            }
        
        # Calculate basic stats
        wins = 0
        score_diffs = []
        reb_counts = []
        three_pt_pcts = []
        
        for _, g in prev_games.iterrows():
            # Determine if this team won and by how much
            if g['Team1'] == team:
                is_win = g['WinA'] == 1
                if 'ScoreDiffNorm' in g:
                    score_diff = g['ScoreDiffNorm']
                else:
                    # If normalized score diff isn't available, use a default
                    score_diff = 5 if is_win else -5
                
                # Get rebounding stats if available
                if 'WOR' in g and 'WDR' in g:
                    reb_count = g['WOR'] + g['WDR'] if is_win else g['LOR'] + g['LDR']
                    reb_counts.append(reb_count)
                
                # Get three-point shooting stats if available
                if 'WFGM3' in g and 'WFGA3' in g:
                    if is_win:
                        three_pt_pct = g['WFGM3'] / g['WFGA3'] if g['WFGA3'] > 0 else 0.33
                    else:
                        three_pt_pct = g['LFGM3'] / g['LFGA3'] if g['LFGA3'] > 0 else 0.33
                    three_pt_pcts.append(three_pt_pct)
            else:  # Team2
                is_win = g['WinA'] == 0
                if 'ScoreDiffNorm' in g:
                    score_diff = -g['ScoreDiffNorm']  # Reverse for Team2
                else:
                    score_diff = 5 if is_win else -5
                
                # Get rebounding stats if available
                if 'WOR' in g and 'WDR' in g:
                    reb_count = g['WOR'] + g['WDR'] if not is_win else g['LOR'] + g['LDR']
                    reb_counts.append(reb_count)
                
                # Get three-point shooting stats if available
                if 'WFGM3' in g and 'WFGA3' in g:
                    if not is_win:
                        three_pt_pct = g['WFGM3'] / g['WFGA3'] if g['WFGA3'] > 0 else 0.33
                    else:
                        three_pt_pct = g['LFGM3'] / g['LFGA3'] if g['LFGA3'] > 0 else 0.33
                    three_pt_pcts.append(three_pt_pct)
            
            if is_win:
                wins += 1
            
            score_diffs.append(score_diff)
        
        win_rate = wins / len(prev_games)
        avg_score_diff = sum(score_diffs) / len(prev_games)
        
        # Calculate momentum (weighted recent performance)
        weights = [1.0, 0.8, 0.6, 0.4, 0.2][:len(score_diffs)]
        weights = [w / sum(weights) for w in weights]
        momentum = sum(s * w for s, w in zip(score_diffs, weights))
        
        # Calculate rebounding and shooting statistics
        reb_avg = sum(reb_counts) / len(reb_counts) if reb_counts else 0
        reb_std = np.std(reb_counts) if len(reb_counts) > 1 else 0
        
        three_pt_pct_avg = sum(three_pt_pcts) / len(three_pt_pcts) if three_pt_pcts else 0.33
        three_pt_pct_std = np.std(three_pt_pcts) if len(three_pt_pcts) > 1 else 0.05
        
        return {
            'win_rate': win_rate,
            'score_diff': avg_score_diff,
            'momentum': momentum,
            'reb_avg': reb_avg,
            'reb_std': reb_std,
            'three_pt_pct_avg': three_pt_pct_avg,
            'three_pt_pct_std': three_pt_pct_std
        }
    
    # Process each game
    for idx, row in result.iterrows():
        team1 = row['Team1']
        team2 = row['Team2']
        season = row['Season']
        day = row['DayNum']
        
        # Get recent stats for Team1
        key1 = f"{season}_{team1}_{day}"
        if key1 not in recent_games:
            recent_games[key1] = recent_stats(row, team1, games)
        
        # Get recent stats for Team2
        key2 = f"{season}_{team2}_{day}"
        if key2 not in recent_games:
            recent_games[key2] = recent_stats(row, team2, games)
        
        # Add to result DataFrame
        team1_stats = recent_games[key1]
        team2_stats = recent_games[key2]
        
        result.loc[idx, 'Team1_RecentWinRate'] = team1_stats['win_rate']
        result.loc[idx, 'Team2_RecentWinRate'] = team2_stats['win_rate']
        result.loc[idx, 'Team1_RecentScoreDiff'] = team1_stats['score_diff']
        result.loc[idx, 'Team2_RecentScoreDiff'] = team2_stats['score_diff']
        result.loc[idx, 'Team1_Momentum'] = team1_stats['momentum']
        result.loc[idx, 'Team2_Momentum'] = team2_stats['momentum']
        
        # Add new rebounding and three-point fluctuation metrics
        result.loc[idx, 'Team1_RecentRebAvg'] = team1_stats['reb_avg']
        result.loc[idx, 'Team2_RecentRebAvg'] = team2_stats['reb_avg']
        result.loc[idx, 'Team1_RecentRebStd'] = team1_stats['reb_std']
        result.loc[idx, 'Team2_RecentRebStd'] = team2_stats['reb_std']
        result.loc[idx, 'Team1_Recent3PtPctAvg'] = team1_stats['three_pt_pct_avg']
        result.loc[idx, 'Team2_Recent3PtPctAvg'] = team2_stats['three_pt_pct_avg']
        result.loc[idx, 'Team1_Recent3PtPctStd'] = team1_stats['three_pt_pct_std']
        result.loc[idx, 'Team2_Recent3PtPctStd'] = team2_stats['three_pt_pct_std']
    
    # Calculate differential metrics
    result['RecentWinRateDiff'] = result['Team1_RecentWinRate'] - result['Team2_RecentWinRate']
    result['RecentScoreDiffDiff'] = result['Team1_RecentScoreDiff'] - result['Team2_RecentScoreDiff']
    result['MomentumDiff'] = result['Team1_Momentum'] - result['Team2_Momentum']
    
    # New differential features for rebounding and shooting
    result['RecentRebAvgDiff'] = result['Team1_RecentRebAvg'] - result['Team2_RecentRebAvg']
    result['RecentRebStdDiff'] = result['Team1_RecentRebStd'] - result['Team2_RecentRebStd']
    result['Recent3PtPctAvgDiff'] = result['Team1_Recent3PtPctAvg'] - result['Team2_Recent3PtPctAvg']
    result['Recent3PtPctStdDiff'] = result['Team1_Recent3PtPctStd'] - result['Team2_Recent3PtPctStd']
    
    # Create composite measures of consistency
    result['Team1_Consistency'] = -result['Team1_RecentRebStd'] - result['Team1_Recent3PtPctStd'] * 100
    result['Team2_Consistency'] = -result['Team2_RecentRebStd'] - result['Team2_Recent3PtPctStd'] * 100
    result['ConsistencyDiff'] = result['Team1_Consistency'] - result['Team2_Consistency']
    
    print("  Recent performance features added")
    return result

def add_feature_crosses(games):
    """
    Add feature crossing to enhance the predictive power of the model.
    Creates interaction terms between important features.
    
    Args:
        games: DataFrame with game data containing features to cross
        
    Returns:
        DataFrame with added feature crossings
    """
    print("Adding feature crossing interactions...")
    
    # Make a copy to avoid modifying the original DataFrame
    result = games.copy()
    
    # List of features to cross (important predictors)
    seed_features = ['SeedDiff', 'SeedStrengthDiff']
    elo_features = ['EloDiff'] if 'EloDiff' in games.columns else []
    kenpom_features = ['AdjEM_Diff', 'AdjO_Diff', 'AdjD_Diff'] if 'AdjEM_Diff' in games.columns else []
    shooting_features = ['EFGPctDiff', 'TSPctDiff'] if 'EFGPctDiff' in games.columns and 'TSPctDiff' in games.columns else []
    possession_features = ['OffRebRateDiff'] if 'OffRebRateDiff' in games.columns else []
    momentum_features = ['RecentWinRateDiff', 'MomentumDiff'] if 'RecentWinRateDiff' in games.columns and 'MomentumDiff' in games.columns else []
    
    # Check if gender feature is available (now using GenderCode instead of Gender)
    has_gender = 'GenderCode' in result.columns
    
    # Check if features are available and create crosses
    
    # 1. Seed × Elo interactions
    for seed_feat in seed_features:
        for elo_feat in elo_features:
            if seed_feat in result.columns and elo_feat in result.columns:
                result[f'{seed_feat}_x_{elo_feat}'] = result[seed_feat] * result[elo_feat]
                print(f"  Created {seed_feat} × {elo_feat} interaction")
                
                # Add gender-specific interaction if gender is available
                if has_gender:
                    result[f'{seed_feat}_x_{elo_feat}_x_Gender'] = result[f'{seed_feat}_x_{elo_feat}'] * result['GenderCode']
                    print(f"  Created {seed_feat} × {elo_feat} × Gender interaction")
    
    # 2. Seed × KenPom interactions
    for seed_feat in seed_features:
        for kp_feat in kenpom_features:
            if seed_feat in result.columns and kp_feat in result.columns:
                result[f'{seed_feat}_x_{kp_feat}'] = result[seed_feat] * result[kp_feat]
                print(f"  Created {seed_feat} × {kp_feat} interaction")
                
                # Add gender-specific interaction if gender is available
                if has_gender:
                    result[f'{seed_feat}_x_{kp_feat}_x_Gender'] = result[f'{seed_feat}_x_{kp_feat}'] * result['GenderCode']
                    print(f"  Created {seed_feat} × {kp_feat} × Gender interaction")
    
    # 3. Elo × KenPom interactions
    for elo_feat in elo_features:
        for kp_feat in kenpom_features:
            if elo_feat in result.columns and kp_feat in result.columns:
                result[f'{elo_feat}_x_{kp_feat}'] = result[elo_feat] * result[kp_feat]
                print(f"  Created {elo_feat} × {kp_feat} interaction")
                
                # Add gender-specific interaction if gender is available
                if has_gender:
                    result[f'{elo_feat}_x_{kp_feat}_x_Gender'] = result[f'{elo_feat}_x_{kp_feat}'] * result['GenderCode']
                    print(f"  Created {elo_feat} × {kp_feat} × Gender interaction")
    
    # 4. Shooting × Possession interactions (Four Factors interactions)
    for shoot_feat in shooting_features:
        for poss_feat in possession_features:
            if shoot_feat in result.columns and poss_feat in result.columns:
                result[f'{shoot_feat}_x_{poss_feat}'] = result[shoot_feat] * result[poss_feat]
                print(f"  Created {shoot_feat} × {poss_feat} interaction")
                
                # Add gender-specific interaction if gender is available
                if has_gender:
                    result[f'{shoot_feat}_x_{poss_feat}_x_Gender'] = result[f'{shoot_feat}_x_{poss_feat}'] * result['GenderCode']
                    print(f"  Created {shoot_feat} × {poss_feat} × Gender interaction")
    
    # 5. Recent performance × seed/elo interactions
    for momnt_feat in momentum_features:
        for seed_feat in seed_features:
            if momnt_feat in result.columns and seed_feat in result.columns:
                result[f'{momnt_feat}_x_{seed_feat}'] = result[momnt_feat] * result[seed_feat]
                print(f"  Created {momnt_feat} × {seed_feat} interaction")
                
                # Add gender-specific interaction if gender is available
                if has_gender:
                    result[f'{momnt_feat}_x_{seed_feat}_x_Gender'] = result[f'{momnt_feat}_x_{seed_feat}'] * result['GenderCode']
                    print(f"  Created {momnt_feat} × {seed_feat} × Gender interaction")
        
        for elo_feat in elo_features:
            if momnt_feat in result.columns and elo_feat in result.columns:
                result[f'{momnt_feat}_x_{elo_feat}'] = result[momnt_feat] * result[elo_feat]
                print(f"  Created {momnt_feat} × {elo_feat} interaction")
                
                # Add gender-specific interaction if gender is available
                if has_gender:
                    result[f'{momnt_feat}_x_{elo_feat}_x_Gender'] = result[f'{momnt_feat}_x_{elo_feat}'] * result['GenderCode']
                    print(f"  Created {momnt_feat} × {elo_feat} × Gender interaction")
    
    # 6. Create higher-order terms for key differentials
    key_diffs = ['SeedDiff', 'EloDiff', 'AdjEM_Diff', 'ChampionComposite']
    for feat in key_diffs:
        if feat in result.columns:
            # Square term (quadratic effect)
            result[f'{feat}_squared'] = result[feat] ** 2
            print(f"  Created {feat}² squared term")
            
            # Add gender-specific squared terms if gender is available
            if has_gender:
                result[f'{feat}_squared_x_Gender'] = result[f'{feat}_squared'] * result['GenderCode']
                print(f"  Created {feat}² × Gender interaction")
    
    # Create a feature category that amplifies underdog advantage when present
    if 'SeedDiff' in result.columns and 'EloDiff' in result.columns:
        # A positive SeedDiff means Team1 has a higher seed number (worse seed)
        # A negative EloDiff means Team1 has lower Elo (worse team)
        # When SeedDiff > 0 and EloDiff < 0, this indicates a potential upset
        result['UnderdogAdvantage'] = np.where(
            (result['SeedDiff'] > 0) & (result['EloDiff'] < 0),
            result['SeedDiff'] * -result['EloDiff'] / 100,  # Scale appropriately
            0
        )
        print("  Created UnderdogAdvantage feature")
        
        # Add gender-specific underdog advantage if gender is available
        if has_gender:
            result['UnderdogAdvantage_x_Gender'] = result['UnderdogAdvantage'] * result['GenderCode']
            print("  Created UnderdogAdvantage × Gender interaction")
    
    # Championship potential composite
    if all(feat in result.columns for feat in ['EFGPctDiff', 'TSPctDiff', 'OffRebRateDiff', 'DefRebRateDiff']):
        # Formula based on championship team profiles
        result['ChampionshipPotential'] = (
            0.35 * result['EFGPctDiff'] + 
            0.25 * result['TSPctDiff'] +
            0.20 * result['OffRebRateDiff'] + 
            0.20 * result['DefRebRateDiff']
        )
        print("  Created ChampionshipPotential combined feature")
        
        # Add gender-specific championship potential if gender is available
        if has_gender:
            result['ChampionshipPotential_x_Gender'] = result['ChampionshipPotential'] * result['GenderCode']
            print("  Created ChampionshipPotential × Gender interaction")
    
    print(f"Added {sum(1 for col in result.columns if '_x_' in col or '_squared' in col or col in ['UnderdogAdvantage', 'ChampionshipPotential'])} feature crosses")
    return result

def aggregate_features(games):
    """
    Aggregate game statistics for later use, using only regular season data to avoid data leakage.
    
    Args:
        games: DataFrame with game data
        
    Returns:
        DataFrame with aggregated features
    """
    print("聚合团队对阵统计特征（仅使用常规赛数据，避免数据泄露）...")
    
    # Check for GameType column
    if 'GameType' not in games.columns:
        print("警告: GameType列未找到，无法区分常规赛和锦标赛。假设所有数据都是常规赛。")
        regular_games = games.copy()
    else:
        # Use only regular season games for aggregation
        regular_games = games[games['GameType'] == 'Regular'].copy()
        if regular_games.empty:
            print("警告: 没有找到常规赛数据，使用所有数据进行聚合。")
            regular_games = games.copy()
        else:
            print(f"使用 {len(regular_games)} 场常规赛比赛进行特征聚合（总共 {len(games)} 场比赛）")
    
    available_cols = [col for col in ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR',
                                     'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA',
                                     'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO',
                                     'LStl', 'LBlk', 'LPF'] if col in regular_games.columns]
    
    if not available_cols:
        print("警告: 没有找到可用的统计列进行聚合。")
        return pd.DataFrame(columns=['IDTeams_c_score'])
    
    # Group by Season and team pair. Construct a key 'SeasonIDTeams'
    regular_games['SeasonIDTeams'] = regular_games.apply(lambda r: f"{r['Season']}_{r['IDTeams']}", axis=1)
    
    agg_funcs = ['mean', 'median', 'std']
    
    # Sort index first to avoid PerformanceWarning
    regular_games_sorted = regular_games.sort_values('SeasonIDTeams')
    gb = regular_games_sorted.groupby('SeasonIDTeams').agg({col: agg_funcs for col in available_cols}).reset_index()
    
    # Split SeasonIDTeams back into Season and IDTeams
    gb[['Season', 'IDTeams']] = gb['SeasonIDTeams'].str.split('_', n=1, expand=True)
    gb = gb.sort_index()
    gb = gb.drop(columns=['SeasonIDTeams'])

    # Rename columns while keeping 'Season' and 'IDTeams' intact
    new_columns = []
    for col in gb.columns:
        if isinstance(col, tuple):
            col_name = ''.join(col)
            if col_name in ['Season', 'IDTeams']:
                new_columns.append(col_name)
            else:
                new_columns.append(col_name + '_c_score')
        else:
            new_columns.append(col)
    gb.columns = new_columns
    
    print(f"聚合特征计算完成，共生成 {len(gb)} 个赛季-团队对组合的特征")
    
    return gb

def prepare_submission_features(submission_df, seed_dict, team_stats, extract_game_info_func):
    """
    Process the sample submission file to prepare features for prediction.
    Uses similar feature mapping as training data.
    
    Args:
        submission_df: DataFrame with sample submission data
        seed_dict: Dictionary mapping Season_TeamID to seed value
        team_stats: Dictionary with team statistics
        extract_game_info_func: Function to extract game information
        
    Returns:
        DataFrame with features for submission predictions
    """
    print("为提交文件准备特征数据...")
    
    # Create a deep copy and add an original index column to track row identity
    submission_df = submission_df.copy()
    submission_df['original_index'] = np.arange(len(submission_df))
    original_ids = submission_df['ID'].copy()  # Save original IDs
    
    game_info = submission_df['ID'].apply(extract_game_info_func)
    submission_df['Season'] = [info[0] for info in game_info]
    submission_df['Team1'] = [info[1] for info in game_info]
    submission_df['Team2'] = [info[2] for info in game_info]
    
    submission_df['IDTeams'] = submission_df.apply(lambda r: f"{r['Team1']}_{r['Team2']}", axis=1)
    submission_df['IDTeam1'] = submission_df.apply(lambda r: f"{r['Season']}_{r['Team1']}", axis=1)
    submission_df['IDTeam2'] = submission_df.apply(lambda r: f"{r['Season']}_{r['Team2']}", axis=1)
    
    # Add gender features based on TeamID
    submission_df['Team1Gender'] = submission_df['Team1'].apply(get_gender_from_teamid)
    submission_df['Team2Gender'] = submission_df['Team2'].apply(get_gender_from_teamid)
    
    # Ensure the gender is the same for both teams (validation check)
    gender_mismatch = submission_df[submission_df['Team1Gender'] != submission_df['Team2Gender']]
    if not gender_mismatch.empty:
        print(f"Warning: {len(gender_mismatch)} submission games have mismatched genders between teams.")
        # For debugging purposes, show some examples
        if len(gender_mismatch) > 0:
            print("Sample of gender mismatches in submission data:")
            print(gender_mismatch[['Season', 'Team1', 'Team2', 'Team1Gender', 'Team2Gender']].head(3))
    
    # Add a single gender column for the game (since both teams should have the same gender)
    submission_df['Gender'] = submission_df['Team1Gender']
    
    # Convert string gender to numeric values (1 for men, 0 for women)
    submission_df['GenderCode'] = submission_df['Gender'].map({'M': 1, 'W': 0}).fillna(0.5)  # Default 0.5 if unknown
    
    # Drop the string gender columns as they can't be used by the model directly
    submission_df = submission_df.drop(columns=['Team1Gender', 'Team2Gender', 'Gender'])
    
    # Seed features
    submission_df['Team1Seed'] = submission_df['IDTeam1'].map(seed_dict).fillna(16)
    submission_df['Team2Seed'] = submission_df['IDTeam2'].map(seed_dict).fillna(16)
    submission_df['SeedDiff'] = submission_df['Team1Seed'] - submission_df['Team2Seed']
    
    submission_df['Team1SeedStrength'] = np.exp(-submission_df['Team1Seed'] / 4)
    submission_df['Team2SeedStrength'] = np.exp(-submission_df['Team2Seed'] / 4)
    submission_df['SeedStrengthDiff'] = submission_df['Team1SeedStrength'] - submission_df['Team2SeedStrength']
    
    # Basic team statistics
    submission_df['Team1_WinRate'] = submission_df['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('win_rate', 0.5))
    submission_df['Team2_WinRate'] = submission_df['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('win_rate', 0.5))
    submission_df['Team1_GamesPlayed'] = submission_df['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('games_played', 0))
    submission_df['Team2_GamesPlayed'] = submission_df['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('games_played', 0))
    submission_df['Team1_AvgScore'] = submission_df['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('avg_score', 60))
    submission_df['Team2_AvgScore'] = submission_df['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('avg_score', 60))
    submission_df['Team1_AvgAllowed'] = submission_df['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('avg_allowed', 60))
    submission_df['Team2_AvgAllowed'] = submission_df['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('avg_allowed', 60))
    submission_df['Team1_PointDiff'] = submission_df['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('point_diff', 0))
    submission_df['Team2_PointDiff'] = submission_df['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('point_diff', 0))
    
    # New team statistics
    submission_df['Team1_HomeWinRate'] = submission_df['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('home_win_rate', 0.5))
    submission_df['Team2_HomeWinRate'] = submission_df['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('home_win_rate', 0.5))
    submission_df['Team1_AwayWinRate'] = submission_df['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('away_win_rate', 0.5))
    submission_df['Team2_AwayWinRate'] = submission_df['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('away_win_rate', 0.5))
    submission_df['Team1_FGPct'] = submission_df['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('fg_pct', 0.45))
    submission_df['Team2_FGPct'] = submission_df['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('fg_pct', 0.45))
    submission_df['Team1_3PPct'] = submission_df['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('fg3_pct', 0.33))
    submission_df['Team2_3PPct'] = submission_df['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('fg3_pct', 0.33))
    submission_df['Team1_FTPct'] = submission_df['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('ft_pct', 0.7))
    submission_df['Team2_FTPct'] = submission_df['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('ft_pct', 0.7))
    
    # Basic difference features
    submission_df['WinRateDiff'] = submission_df['Team1_WinRate'] - submission_df['Team2_WinRate']
    submission_df['GamesPlayedDiff'] = submission_df['Team1_GamesPlayed'] - submission_df['Team2_GamesPlayed']
    submission_df['AvgScoreDiff'] = submission_df['Team1_AvgScore'] - submission_df['Team2_AvgScore']
    submission_df['AvgAllowedDiff'] = submission_df['Team1_AvgAllowed'] - submission_df['Team2_AvgAllowed']
    submission_df['PointDiffDiff'] = submission_df['Team1_PointDiff'] - submission_df['Team2_PointDiff']
    
    # New difference features
    submission_df['HomeWinRateDiff'] = submission_df['Team1_HomeWinRate'] - submission_df['Team2_HomeWinRate']
    submission_df['AwayWinRateDiff'] = submission_df['Team1_AwayWinRate'] - submission_df['Team2_AwayWinRate']
    submission_df['FGPctDiff'] = submission_df['Team1_FGPct'] - submission_df['Team2_FGPct']
    submission_df['3PPctDiff'] = submission_df['Team1_3PPct'] - submission_df['Team2_3PPct']
    submission_df['FTPctDiff'] = submission_df['Team1_FTPct'] - submission_df['Team2_FTPct']
    
    # Force the GameType for submission to "Tournament"
    submission_df['GameType'] = 'Tournament'
    
    # Add placeholder values for head-to-head features
    submission_df['H2H_WinRate'] = 0.5
    submission_df['H2H_GamesCount'] = 0
    submission_df['H2H_AvgScoreDiff'] = 0.0
    submission_df['H2H_Last3WinRate'] = 0.5
    submission_df['H2H_Last5WinRate'] = 0.5
    
    # Add placeholder values for recent performance features
    submission_df['Team1_RecentWinRate'] = 0.5
    submission_df['Team2_RecentWinRate'] = 0.5
    submission_df['Team1_RecentScoreDiff'] = 0.0
    submission_df['Team2_RecentScoreDiff'] = 0.0
    submission_df['Team1_Streak'] = 0
    submission_df['Team2_Streak'] = 0
    submission_df['Team1_RecentAvgScore'] = 60.0
    submission_df['Team2_RecentAvgScore'] = 60.0
    submission_df['Team1_RecentAvgAllowed'] = 60.0
    submission_df['Team2_RecentAvgAllowed'] = 60.0
    submission_df['Team1_RecentGamesCount'] = 0
    submission_df['Team2_RecentGamesCount'] = 0
    
    # Recent difference features
    submission_df['RecentWinRateDiff'] = 0.0
    submission_df['RecentScoreDiffDiff'] = 0.0
    submission_df['StreakDiff'] = 0
    submission_df['RecentAvgScoreDiff'] = 0.0
    submission_df['RecentAvgAllowedDiff'] = 0.0
    submission_df['Team1_Momentum'] = 0.0
    submission_df['Team2_Momentum'] = 0.0
    submission_df['MomentumDiff'] = 0.0
    
    # Store original IDs for final alignment
    submission_df['original_ID'] = original_ids
    
    print("提交文件特征准备完成")
    return submission_df

def merge_kenpom_features(games, kenpom_df):
    """
    Merge KenPom features into games DataFrame.
    
    Args:
        games: DataFrame with game information.
        kenpom_df: DataFrame with KenPom metrics.
        
    Returns:
        games: DataFrame with new KenPom feature columns added.
    """
    if kenpom_df.empty:
        print("Warning: KenPom DataFrame is empty. No features will be added.")
        return games
    
    print(f"Available columns in KenPom data: {kenpom_df.columns.tolist()}")
    
    if 'Season' not in kenpom_df.columns:
        if 'Year' in kenpom_df.columns:
            kenpom_df.rename(columns={'Year': 'Season'}, inplace=True)
        else:
            print("Warning: No 'Season' column found in KenPom data. Using 2025 as default.")
            kenpom_df['Season'] = 2025
    
    kenpom_df['Season'] = kenpom_df['Season'].astype(int)
    
    efficiency_column = None
    if 'AdjEM' in kenpom_df.columns:
        efficiency_column = 'AdjEM'
    elif 'AdjEff' in kenpom_df.columns:
        efficiency_column = 'AdjEff'
    elif 'NetRtg' in kenpom_df.columns:
        efficiency_column = 'NetRtg'
    else:
        possible_columns = [col for col in kenpom_df.columns if 'Eff' in col or 'Rtg' in col or 'EM' in col]
        if possible_columns:
            efficiency_column = possible_columns[0]
            print(f"Using '{efficiency_column}' as efficiency metric.")
        else:
            print("Warning: No efficiency metric found in KenPom data. Returning games without KenPom features.")
            return games
    
    try:
        kenpom_df[efficiency_column] = pd.to_numeric(kenpom_df[efficiency_column], errors='coerce')
        if kenpom_df[efficiency_column].isna().any():
            print(f"Warning: Found {kenpom_df[efficiency_column].isna().sum()} NaN values in {efficiency_column}. Filling with median.")
            kenpom_df[efficiency_column] = kenpom_df[efficiency_column].fillna(kenpom_df[efficiency_column].median())
    except Exception as e:
        print(f"Error converting {efficiency_column} to numeric: {e}")
        print("Returning games without KenPom features.")
        return games
    
    # Merge for Team1
    kenpom_team1 = kenpom_df[['Season', 'TeamID', efficiency_column]].copy()
    kenpom_team1.rename(columns={'TeamID': 'Team1', efficiency_column: 'Team1_AdjEM'}, inplace=True)
    games = games.merge(kenpom_team1, how='left', on=['Season', 'Team1'])
    
    # Merge for Team2
    kenpom_team2 = kenpom_df[['Season', 'TeamID', efficiency_column]].copy()
    kenpom_team2.rename(columns={'TeamID': 'Team2', efficiency_column: 'Team2_AdjEM'}, inplace=True)
    games = games.merge(kenpom_team2, how='left', on=['Season', 'Team2'])
    
    games['Team1_AdjEM'] = pd.to_numeric(games['Team1_AdjEM'], errors='coerce').fillna(0)
    games['Team2_AdjEM'] = pd.to_numeric(games['Team2_AdjEM'], errors='coerce').fillna(0)
    
    games['AdjEM_Diff'] = games['Team1_AdjEM'] - games['Team2_AdjEM']
    print("KenPom features merged.")
    
    return games

def add_season_gametype_label(games):
    """
    Create a new feature that combines Season and GameType into descriptive labels.
    For the current season (assumed to be Regular), label as 'current_year_regular'.
    For previous seasons (Tournament), label as 'last_year_tournament', '2_years_ago_tournament', etc.
    
    Args:
        games: DataFrame with game data, must include 'Season' and 'GameType'.
    
    Returns:
        DataFrame with an added column 'Season_GameType_Label'.
    """
    current_season = games['Season'].max()
    
    def get_season_gametype_label(row):
        season = row['Season']
        # Lowercase the GameType for consistency
        game_type = row['GameType'].lower()
        if season == current_season:
            return "current_year_regular"
        else:
            diff = current_season - season
            if diff == 1:
                return "last_year_tournament"
            else:
                return f"{diff}_years_ago_tournament"
    
    games['Season_GameType_Label'] = games.apply(get_season_gametype_label, axis=1)
    return games

def add_elo_ratings(games, k_factor=20, initial_elo=1500, reset_each_season=True):
    """
    Add Elo rating features to the games DataFrame
    
    Args:
        games: DataFrame with game data
        k_factor: K-factor for Elo calculation (higher = more volatility)
        initial_elo: Starting Elo rating for new teams
        reset_each_season: Whether to reset ratings each season
        
    Returns:
        DataFrame with added Elo rating features
    """
    print("Calculating Elo rating features...")
    
    # Make a copy to avoid modifying the original DataFrame
    result = games.copy()
    
    # Check if we have required columns for proper Elo calculation
    is_submission_data = 'WTeamID' not in result.columns or 'LTeamID' not in result.columns
    has_daynum = 'DayNum' in result.columns
    
    if not has_daynum:
        # This is expected for submission data, so we make it an informational message rather than a warning
        if is_submission_data:
            print("  Info: Using row index for game ordering in submission data (no DayNum column needed).")
        else:
            print("  Info: DayNum column not found. Using row index for chronological ordering.")
        # Create a temporary DayNum column if not present (for submission data)
        # We'll assume the data is already sorted or it's submission data where order doesn't matter
        result['DayNum'] = result.index
        result['_temp_daynum_added'] = True  # Flag to remove this later
    
    # Initialize team Elo ratings dictionaries
    team_elos = {}
    team_elos_before = {}
    
    # Sort games by Season and DayNum to ensure chronological order within each season
    if has_daynum or '_temp_daynum_added' in result.columns:
        result = result.sort_values(['Season', 'DayNum'])
    
    # Process each game
    for idx, row in result.iterrows():
        season = row['Season']
        
        # Get teams involved
        if is_submission_data:
            team_a = row['Team1']
            team_b = row['Team2']
            # For submission data, we don't know the winner yet
            winner = None
            loser = None
        else:
            team_a = row['WTeamID'] if 'WTeamID' in row else row['Team1']
            team_b = row['LTeamID'] if 'LTeamID' in row else row['Team2']
            winner = team_a
            loser = team_b
        
        # Create season-specific team keys
        team_a_key = f"{season}_{team_a}"
        team_b_key = f"{season}_{team_b}"
        
        # Initialize team Elo ratings if not already present
        if reset_each_season:
            # If we reset ratings each season, check season-specific keys
            if team_a_key not in team_elos:
                # Look for previous season's final Elo for this team
                prev_season = season - 1
                prev_team_key = f"{prev_season}_{team_a}"
                if prev_team_key in team_elos:
                    # Use previous season's rating, but regress towards the mean
                    team_elos[team_a_key] = initial_elo + 0.75 * (team_elos[prev_team_key] - initial_elo)
                else:
                    team_elos[team_a_key] = initial_elo
            
            if team_b_key not in team_elos:
                # Look for previous season's final Elo for this team
                prev_season = season - 1
                prev_team_key = f"{prev_season}_{team_b}"
                if prev_team_key in team_elos:
                    # Use previous season's rating, but regress towards the mean
                    team_elos[team_b_key] = initial_elo + 0.75 * (team_elos[prev_team_key] - initial_elo)
                else:
                    team_elos[team_b_key] = initial_elo
        else:
            # If we don't reset ratings, just check if the team exists in our dict
            if team_a not in team_elos:
                team_elos[team_a] = initial_elo
            if team_b not in team_elos:
                team_elos[team_b] = initial_elo
        
        # Retrieve current Elo ratings
        team_a_elo = team_elos[team_a_key] if reset_each_season else team_elos[team_a]
        team_b_elo = team_elos[team_b_key] if reset_each_season else team_elos[team_b]
        
        # Store Elo ratings before update
        team_elos_before[team_a_key] = team_a_elo
        team_elos_before[team_b_key] = team_b_elo
        
        # Calculate Elo if we know the actual result
        if not is_submission_data:
            # Elo update: Calculate expected win probability
            elo_diff = team_a_elo - team_b_elo
            team_a_win_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
            
            # Update Elo ratings based on actual outcome
            if winner == team_a:  # team_a won
                team_a_new_elo = team_a_elo + k_factor * (1.0 - team_a_win_prob)
                team_b_new_elo = team_b_elo + k_factor * (0.0 - (1.0 - team_a_win_prob))
            else:  # team_b won
                team_a_new_elo = team_a_elo + k_factor * (0.0 - team_a_win_prob)
                team_b_new_elo = team_b_elo + k_factor * (1.0 - (1.0 - team_a_win_prob))
            
            # Update team Elo ratings
            if reset_each_season:
                team_elos[team_a_key] = team_a_new_elo
                team_elos[team_b_key] = team_b_new_elo
            else:
                team_elos[team_a] = team_a_new_elo
                team_elos[team_b] = team_b_new_elo
    
    # Add Elo rating features to the result DataFrame
    for idx, row in result.iterrows():
        season = row['Season']
        
        if is_submission_data:
            team1 = row['Team1']
            team2 = row['Team2']
        else:
            team1 = row['WTeamID'] if 'WTeamID' in row else row['Team1']
            team2 = row['LTeamID'] if 'LTeamID' in row else row['Team2']
        
        # Get keys based on whether we're resetting per season
        team1_key = f"{season}_{team1}" if reset_each_season else team1
        team2_key = f"{season}_{team2}" if reset_each_season else team2
        
        # Set Elo ratings in the result DataFrame
        if team1_key in team_elos_before:
            result.loc[idx, 'Team1_Elo'] = team_elos_before[team1_key]
        else:
            result.loc[idx, 'Team1_Elo'] = initial_elo
        
        if team2_key in team_elos_before:
            result.loc[idx, 'Team2_Elo'] = team_elos_before[team2_key]
        else:
            result.loc[idx, 'Team2_Elo'] = initial_elo
        
        # Calculate Elo difference
        result.loc[idx, 'EloDiff'] = result.loc[idx, 'Team1_Elo'] - result.loc[idx, 'Team2_Elo']
    
    # If we added a temporary DayNum column, remove it
    if '_temp_daynum_added' in result.columns:
        result = result.drop(['DayNum', '_temp_daynum_added'], axis=1)
    
    print(f"Elo rating features calculation complete, processed {len(team_elos) // (2 if reset_each_season else 1)} teams")
    return result

def add_strength_of_schedule(games, team_stats, current_season=None):
    """
    Add strength of schedule (SOS) features to the games DataFrame
    
    Args:
        games: DataFrame with game data
        team_stats: DataFrame with team statistics
        current_season: Current season for filtering (optional)
        
    Returns:
        DataFrame with added strength of schedule features
    """
    print("Calculating strength of schedule (SOS) features...")
    
    # Make a copy to avoid modifying the original DataFrame
    result = games.copy()
    
    # Determine if we're dealing with submission data
    is_submission_data = 'WTeamID' not in result.columns or 'LTeamID' not in result.columns
    
    # Set current season if not provided
    if current_season is None:
        current_season = result['Season'].max()
    
    # If team_stats is empty, set default SOS values
    if not team_stats:
        print("  No team statistics found. Setting default SOS values.")
        sos_columns = [
            'Team1_SOS_WinRate', 'Team1_SOS_OffRating', 'Team1_SOS_DefRating', 'Team1_SOS_Combined',
            'Team2_SOS_WinRate', 'Team2_SOS_OffRating', 'Team2_SOS_DefRating', 'Team2_SOS_Combined',
            'SOS_WinRateDiff', 'SOS_OffRatingDiff', 'SOS_DefRatingDiff', 'SOS_CombinedDiff'
        ]
        for col in sos_columns:
            result[col] = 0.0
        return result
    
    # Create a dictionary to store Strength of Schedule (SOS) metrics by team
    sos_metrics = {}
    
    # Process games to calculate SOS metrics
    for idx, row in result.iterrows():
        season = row['Season']
        
        if is_submission_data:
            team1 = row['Team1']
            team2 = row['Team2']
            
            # Create team keys
            team1_key = f"{season}_{team1}"
            team2_key = f"{season}_{team2}"
        else:
            # Extract team IDs from game data
            wteam = row['WTeamID'] if 'WTeamID' in row else row['Team1']
            lteam = row['LTeamID'] if 'LTeamID' in row else row['Team2']
            
            # Create team keys
            team1_key = f"{season}_{wteam}"
            team2_key = f"{season}_{lteam}"
        
        # Initialize SOS metrics for teams if they don't exist yet
        for team_key in [team1_key, team2_key]:
            if team_key not in sos_metrics:
                sos_metrics[team_key] = {
                    'opponents': set(),
                    'total_games': 0
                }
            
        # Add opponent to SOS calculation
        if not is_submission_data:
            # Regular game data - we know who played whom
            sos_metrics[team1_key]['opponents'].add(team2_key)
            sos_metrics[team2_key]['opponents'].add(team1_key)
            sos_metrics[team1_key]['total_games'] += 1
            sos_metrics[team2_key]['total_games'] += 1
    
    # Calculate SOS metrics for each team
    for team_key, data in sos_metrics.items():
        # Calculate average opponent metrics
        opponents = data['opponents']
        
        if not opponents:
            # No opponents found (possibly for submission data)
            sos_metrics[team_key]['win_rate'] = 0.5
            sos_metrics[team_key]['off_rating'] = 100.0
            sos_metrics[team_key]['def_rating'] = 100.0
            sos_metrics[team_key]['combined'] = 0.0
            continue
        
        # Calculate opponent average win rate
        opp_win_rates = [team_stats.get(opp, {}).get('win_rate', 0.5) for opp in opponents]
        avg_opp_win_rate = sum(opp_win_rates) / len(opponents)
        
        # Calculate opponent average offensive rating (points scored)
        opp_off_ratings = [team_stats.get(opp, {}).get('avg_score', 60.0) for opp in opponents]
        avg_opp_off = sum(opp_off_ratings) / len(opponents)
        
        # Calculate opponent average defensive rating (points allowed)
        opp_def_ratings = [team_stats.get(opp, {}).get('avg_allowed', 60.0) for opp in opponents]
        avg_opp_def = sum(opp_def_ratings) / len(opponents)
        
        # Combined SOS metric (higher = tougher schedule)
        combined_sos = (avg_opp_win_rate - 0.5) * 100 + (avg_opp_off - avg_opp_def) / 2
        
        # Store SOS metrics
        sos_metrics[team_key]['win_rate'] = avg_opp_win_rate
        sos_metrics[team_key]['off_rating'] = avg_opp_off
        sos_metrics[team_key]['def_rating'] = avg_opp_def
        sos_metrics[team_key]['combined'] = combined_sos
    
    # Add SOS features to the result DataFrame
    for idx, row in result.iterrows():
        season = row['Season']
        
        if is_submission_data:
            team1 = row['Team1']
            team2 = row['Team2']
        else:
            # Extract team IDs from game data
            if 'Team1' in row and 'Team2' in row:
                team1 = row['Team1']
                team2 = row['Team2']
            else:
                team1 = row['WTeamID']
                team2 = row['LTeamID']
        
        # Create team keys
        team1_key = f"{season}_{team1}"
        team2_key = f"{season}_{team2}"
        
        # Add Team1 SOS features
        if team1_key in sos_metrics:
            result.loc[idx, 'Team1_SOS_WinRate'] = sos_metrics[team1_key]['win_rate']
            result.loc[idx, 'Team1_SOS_OffRating'] = sos_metrics[team1_key]['off_rating']
            result.loc[idx, 'Team1_SOS_DefRating'] = sos_metrics[team1_key]['def_rating']
            result.loc[idx, 'Team1_SOS_Combined'] = sos_metrics[team1_key]['combined']
        else:
            # Default values if team not found
            result.loc[idx, 'Team1_SOS_WinRate'] = 0.5
            result.loc[idx, 'Team1_SOS_OffRating'] = 100.0
            result.loc[idx, 'Team1_SOS_DefRating'] = 100.0
            result.loc[idx, 'Team1_SOS_Combined'] = 0.0
        
        # Add Team2 SOS features
        if team2_key in sos_metrics:
            result.loc[idx, 'Team2_SOS_WinRate'] = sos_metrics[team2_key]['win_rate']
            result.loc[idx, 'Team2_SOS_OffRating'] = sos_metrics[team2_key]['off_rating']
            result.loc[idx, 'Team2_SOS_DefRating'] = sos_metrics[team2_key]['def_rating']
            result.loc[idx, 'Team2_SOS_Combined'] = sos_metrics[team2_key]['combined']
        else:
            # Default values if team not found
            result.loc[idx, 'Team2_SOS_WinRate'] = 0.5
            result.loc[idx, 'Team2_SOS_OffRating'] = 100.0
            result.loc[idx, 'Team2_SOS_DefRating'] = 100.0
            result.loc[idx, 'Team2_SOS_Combined'] = 0.0
        
        # Calculate differential features
        result.loc[idx, 'SOS_WinRateDiff'] = result.loc[idx, 'Team1_SOS_WinRate'] - result.loc[idx, 'Team2_SOS_WinRate']
        result.loc[idx, 'SOS_OffRatingDiff'] = result.loc[idx, 'Team1_SOS_OffRating'] - result.loc[idx, 'Team2_SOS_OffRating']
        result.loc[idx, 'SOS_DefRatingDiff'] = result.loc[idx, 'Team1_SOS_DefRating'] - result.loc[idx, 'Team2_SOS_DefRating']
        result.loc[idx, 'SOS_CombinedDiff'] = result.loc[idx, 'Team1_SOS_Combined'] - result.loc[idx, 'Team2_SOS_Combined']
    
    print(f"赛程强度特征计算完成，共处理了 {len(sos_metrics)} 个团队-赛季组合")
    return result

def add_key_stat_differentials(games):
    """
    Add key statistical differentials for shooting, rebounding, and other important metrics
    
    Args:
        games: DataFrame with game data
        
    Returns:
        DataFrame with added key statistical differentials
    """
    print("Calculating key statistical differential features...")
    
    # Make a copy to avoid modifying the original DataFrame
    result = games.copy()
    
    # Determine if we have detailed game statistics
    # Check for the presence of detailed game statistics columns
    required_stats = ['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA',
                      'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF',
                      'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
                      'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
    
    has_stats = all(col in games.columns for col in required_stats)
    
    if not has_stats:
        print("  No detailed game statistics found. This is likely submission data.")
        print("  Setting default statistical values for differentials.")
        
        # Set default values for all statistical differentials
        default_stats = {
            'FGPctDiff': 0.0,        # Field goal percentage differential
            '3PPctDiff': 0.0,         # 3-point percentage differential
            'FTPctDiff': 0.0,         # Free throw percentage differential
            'ReboundDiff': 0.0,       # Rebounding differential
            'AstDiff': 0.0,           # Assist differential
            'TODiff': 0.0,            # Turnover differential
            'StlDiff': 0.0,           # Steal differential
            'BlkDiff': 0.0,           # Block differential
            'PFDiff': 0.0,            # Personal foul differential
            'OffRebDiff': 0.0,        # Offensive rebound differential
            'DefRebDiff': 0.0,        # Defensive rebound differential
            'EffFGPctDiff': 0.0,      # Effective field goal percentage differential
            'TORatioDiff': 0.0,       # Turnover ratio differential
            'OffRebRateDiff': 0.0,    # Offensive rebounding rate differential
            'FTRateDiff': 0.0,        # Free throw rate differential
            'AstTORatioDiff': 0.0     # Assist-to-turnover ratio differential
        }
        
        for stat, default_value in default_stats.items():
            result[stat] = default_value
        
        return result
    
    # Calculate basic shooting percentages if we have stats
    if 'WFGM' in games.columns and 'WFGA' in games.columns:
        # Team1 shooting
        mask = (result['WinA'] == 1)
        result.loc[mask, 'Team1_FGM'] = result.loc[mask, 'WFGM']
        result.loc[mask, 'Team1_FGA'] = result.loc[mask, 'WFGA']
        result.loc[mask, 'Team1_FGM3'] = result.loc[mask, 'WFGM3']
        result.loc[mask, 'Team1_FGA3'] = result.loc[mask, 'WFGA3']
        result.loc[mask, 'Team1_FTM'] = result.loc[mask, 'WFTM']
        result.loc[mask, 'Team1_FTA'] = result.loc[mask, 'WFTA']
        
        result.loc[~mask, 'Team1_FGM'] = result.loc[~mask, 'LFGM']
        result.loc[~mask, 'Team1_FGA'] = result.loc[~mask, 'LFGA']
        result.loc[~mask, 'Team1_FGM3'] = result.loc[~mask, 'LFGM3']
        result.loc[~mask, 'Team1_FGA3'] = result.loc[~mask, 'LFGA3']
        result.loc[~mask, 'Team1_FTM'] = result.loc[~mask, 'LFTM']
        result.loc[~mask, 'Team1_FTA'] = result.loc[~mask, 'LFTA']
        
        # Team2 shooting
        result.loc[mask, 'Team2_FGM'] = result.loc[mask, 'LFGM']
        result.loc[mask, 'Team2_FGA'] = result.loc[mask, 'LFGA']
        result.loc[mask, 'Team2_FGM3'] = result.loc[mask, 'LFGM3']
        result.loc[mask, 'Team2_FGA3'] = result.loc[mask, 'LFGA3']
        result.loc[mask, 'Team2_FTM'] = result.loc[mask, 'LFTM']
        result.loc[mask, 'Team2_FTA'] = result.loc[mask, 'LFTA']
        
        result.loc[~mask, 'Team2_FGM'] = result.loc[~mask, 'WFGM']
        result.loc[~mask, 'Team2_FGA'] = result.loc[~mask, 'WFGA']
        result.loc[~mask, 'Team2_FGM3'] = result.loc[~mask, 'WFGM3']
        result.loc[~mask, 'Team2_FGA3'] = result.loc[~mask, 'WFGA3']
        result.loc[~mask, 'Team2_FTM'] = result.loc[~mask, 'WFTM']
        result.loc[~mask, 'Team2_FTA'] = result.loc[~mask, 'WFTA']
        
        # Calculate field goal percentages
        result['Team1_FGPct'] = result['Team1_FGM'] / result['Team1_FGA'].replace(0, 1)
        result['Team2_FGPct'] = result['Team2_FGM'] / result['Team2_FGA'].replace(0, 1)
        result['FGPctDiff'] = result['Team1_FGPct'] - result['Team2_FGPct']
        
        # Calculate 3-point percentages
        result['Team1_3PPct'] = result['Team1_FGM3'] / result['Team1_FGA3'].replace(0, 1)
        result['Team2_3PPct'] = result['Team2_FGM3'] / result['Team2_FGA3'].replace(0, 1)
        result['3PPctDiff'] = result['Team1_3PPct'] - result['Team2_3PPct']
        
        # Calculate free throw percentages
        result['Team1_FTPct'] = result['Team1_FTM'] / result['Team1_FTA'].replace(0, 1)
        result['Team2_FTPct'] = result['Team2_FTM'] / result['Team2_FTA'].replace(0, 1)
        result['FTPctDiff'] = result['Team1_FTPct'] - result['Team2_FTPct']
    
    # Calculate rebounding if we have stats
    if 'WOR' in games.columns and 'WDR' in games.columns:
        # Team1 rebounding
        mask = (result['WinA'] == 1)
        result.loc[mask, 'Team1_OR'] = result.loc[mask, 'WOR']
        result.loc[mask, 'Team1_DR'] = result.loc[mask, 'WDR']
        result.loc[~mask, 'Team1_OR'] = result.loc[~mask, 'LOR']
        result.loc[~mask, 'Team1_DR'] = result.loc[~mask, 'LDR']
        
        # Team2 rebounding
        result.loc[mask, 'Team2_OR'] = result.loc[mask, 'LOR']
        result.loc[mask, 'Team2_DR'] = result.loc[mask, 'LDR']
        result.loc[~mask, 'Team2_OR'] = result.loc[~mask, 'WOR']
        result.loc[~mask, 'Team2_DR'] = result.loc[~mask, 'WDR']
        
        # Calculate rebounding totals and differentials
        result['Team1_TR'] = result['Team1_OR'] + result['Team1_DR']
        result['Team2_TR'] = result['Team2_OR'] + result['Team2_DR']
        result['ReboundDiff'] = result['Team1_TR'] - result['Team2_TR']
        result['OffRebDiff'] = result['Team1_OR'] - result['Team2_OR']
        result['DefRebDiff'] = result['Team1_DR'] - result['Team2_DR']
    
    # Calculate assists, turnovers, steals, blocks, and fouls if we have stats
    if 'WAst' in games.columns and 'WTO' in games.columns:
        # Team1 stats
        mask = (result['WinA'] == 1)
        result.loc[mask, 'Team1_Ast'] = result.loc[mask, 'WAst']
        result.loc[mask, 'Team1_TO'] = result.loc[mask, 'WTO']
        result.loc[mask, 'Team1_Stl'] = result.loc[mask, 'WStl']
        result.loc[mask, 'Team1_Blk'] = result.loc[mask, 'WBlk']
        result.loc[mask, 'Team1_PF'] = result.loc[mask, 'WPF']
        
        result.loc[~mask, 'Team1_Ast'] = result.loc[~mask, 'LAst']
        result.loc[~mask, 'Team1_TO'] = result.loc[~mask, 'LTO']
        result.loc[~mask, 'Team1_Stl'] = result.loc[~mask, 'LStl']
        result.loc[~mask, 'Team1_Blk'] = result.loc[~mask, 'LBlk']
        result.loc[~mask, 'Team1_PF'] = result.loc[~mask, 'LPF']
        
        # Team2 stats
        result.loc[mask, 'Team2_Ast'] = result.loc[mask, 'LAst']
        result.loc[mask, 'Team2_TO'] = result.loc[mask, 'LTO']
        result.loc[mask, 'Team2_Stl'] = result.loc[mask, 'LStl']
        result.loc[mask, 'Team2_Blk'] = result.loc[mask, 'LBlk']
        result.loc[mask, 'Team2_PF'] = result.loc[mask, 'LPF']
        
        result.loc[~mask, 'Team2_Ast'] = result.loc[~mask, 'WAst']
        result.loc[~mask, 'Team2_TO'] = result.loc[~mask, 'WTO']
        result.loc[~mask, 'Team2_Stl'] = result.loc[~mask, 'WStl']
        result.loc[~mask, 'Team2_Blk'] = result.loc[~mask, 'WBlk']
        result.loc[~mask, 'Team2_PF'] = result.loc[~mask, 'WPF']
        
        # Calculate differentials
        result['AstDiff'] = result['Team1_Ast'] - result['Team2_Ast']
        result['TODiff'] = result['Team1_TO'] - result['Team2_TO']
        result['StlDiff'] = result['Team1_Stl'] - result['Team2_Stl']
        result['BlkDiff'] = result['Team1_Blk'] - result['Team2_Blk']
        result['PFDiff'] = result['Team1_PF'] - result['Team2_PF']
        
        # Calculate assist-to-turnover ratio
        result['Team1_AstTORatio'] = result['Team1_Ast'] / result['Team1_TO'].replace(0, 1)
        result['Team2_AstTORatio'] = result['Team2_Ast'] / result['Team2_TO'].replace(0, 1)
        result['AstTORatioDiff'] = result['Team1_AstTORatio'] - result['Team2_AstTORatio']
    
    # Calculate Four Factors if we have all necessary statistics
    if all(col in result.columns for col in ['Team1_FGM', 'Team1_FGA', 'Team1_FGM3', 'Team1_OR', 'Team1_DR', 'Team1_TO', 'Team1_FTA']):
        # 1. Effective Field Goal Percentage: (FGM + 0.5 * 3PM) / FGA
        result['Team1_EffFGPct'] = (result['Team1_FGM'] + 0.5 * result['Team1_FGM3']) / result['Team1_FGA'].replace(0, 1)
        result['Team2_EffFGPct'] = (result['Team2_FGM'] + 0.5 * result['Team2_FGM3']) / result['Team2_FGA'].replace(0, 1)
        result['EffFGPctDiff'] = result['Team1_EffFGPct'] - result['Team2_EffFGPct']
        
        # Estimate possessions (assuming team possessions are roughly equal)
        result['Team1_Poss'] = result['Team1_FGA'] - result['Team1_OR'] + result['Team1_TO'] + (0.44 * result['Team1_FTA'])
        result['Team2_Poss'] = result['Team2_FGA'] - result['Team2_OR'] + result['Team2_TO'] + (0.44 * result['Team2_FTA'])
        
        # 2. Turnover Rate: TO / Possessions
        result['Team1_TORatio'] = result['Team1_TO'] / result['Team1_Poss'].replace(0, 1)
        result['Team2_TORatio'] = result['Team2_TO'] / result['Team2_Poss'].replace(0, 1)
        result['TORatioDiff'] = result['Team1_TORatio'] - result['Team2_TORatio']
        
        # 3. Offensive Rebounding Rate: OR / (OR + Opponent DR)
        result['Team1_OffRebRate'] = result['Team1_OR'] / (result['Team1_OR'] + result['Team2_DR']).replace(0, 1)
        result['Team2_OffRebRate'] = result['Team2_OR'] / (result['Team2_OR'] + result['Team1_DR']).replace(0, 1)
        result['OffRebRateDiff'] = result['Team1_OffRebRate'] - result['Team2_OffRebRate']
        
        # 4. Free Throw Rate: FTA / FGA
        result['Team1_FTRate'] = result['Team1_FTA'] / result['Team1_FGA'].replace(0, 1)
        result['Team2_FTRate'] = result['Team2_FTA'] / result['Team2_FGA'].replace(0, 1)
        result['FTRateDiff'] = result['Team1_FTRate'] - result['Team2_FTRate']
    
    print("关键统计差异特征计算完成")
    return result

def add_historical_tournament_performance(games, seed_dict, num_years=3):
    """
    Add historical tournament performance features
    
    Args:
        games: DataFrame with game data
        seed_dict: Dictionary with seed information
        num_years: Number of previous years to include
        
    Returns:
        DataFrame with added historical tournament features
    """
    print("Calculating historical tournament performance features...")
    
    # Make a copy to avoid modifying the original DataFrame
    result = games.copy()
    
    # Create dictionary to store tournament history by team
    tournament_history = {}
    
    # Get all seasons in the data
    seasons = sorted(games['Season'].unique())
    
    # For each team in the current season, look back at their tournament history
    current_season = max(seasons)
    
    # Extract all TeamIDs from the current season
    current_teams = set()
    for _, row in games[games['Season'] == current_season].iterrows():
        current_teams.add(row['Team1'])
        current_teams.add(row['Team2'])
    
    # For each team, build tournament history
    for team_id in current_teams:
        team_history = []
        
        # Look back at past tournaments
        for season in reversed(seasons):
            if season == current_season:
                continue
                
            # Get team's seed in that season
            seed_key = f"{season}_{team_id}"
            if seed_key in seed_dict:
                seed = seed_dict[seed_key]
                
                # Get their tournament games in that season
                team_tournament_games = games[
                    (games['Season'] == season) & 
                    (games['GameType'] == 'Tournament') & 
                    ((games['Team1'] == team_id) | (games['Team2'] == team_id))
                ]
                
                # Count wins and calculate deepest round
                wins = 0
                for _, row in team_tournament_games.iterrows():
                    if (row['Team1'] == team_id and row['WinA'] == 1) or (row['Team2'] == team_id and row['WinA'] == 0):
                        wins += 1
                        
                # Calculate metrics for this season
                tournament_appearance = 1
                season_data = {
                    'season': season,
                    'seed': seed,
                    'tournament_appearance': tournament_appearance,
                    'wins': wins,
                    'rounds_advanced': wins  # Simplified estimate
                }
                team_history.append(season_data)
        
        # Store the history for this team
        tournament_history[team_id] = team_history
    
    # Calculate historical tournament features
    for _, row in result.iterrows():
        team1_id = row['Team1']
        team2_id = row['Team2']
        season = row['Season']
        
        # Team 1 features
        if team1_id in tournament_history and tournament_history[team1_id]:
            history = tournament_history[team1_id]
            recent_history = history[:num_years]
            
            # Calculate average seed, appearances, and success over recent years
            avg_seed = sum(h['seed'] for h in recent_history) / len(recent_history) if recent_history else 16
            total_appearances = sum(h['tournament_appearance'] for h in recent_history)
            total_wins = sum(h['wins'] for h in recent_history)
            deepest_run = max((h['rounds_advanced'] for h in recent_history), default=0)
            
            # Calculate seed trend (negative means improving/lower seeds)
            if len(recent_history) >= 2:
                # Calculate average change in seed from year to year
                seed_changes = [history[i-1]['seed'] - history[i]['seed'] for i in range(1, len(history))]
                seed_trend = sum(seed_changes) / len(seed_changes) if seed_changes else 0
            else:
                seed_trend = 0
            
            # Set values
            result.loc[_, 'Team1_HistAvgSeed'] = avg_seed
            result.loc[_, 'Team1_HistTourneyAppearances'] = total_appearances
            result.loc[_, 'Team1_HistTourneyWins'] = total_wins
            result.loc[_, 'Team1_HistDeepestRun'] = deepest_run
            result.loc[_, 'Team1_SeedTrend'] = seed_trend
        else:
            # Default values for teams with no tournament history
            result.loc[_, 'Team1_HistAvgSeed'] = 16
            result.loc[_, 'Team1_HistTourneyAppearances'] = 0
            result.loc[_, 'Team1_HistTourneyWins'] = 0
            result.loc[_, 'Team1_HistDeepestRun'] = 0
            result.loc[_, 'Team1_SeedTrend'] = 0
        
        # Team 2 features (same calculations)
        if team2_id in tournament_history and tournament_history[team2_id]:
            history = tournament_history[team2_id]
            recent_history = history[:num_years]
            
            avg_seed = sum(h['seed'] for h in recent_history) / len(recent_history) if recent_history else 16
            total_appearances = sum(h['tournament_appearance'] for h in recent_history)
            total_wins = sum(h['wins'] for h in recent_history)
            deepest_run = max((h['rounds_advanced'] for h in recent_history), default=0)
            
            if len(recent_history) >= 2:
                seed_changes = [history[i-1]['seed'] - history[i]['seed'] for i in range(1, len(history))]
                seed_trend = sum(seed_changes) / len(seed_changes) if seed_changes else 0
            else:
                seed_trend = 0
            
            result.loc[_, 'Team2_HistAvgSeed'] = avg_seed
            result.loc[_, 'Team2_HistTourneyAppearances'] = total_appearances
            result.loc[_, 'Team2_HistTourneyWins'] = total_wins
            result.loc[_, 'Team2_HistDeepestRun'] = deepest_run
            result.loc[_, 'Team2_SeedTrend'] = seed_trend
        else:
            result.loc[_, 'Team2_HistAvgSeed'] = 16
            result.loc[_, 'Team2_HistTourneyAppearances'] = 0
            result.loc[_, 'Team2_HistTourneyWins'] = 0
            result.loc[_, 'Team2_HistDeepestRun'] = 0
            result.loc[_, 'Team2_SeedTrend'] = 0
    
    # Calculate differential features
    result['HistAvgSeedDiff'] = result['Team1_HistAvgSeed'] - result['Team2_HistAvgSeed']
    result['HistTourneyAppearancesDiff'] = result['Team1_HistTourneyAppearances'] - result['Team2_HistTourneyAppearances']
    result['HistTourneyWinsDiff'] = result['Team1_HistTourneyWins'] - result['Team2_HistTourneyWins']
    result['HistDeepestRunDiff'] = result['Team1_HistDeepestRun'] - result['Team2_HistDeepestRun']
    result['SeedTrendDiff'] = result['Team1_SeedTrend'] - result['Team2_SeedTrend']
    
    print("历史锦标赛表现特征计算完成")
    return result

def enhance_kenpom_features(games, kenpom_df):
    """
    Enhanced version of KenPom feature integration that ensures offensive (AdjO) and 
    defensive (AdjD) efficiency metrics are properly captured and transformed.
    
    Args:
        games: DataFrame with game information
        kenpom_df: DataFrame with KenPom metrics
        
    Returns:
        games: DataFrame with enhanced KenPom features
    """
    if kenpom_df.empty:
        print("Warning: KenPom DataFrame is empty. Setting default efficiency values.")
        result = games.copy()
        # Set default values
        result['Team1_AdjO'] = 100.0
        result['Team1_AdjD'] = 100.0
        result['Team2_AdjO'] = 100.0
        result['Team2_AdjD'] = 100.0
        result['AdjO_Diff'] = 0.0
        result['AdjD_Diff'] = 0.0
        result['AdjEM_Diff'] = 0.0
        return result
    
    print("Enhancing KenPom features with offensive and defensive efficiency...")
    result = games.copy()
    
    # Check for season column
    if 'Season' not in kenpom_df.columns:
        if 'Year' in kenpom_df.columns:
            kenpom_df.rename(columns={'Year': 'Season'}, inplace=True)
        else:
            print("Warning: No 'Season' column found in KenPom data. Using 2025 as default.")
            kenpom_df['Season'] = 2025
    
    # Convert Season to integer for safer merging
    kenpom_df['Season'] = kenpom_df['Season'].astype(int)
    
    # Identify the offensive and defensive efficiency columns
    offensive_col = None
    defensive_col = None
    efficiency_col = None
    
    # Look for offensive efficiency column
    if 'ORtg' in kenpom_df.columns:
        offensive_col = 'ORtg'
    elif 'AdjO' in kenpom_df.columns:
        offensive_col = 'AdjO'
    else:
        possible_o_cols = [col for col in kenpom_df.columns if any(term in col for term in ['Off', 'ORtg', 'Adj0', 'AdjO'])]
        if possible_o_cols:
            offensive_col = possible_o_cols[0]
            print(f"Using '{offensive_col}' as offensive efficiency metric.")
    
    # Look for defensive efficiency column
    if 'DRtg' in kenpom_df.columns:
        defensive_col = 'DRtg'
    elif 'AdjD' in kenpom_df.columns:
        defensive_col = 'AdjD'
    else:
        possible_d_cols = [col for col in kenpom_df.columns if any(term in col for term in ['Def', 'DRtg', 'AdjD'])]
        if possible_d_cols:
            defensive_col = possible_d_cols[0]
            print(f"Using '{defensive_col}' as defensive efficiency metric.")
    
    # Look for net efficiency column
    if 'NetRtg' in kenpom_df.columns:
        efficiency_col = 'NetRtg'
    elif 'AdjEM' in kenpom_df.columns:
        efficiency_col = 'AdjEM'
    else:
        possible_em_cols = [col for col in kenpom_df.columns if any(term in col for term in ['Net', 'EM', 'Eff'])]
        if possible_em_cols:
            efficiency_col = possible_em_cols[0]
            print(f"Using '{efficiency_col}' as net efficiency metric.")
    
    # Ensure numeric values
    if offensive_col:
        kenpom_df[offensive_col] = pd.to_numeric(kenpom_df[offensive_col], errors='coerce').fillna(100.0)
    if defensive_col:
        kenpom_df[defensive_col] = pd.to_numeric(kenpom_df[defensive_col], errors='coerce').fillna(100.0)
    if efficiency_col:
        kenpom_df[efficiency_col] = pd.to_numeric(kenpom_df[efficiency_col], errors='coerce').fillna(0.0)
    
    # Calculate net efficiency if we have offensive and defensive but no net
    if offensive_col and defensive_col and not efficiency_col:
        kenpom_df['AdjEM'] = kenpom_df[offensive_col] - kenpom_df[defensive_col]
        efficiency_col = 'AdjEM'
        print("Calculated AdjEM as the difference between offensive and defensive efficiency.")
    
    # Prepare for merging
    if 'Team' in kenpom_df.columns and 'TeamID' not in kenpom_df.columns:
        # We need to map team names to IDs
        # This would require team name mapping logic that depends on available data
        print("Warning: KenPom data has team names but not IDs. This requires mapping.")
        # For simplicity, we'll assume TeamID exists or has been mapped
    
    # Merge for Team1
    if offensive_col:
        kenpom_team1_o = kenpom_df[['Season', 'TeamID', offensive_col]].copy()
        kenpom_team1_o.rename(columns={'TeamID': 'Team1', offensive_col: 'Team1_AdjO'}, inplace=True)
        result = result.merge(kenpom_team1_o, how='left', on=['Season', 'Team1'])
    
    if defensive_col:
        kenpom_team1_d = kenpom_df[['Season', 'TeamID', defensive_col]].copy()
        kenpom_team1_d.rename(columns={'TeamID': 'Team1', defensive_col: 'Team1_AdjD'}, inplace=True)
        result = result.merge(kenpom_team1_d, how='left', on=['Season', 'Team1'])
    
    if efficiency_col:
        kenpom_team1_em = kenpom_df[['Season', 'TeamID', efficiency_col]].copy()
        kenpom_team1_em.rename(columns={'TeamID': 'Team1', efficiency_col: 'Team1_AdjEM'}, inplace=True)
        result = result.merge(kenpom_team1_em, how='left', on=['Season', 'Team1'])
    
    # Merge for Team2
    if offensive_col:
        kenpom_team2_o = kenpom_df[['Season', 'TeamID', offensive_col]].copy()
        kenpom_team2_o.rename(columns={'TeamID': 'Team2', offensive_col: 'Team2_AdjO'}, inplace=True)
        result = result.merge(kenpom_team2_o, how='left', on=['Season', 'Team2'])
    
    if defensive_col:
        kenpom_team2_d = kenpom_df[['Season', 'TeamID', defensive_col]].copy()
        kenpom_team2_d.rename(columns={'TeamID': 'Team2', defensive_col: 'Team2_AdjD'}, inplace=True)
        result = result.merge(kenpom_team2_d, how='left', on=['Season', 'Team2'])
    
    if efficiency_col:
        kenpom_team2_em = kenpom_df[['Season', 'TeamID', efficiency_col]].copy()
        kenpom_team2_em.rename(columns={'TeamID': 'Team2', efficiency_col: 'Team2_AdjEM'}, inplace=True)
        result = result.merge(kenpom_team2_em, how='left', on=['Season', 'Team2'])
    
    # Fill missing values with defaults
    if offensive_col:
        result['Team1_AdjO'] = result['Team1_AdjO'].fillna(100.0)
        result['Team2_AdjO'] = result['Team2_AdjO'].fillna(100.0)
        result['AdjO_Diff'] = result['Team1_AdjO'] - result['Team2_AdjO']
    else:
        result['Team1_AdjO'] = 100.0
        result['Team2_AdjO'] = 100.0
        result['AdjO_Diff'] = 0.0
    
    if defensive_col:
        result['Team1_AdjD'] = result['Team1_AdjD'].fillna(100.0)
        result['Team2_AdjD'] = result['Team2_AdjD'].fillna(100.0)
        # Note: Lower defensive rating is better (fewer points allowed)
        result['AdjD_Diff'] = result['Team2_AdjD'] - result['Team1_AdjD']
    else:
        result['Team1_AdjD'] = 100.0
        result['Team2_AdjD'] = 100.0
        result['AdjD_Diff'] = 0.0
    
    if efficiency_col:
        result['Team1_AdjEM'] = result['Team1_AdjEM'].fillna(0.0)
        result['Team2_AdjEM'] = result['Team2_AdjEM'].fillna(0.0)
        result['AdjEM_Diff'] = result['Team1_AdjEM'] - result['Team2_AdjEM']
    else:
        result['Team1_AdjEM'] = result['Team1_AdjO'] - result['Team1_AdjD']
        result['Team2_AdjEM'] = result['Team2_AdjO'] - result['Team2_AdjD']
        result['AdjEM_Diff'] = result['Team1_AdjEM'] - result['Team2_AdjEM']
    
    print("KenPom efficiency features enhanced with offensive and defensive components.")
    return result

def enhance_key_stat_differentials(games):
    """
    Enhance statistical differentials with additional important metrics:
    - Rebounding rate differentials (offensive, total)
    - Turnover rate differentials
    - Effective field goal percentage (eFG%) differentials
    - True shooting percentage (TS%) differentials
    - Free throw rate and percentage differentials
    
    These metrics are critical champion indicators based on winning team patterns.
    
    Args:
        games: DataFrame with game data
        
    Returns:
        DataFrame with additional key statistical differences
    """
    print("增强关键统计差异特征，添加篮板率、失误率、有效命中率等冠军指标...")
    
    # Make a copy to avoid modifying the original DataFrame
    result = games.copy()
    
    # Check if we're working with submission data or data that lacks detailed stats
    is_submission = 'WTeamID' not in result.columns or 'LTeamID' not in result.columns
    
    # Columns to check for detailed stats
    required_shooting_cols = ['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 
                             'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA']
    
    required_rebounding_cols = ['WOR', 'WDR', 'LOR', 'LDR']
    
    required_turnover_cols = ['WTO', 'LTO']
    
    # Alternative columns that might be present in already processed data
    alt_shooting_cols = ['Team1_FGM', 'Team1_FGA', 'Team1_FGM3', 'Team1_FGA3', 'Team1_FTM', 'Team1_FTA',
                         'Team2_FGM', 'Team2_FGA', 'Team2_FGM3', 'Team2_FGA3', 'Team2_FTM', 'Team2_FTA']
    
    alt_rebounding_cols = ['Team1_OR', 'Team1_DR', 'Team2_OR', 'Team2_DR']
    
    alt_turnover_cols = ['Team1_TO', 'Team2_TO']
    
    # Check if detailed statistics are available in either form
    has_shooting_stats = (all(col in games.columns for col in required_shooting_cols) or 
                         all(col in games.columns for col in alt_shooting_cols))
    
    has_rebounding_stats = (all(col in games.columns for col in required_rebounding_cols) or 
                           all(col in games.columns for col in alt_rebounding_cols))
    
    has_turnover_stats = (all(col in games.columns for col in required_turnover_cols) or 
                         all(col in games.columns for col in alt_turnover_cols))
    
    # Default critical indicators to ensure they exist
    critical_indicators = [
        'OffRebRateDiff',   # Offensive rebounding rate difference
        'FTRateDiff',       # Free throw rate difference 
        'FTPctDiff'         # Free throw percentage difference
    ]
    
    # If we don't have the necessary stats, set default values and return
    if not (has_shooting_stats and has_rebounding_stats and has_turnover_stats):
        print("  部分统计数据未找到，设置默认值...")
        
        # First check if any of these differential features already exist in the data
        missing_indicators = [ind for ind in critical_indicators if ind not in result.columns]
        
        # Set default values for critical metrics that are missing
        for indicator in missing_indicators:
            result[indicator] = 0.0
        
        # Create a composite champion indicator with default values if it doesn't exist
        if 'ChampionComposite' not in result.columns:
            # Create a basic composite score using whatever indicators we do have
            if 'OffRebRateDiff' in result.columns and 'FTRateDiff' in result.columns:
                result['ChampionComposite'] = (
                    0.4 * result['OffRebRateDiff'] + 
                    0.25 * -result['FTRateDiff']  # Negative because lower OR is better
                )
                
                # Add more components if available
                if 'FTPctDiff' in result.columns:
                    result['ChampionComposite'] += 0.15 * result['FTPctDiff']
            else:
                # Default value if we can't calculate
                result['ChampionComposite'] = 0.0
                
        if 'ChampionCompositeV2' not in result.columns:
            result['ChampionCompositeV2'] = 0.0
            
        print("  设置了默认值的关键指标:", missing_indicators)
        return result
    
    print("  计算详细的高级统计指标差异...")
    
    # Initialize team stats dictionaries
    team_shooting_stats = {}
    team_rebounding_stats = {}
    team_turnover_stats = {}
    
    # Check if GameType column exists to filter for regular season games
    if 'GameType' not in games.columns:
        print("GameType column not found, cannot distinguish between regular season and tournament. Adding default GameType column...")
        games['GameType'] = 'Regular'  # Default to regular season
    
    # Filter for only regular season games for statistics calculation
    regular_games = games[games['GameType'] == 'Regular'].copy()
    
    if regular_games.empty:
        print("No regular season games found, using all data for team statistics...")
        regular_games = games.copy()
    else:
        print("Using only regular season games for team statistics to avoid data leakage...")
        
    # Sort by Season and DayNum to ensure correct order if they exist
    if 'Season' in regular_games.columns and 'DayNum' in regular_games.columns:
        regular_games = regular_games.sort_values(by=['Season', 'DayNum']).reset_index(drop=True)
    
    # Process historical game data to calculate advanced team stats
    for _, row in regular_games.iterrows():
        season = row['Season']
        
        # Get teams - handle both raw and processed data formats
        if is_submission:
            wteam_id = row['Team1']
            lteam_id = row['Team2']
        else:
            wteam_id = row['WTeamID'] if 'WTeamID' in row else row['Team1']
            lteam_id = row['LTeamID'] if 'LTeamID' in row else row['Team2']
        
        # Use the Team1/Team2 columns if they're already processed, otherwise use W/L columns
        using_processed = 'Team1_FGM' in games.columns
        
        # Process shooting stats
        if has_shooting_stats:
            # For winning team
            if f"{season}_{wteam_id}" not in team_shooting_stats:
                team_shooting_stats[f"{season}_{wteam_id}"] = {
                    'fgm': 0, 'fga': 0, 'fg3m': 0, 'fg3a': 0, 
                    'ftm': 0, 'fta': 0, 'pts': 0, 'games': 0
                }
            
            # Add stats based on data format
            if using_processed:
                # If data is already processed to Team1/Team2 format
                if is_submission or (row['Team1'] == wteam_id):
                    team_shooting_stats[f"{season}_{wteam_id}"]['fgm'] += row['Team1_FGM']
                    team_shooting_stats[f"{season}_{wteam_id}"]['fga'] += row['Team1_FGA']
                    team_shooting_stats[f"{season}_{wteam_id}"]['fg3m'] += row['Team1_FGM3']
                    team_shooting_stats[f"{season}_{wteam_id}"]['fg3a'] += row['Team1_FGA3']
                    team_shooting_stats[f"{season}_{wteam_id}"]['ftm'] += row['Team1_FTM']
                    team_shooting_stats[f"{season}_{wteam_id}"]['fta'] += row['Team1_FTA']
                else:
                    team_shooting_stats[f"{season}_{wteam_id}"]['fgm'] += row['Team2_FGM']
                    team_shooting_stats[f"{season}_{wteam_id}"]['fga'] += row['Team2_FGA']
                    team_shooting_stats[f"{season}_{wteam_id}"]['fg3m'] += row['Team2_FGM3']
                    team_shooting_stats[f"{season}_{wteam_id}"]['fg3a'] += row['Team2_FGA3']
                    team_shooting_stats[f"{season}_{wteam_id}"]['ftm'] += row['Team2_FTM']
                    team_shooting_stats[f"{season}_{wteam_id}"]['fta'] += row['Team2_FTA']
            else:
                # Raw data format
                team_shooting_stats[f"{season}_{wteam_id}"]['fgm'] += row['WFGM']
                team_shooting_stats[f"{season}_{wteam_id}"]['fga'] += row['WFGA']
                team_shooting_stats[f"{season}_{wteam_id}"]['fg3m'] += row['WFGM3']
                team_shooting_stats[f"{season}_{wteam_id}"]['fg3a'] += row['WFGA3']
                team_shooting_stats[f"{season}_{wteam_id}"]['ftm'] += row['WFTM']
                team_shooting_stats[f"{season}_{wteam_id}"]['fta'] += row['WFTA']
            
            # Calculate points
            pts = (2 * (team_shooting_stats[f"{season}_{wteam_id}"]['fgm'] - team_shooting_stats[f"{season}_{wteam_id}"]['fg3m']) + 
                   3 * team_shooting_stats[f"{season}_{wteam_id}"]['fg3m'] + 
                   team_shooting_stats[f"{season}_{wteam_id}"]['ftm'])
            team_shooting_stats[f"{season}_{wteam_id}"]['pts'] += pts
            team_shooting_stats[f"{season}_{wteam_id}"]['games'] += 1
        
        # Process rebounding stats (similar structure for both teams)
        if has_rebounding_stats:
            # For winning team rebounding
            if f"{season}_{wteam_id}" not in team_rebounding_stats:
                team_rebounding_stats[f"{season}_{wteam_id}"] = {
                    'oreb': 0, 'dreb': 0, 'total_opp_dreb': 0, 'total_opp_oreb': 0, 'games': 0
                }
            
            # Add stats based on data format
            if using_processed:
                # If data is already processed to Team1/Team2 format
                if is_submission or (row['Team1'] == wteam_id):
                    team_rebounding_stats[f"{season}_{wteam_id}"]['oreb'] += row['Team1_OR']
                    team_rebounding_stats[f"{season}_{wteam_id}"]['dreb'] += row['Team1_DR']
                    team_rebounding_stats[f"{season}_{wteam_id}"]['total_opp_dreb'] += row['Team2_DR']
                    team_rebounding_stats[f"{season}_{wteam_id}"]['total_opp_oreb'] += row['Team2_OR']
                else:
                    team_rebounding_stats[f"{season}_{wteam_id}"]['oreb'] += row['Team2_OR']
                    team_rebounding_stats[f"{season}_{wteam_id}"]['dreb'] += row['Team2_DR']
                    team_rebounding_stats[f"{season}_{wteam_id}"]['total_opp_dreb'] += row['Team1_DR']
                    team_rebounding_stats[f"{season}_{wteam_id}"]['total_opp_oreb'] += row['Team1_OR']
            else:
                # Raw data format
                team_rebounding_stats[f"{season}_{wteam_id}"]['oreb'] += row['WOR']
                team_rebounding_stats[f"{season}_{wteam_id}"]['dreb'] += row['WDR']
                team_rebounding_stats[f"{season}_{wteam_id}"]['total_opp_dreb'] += row['LDR']
                team_rebounding_stats[f"{season}_{wteam_id}"]['total_opp_oreb'] += row['LOR']
            
            team_rebounding_stats[f"{season}_{wteam_id}"]['games'] += 1
            
            # For losing team rebounding
            if f"{season}_{lteam_id}" not in team_rebounding_stats:
                team_rebounding_stats[f"{season}_{lteam_id}"] = {
                    'oreb': 0, 'dreb': 0, 'total_opp_dreb': 0, 'total_opp_oreb': 0, 'games': 0
                }
            
            # Add stats based on data format
            if using_processed:
                # If data is already processed to Team1/Team2 format
                if is_submission or (row['Team1'] == lteam_id):
                    team_rebounding_stats[f"{season}_{lteam_id}"]['oreb'] += row['Team1_OR']
                    team_rebounding_stats[f"{season}_{lteam_id}"]['dreb'] += row['Team1_DR']
                    team_rebounding_stats[f"{season}_{lteam_id}"]['total_opp_dreb'] += row['Team2_DR']
                    team_rebounding_stats[f"{season}_{lteam_id}"]['total_opp_oreb'] += row['Team2_OR']
                else:
                    team_rebounding_stats[f"{season}_{lteam_id}"]['oreb'] += row['Team2_OR']
                    team_rebounding_stats[f"{season}_{lteam_id}"]['dreb'] += row['Team2_DR']
                    team_rebounding_stats[f"{season}_{lteam_id}"]['total_opp_dreb'] += row['Team1_DR']
                    team_rebounding_stats[f"{season}_{lteam_id}"]['total_opp_oreb'] += row['Team1_OR']
            else:
                # Raw data format
                team_rebounding_stats[f"{season}_{lteam_id}"]['oreb'] += row['LOR']
                team_rebounding_stats[f"{season}_{lteam_id}"]['dreb'] += row['LDR']
                team_rebounding_stats[f"{season}_{lteam_id}"]['total_opp_dreb'] += row['WDR']
                team_rebounding_stats[f"{season}_{lteam_id}"]['total_opp_oreb'] += row['WOR']
            
            team_rebounding_stats[f"{season}_{lteam_id}"]['games'] += 1
    
    # Calculate advanced shooting metrics for each team
    for team_key in team_shooting_stats:
        stats = team_shooting_stats[team_key]
        
        # Free throw rate (FTA/FGA)
        stats['ft_rate'] = stats['fta'] / stats['fga'] if stats['fga'] > 0 else 0.25
        
        # Free throw percentage
        stats['ft_pct'] = stats['ftm'] / stats['fta'] if stats['fta'] > 0 else 0.7
        
        # Effective field goal percentage: (FGM + 0.5 * 3PM) / FGA
        stats['efg_pct'] = (stats['fgm'] + 0.5 * stats['fg3m']) / stats['fga'] if stats['fga'] > 0 else 0.45
        
        # True shooting percentage: PTS / (2 * (FGA + 0.44 * FTA))
        stats['ts_pct'] = stats['pts'] / (2 * (stats['fga'] + 0.44 * stats['fta'])) if (stats['fga'] + 0.44 * stats['fta']) > 0 else 0.5
    
    # Calculate rebounding rates for each team
    for team_key in team_rebounding_stats:
        stats = team_rebounding_stats[team_key]
        
        # Offensive rebounding rate: ORB / (ORB + Opp DRB)
        total_oreb_opportunities = stats['oreb'] + stats['total_opp_dreb']
        stats['oreb_rate'] = stats['oreb'] / total_oreb_opportunities if total_oreb_opportunities > 0 else 0.3
        
        # Defensive rebounding rate: DRB / (DRB + Opp ORB)
        total_dreb_opportunities = stats['dreb'] + stats['total_opp_oreb']
        stats['dreb_rate'] = stats['dreb'] / total_dreb_opportunities if total_dreb_opportunities > 0 else 0.7
        
        # Total rebounding rate: (ORB + DRB) / (ORB + DRB + Opp ORB + Opp DRB)
        total_reb = stats['oreb'] + stats['dreb']
        total_opp_reb = stats['total_opp_oreb'] + stats['total_opp_dreb']
        stats['total_reb_rate'] = total_reb / (total_reb + total_opp_reb) if (total_reb + total_opp_reb) > 0 else 0.5
    
    # Calculate turnover rates for each team
    for team_key in team_turnover_stats:
        stats = team_turnover_stats[team_key]
        
        # Estimate possessions: FGA + 0.44*FTA + TO - ORB
        # Since we don't directly track possessions, this is an approximation
        # Using FGA and FTA as a proxy for possessions
        possessions = stats['fga'] + 0.44 * stats['fta'] if stats['fga'] > 0 else 100 * stats['games']
        
        # Turnover rate: TO / Possessions
        stats['to_rate'] = stats['to'] / possessions if possessions > 0 else 0.15
    
    # Map team ID keys based on data format
    # For submission data or already processed data
    team1_key = 'IDTeam1' if 'IDTeam1' in result.columns else None
    team2_key = 'IDTeam2' if 'IDTeam2' in result.columns else None
    
    if team1_key is None or team2_key is None:
        # Try to construct the keys from Season and Team columns
        if 'Season' in result.columns and 'Team1' in result.columns and 'Team2' in result.columns:
            result['IDTeam1'] = result.apply(lambda r: f"{r['Season']}_{r['Team1']}", axis=1)
            result['IDTeam2'] = result.apply(lambda r: f"{r['Season']}_{r['Team2']}", axis=1)
            team1_key = 'IDTeam1'
            team2_key = 'IDTeam2'
        else:
            print("  警告: 无法构造队伍ID键，使用默认值")
            # Set default values for critical metrics
            for indicator in critical_indicators:
                result[indicator] = 0.0
            
            # Create default composite indicators
            result['ChampionComposite'] = 0.0
            result['ChampionCompositeV2'] = 0.0
            return result
    
    # Add shooting features
    if team_shooting_stats:
        # Team1 shooting features
        result['Team1_FTRate'] = result[team1_key].map(lambda x: team_shooting_stats.get(x, {}).get('ft_rate', 0.25))
        result['Team1_FTPct'] = result[team1_key].map(lambda x: team_shooting_stats.get(x, {}).get('ft_pct', 0.7))
        result['Team1_EFGPct'] = result[team1_key].map(lambda x: team_shooting_stats.get(x, {}).get('efg_pct', 0.45))
        result['Team1_TSPct'] = result[team1_key].map(lambda x: team_shooting_stats.get(x, {}).get('ts_pct', 0.5))
        
        # Team2 shooting features
        result['Team2_FTRate'] = result[team2_key].map(lambda x: team_shooting_stats.get(x, {}).get('ft_rate', 0.25))
        result['Team2_FTPct'] = result[team2_key].map(lambda x: team_shooting_stats.get(x, {}).get('ft_pct', 0.7))
        result['Team2_EFGPct'] = result[team2_key].map(lambda x: team_shooting_stats.get(x, {}).get('efg_pct', 0.45))
        result['Team2_TSPct'] = result[team2_key].map(lambda x: team_shooting_stats.get(x, {}).get('ts_pct', 0.5))
        
        # Calculate shooting differentials
        result['FTRateDiff'] = result['Team1_FTRate'] - result['Team2_FTRate']
        result['FTPctDiff'] = result['Team1_FTPct'] - result['Team2_FTPct']
        result['EFGPctDiff'] = result['Team1_EFGPct'] - result['Team2_EFGPct']
        result['TSPctDiff'] = result['Team1_TSPct'] - result['Team2_TSPct']
    
    # Add rebounding features
    if team_rebounding_stats:
        # Team1 rebounding features
        result['Team1_OffRebRate'] = result[team1_key].map(lambda x: team_rebounding_stats.get(x, {}).get('oreb_rate', 0.3))
        result['Team1_DefRebRate'] = result[team1_key].map(lambda x: team_rebounding_stats.get(x, {}).get('dreb_rate', 0.7))
        result['Team1_TotalRebRate'] = result[team1_key].map(lambda x: team_rebounding_stats.get(x, {}).get('total_reb_rate', 0.5))
        
        # Team2 rebounding features
        result['Team2_OffRebRate'] = result[team2_key].map(lambda x: team_rebounding_stats.get(x, {}).get('oreb_rate', 0.3))
        result['Team2_DefRebRate'] = result[team2_key].map(lambda x: team_rebounding_stats.get(x, {}).get('dreb_rate', 0.7))
        result['Team2_TotalRebRate'] = result[team2_key].map(lambda x: team_rebounding_stats.get(x, {}).get('total_reb_rate', 0.5))
        
        # Calculate rebounding differentials
        result['OffRebRateDiff'] = result['Team1_OffRebRate'] - result['Team2_OffRebRate']
        result['DefRebRateDiff'] = result['Team1_DefRebRate'] - result['Team2_DefRebRate']
        result['TotalRebRateDiff'] = result['Team1_TotalRebRate'] - result['Team2_TotalRebRate']
    
    # Add turnover features
    # Turnover features disabled
    
    # Ensure we have all the critical champion indicators
    critical_indicators = [
        'OffRebRateDiff',   # Offensive rebounding rate difference
        'FTRateDiff',       # Free throw rate difference 
        'FTPctDiff'         # Free throw percentage difference
    ]
    
    # Check which indicators we're missing and initialize them if needed
    for indicator in critical_indicators:
        if indicator not in result.columns:
            print(f"  Missing critical indicator: {indicator}. Initializing with zeros.")
            result[indicator] = 0.0
    
    # Create a composite champion indicator based on these factors
    # This combines the 'Four Factors' indicators with some weighting
    # Weights based on research that shows eFG% is most important, followed by offensive rebounding, etc.
    result['ChampionComposite'] = (
        0.5 * result['OffRebRateDiff'] +           # 50% weight to shooting efficiency
        0.30 * result['FTRateDiff'] +      # 30% weight to getting to the line
        0.20 * result['FTPctDiff']            # 20% weight to free throw rate
    )
    
    # Create enhanced versions of the champion composite indicator
    if all(feat in result.columns for feat in ['TSPctDiff', 'TotalRebRateDiff', 'OffRebRateDiff', 'FTRateDiff']):
        # Create an enhanced champion composite score with updated weights
        result['ChampionCompositeV2'] = (
            0.40 * result['TSPctDiff'] +           # 40% weight to true shooting
            0.30 * result['TotalRebRateDiff'] +    # 30% weight to total rebounding
            0.20 * result['OffRebRateDiff'] +      # 20% additional weight to offensive boards
            0.10 * result['FTRateDiff']            # 10% weight to free throw rate
        )
    
    print("关键冠军指标增强完成，添加了以下新指标:")
    print("  - 有效投篮命中率差异 (EFGPctDiff)")
    print("  - 真实命中率差异 (TSPctDiff)")
    print("  - 罚球率及命中率差异 (FTRateDiff, FTPctDiff)")
    print("  - 进攻篮板率、防守篮板率及总篮板率差异")
    print("  - 冠军综合指标 (ChampionComposite, ChampionCompositeV2)")
    
    # Create an overall 'Four Factors' composite metric if all required metrics are available
    if all(feat in result.columns for feat in ['EFGPctDiff', 'TSPctDiff', 'OffRebRateDiff', 'DefRebRateDiff']):
        print("  所有关键指标可用，创建综合'四因素'指标...")
    
    return result

def merge_merged_kenpom_features(games, kenpom_df):
    """
    Merge KenPom features from the merged KenPom dataset into the games DataFrame.
    Includes enhanced missing value handling and feature transformation.
    
    Args:
        games: DataFrame with game data
        kenpom_df: DataFrame with KenPom data, including TeamID and Season
    
    Returns:
        Games DataFrame with added KenPom features
    """
    print("Merging merged KenPom features...")
    if 'TeamID' not in kenpom_df.columns or 'Season' not in kenpom_df.columns:
        print("Error: KenPom data missing required columns (TeamID or Season)")
        return games
    
    # 记录匹配前游戏数统计
    total_games = len(games)
    
    # 创建副本以避免修改原始数据
    kenpom_features = kenpom_df.copy()
    games_with_kenpom = games.copy()
    
    # 移除任何可能存在的旧KenPom特征列，避免重复合并
    kenpom_cols = [col for col in games_with_kenpom.columns if col.startswith(('KP_W_', 'KP_L_'))]
    if kenpom_cols:
        print(f"Removing {len(kenpom_cols)} existing KenPom columns before re-merging")
        games_with_kenpom = games_with_kenpom.drop(columns=kenpom_cols)
    
    # 列出需要排除的列
    excluded_columns = ['Rk', 'W-L', 'Year', 'Season', 'TeamID', 'Team_Name']
    print(f"Explicitly excluding columns: {excluded_columns}")
    
    # 定义所有需要使用的KenPom特征列
    # 包括明确的数值列和那些应该是数值但可能被解析为字符串的列
    potential_numeric_columns = [
        'NetRtg', 'ORtg', 'DRtg', 'AdjT', 'Luck',
        'SOS_NetRtg', 'SOS_ORtg', 'SOS_DRtg', 'NCSOS_NetRtg',
        # 排名列 - 也是有用的预测特征
        'ORtg_Rk', 'DRtg_Rk', 'AdjT_Rk', 'Luck_Rk',
        'SOS_NetRtg_Rk', 'SOS_ORtg_Rk', 'SOS_DRtg_Rk', 'NCSOS_NetRtg_Rk'
    ]
    
    # 检查这些列是否存在于数据中，并排除指定不使用的列
    available_columns = [col for col in potential_numeric_columns 
                        if col in kenpom_features.columns and col not in excluded_columns]
    print(f"Found {len(available_columns)} potential KenPom feature columns after exclusions")
    print(f"Available columns: {available_columns}")
    
    # 尝试将所有潜在数值列转换为数值类型
    for col in available_columns:
        try:
            # 如果列是对象类型，先尝试清理特殊字符
            if kenpom_features[col].dtype == 'object':
                # 移除百分比符号、空格和其他干扰字符
                kenpom_features[col] = kenpom_features[col].astype(str).str.replace('%', '')
                kenpom_features[col] = kenpom_features[col].astype(str).str.replace('$', '')
                kenpom_features[col] = kenpom_features[col].astype(str).str.strip()
                # 转换为数值
                kenpom_features[col] = pd.to_numeric(kenpom_features[col], errors='coerce')
                print(f"  Converted '{col}' from string to numeric")
        except Exception as e:
            print(f"  Warning: Could not convert '{col}' to numeric: {e}")
    
    # 现在确定最终要使用的数值列
    numeric_columns = [col for col in kenpom_features.select_dtypes(include=['number']).columns.tolist() 
                      if col not in excluded_columns]
    
    # 检查是否有数值列可用
    if not numeric_columns:
        print("Warning: No numeric KenPom features found for merging")
        return games_with_kenpom
    
    print(f"Final numeric KenPom features to merge: {len(numeric_columns)}")
    print(f"Features: {numeric_columns}")
    
    # 增强特征预处理 - 标准化/归一化数值特征
    for col in numeric_columns:
        # 只对数值列进行处理
        if col in kenpom_features.columns and pd.api.types.is_numeric_dtype(kenpom_features[col]):
            # 检查是否有极端异常值
            q1 = kenpom_features[col].quantile(0.01)
            q3 = kenpom_features[col].quantile(0.99)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            # 截断极端异常值
            kenpom_features[col] = kenpom_features[col].clip(lower_bound, upper_bound)
            
            # 保留原始缩放前的值（为调试目的）
            original_values = kenpom_features[col].copy()
            
            # 计算均值和标准差（用于标准化，处理后均值为0，标准差为1）
            col_mean = kenpom_features[col].mean()
            col_std = kenpom_features[col].std()
            
            # 只有当数据有足够的变异性时才进行标准化
            if col_std > 1e-10:  # 避免除以接近零的值
                # 应用标准化
                kenpom_features[col] = (kenpom_features[col] - col_mean) / col_std
                
                # 记录缩放的效果
                print(f"  Standardized {col}: Before [min={original_values.min():.2f}, max={original_values.max():.2f}], "
                      f"After [min={kenpom_features[col].min():.2f}, max={kenpom_features[col].max():.2f}]")
    
    # 在TeamID列上添加特殊处理，解决未匹配的球队问题
    enhanced_kenpom = kenpom_features.copy()
    
    # 修复未匹配的球队名称问题 - 检查列是否存在
    print("添加特殊球队名称映射...")
    
    # 1. 查找NC State相关记录
    if 'Team_Name' in kenpom_features.columns:
        nc_state_mask = kenpom_features['Team_Name'].str.contains('N.C. State', case=False, na=False)
        if nc_state_mask.any():
            print(f"Found {nc_state_mask.sum()} NC State records. Adding mapping to TeamID 1314 (North Carolina State)")
            enhanced_kenpom.loc[nc_state_mask, 'TeamID'] = 1314
    
    # 2. 查找Stephen F. Austin相关记录
    if 'Team_Name' in kenpom_features.columns:
        sfa_mask = kenpom_features['Team_Name'].str.contains('Stephen F', case=False, na=False)
        if sfa_mask.any():
            print(f"Found {sfa_mask.sum()} Stephen F. Austin records. Adding mapping to TeamID 1386 (Stephen F Austin)")
            enhanced_kenpom.loc[sfa_mask, 'TeamID'] = 1386  # 使用估计的ID
    
    # 3. 查找UMass Lowell相关记录
    if 'Team_Name' in kenpom_features.columns:
        umass_lowell_mask = kenpom_features['Team_Name'].str.contains('UMass Lowell', case=False, na=False)
        if umass_lowell_mask.any():
            print(f"Found {umass_lowell_mask.sum()} UMass Lowell records. Adding mapping to TeamID 1265 (Massachusetts Lowell)")
            enhanced_kenpom.loc[umass_lowell_mask, 'TeamID'] = 1265  # 使用估计的ID
    
    # 4. 查找USC Upstate相关记录
    if 'Team_Name' in kenpom_features.columns:
        usc_upstate_mask = kenpom_features['Team_Name'].str.contains('USC Upstate', case=False, na=False)
        if usc_upstate_mask.any():
            print(f"Found {usc_upstate_mask.sum()} USC Upstate records. Adding mapping to TeamID 1423 (South Carolina Upstate)")
            enhanced_kenpom.loc[usc_upstate_mask, 'TeamID'] = 1423  # 使用估计的ID
    
    # 统计更新后的匹配率
    missing_before = kenpom_features['TeamID'].isna().mean() * 100
    missing_after = enhanced_kenpom['TeamID'].isna().mean() * 100
    print(f"Manual team mapping improved match rate: {missing_before:.1f}% missing -> {missing_after:.1f}% missing")
    
    # 使用增强后的数据集进行合并
    kenpom_features = enhanced_kenpom
    
    # 检查数据集中的团队ID列名
    # 对于比赛数据，可能使用WTeamID/LTeamID或Team1ID/Team2ID
    team1_id_col = 'WTeamID' if 'WTeamID' in games_with_kenpom.columns else 'Team1ID'
    team2_id_col = 'LTeamID' if 'LTeamID' in games_with_kenpom.columns else 'Team2ID'
    
    if team1_id_col not in games_with_kenpom.columns or team2_id_col not in games_with_kenpom.columns:
        print(f"Error: Required team ID columns not found. Needed {team1_id_col} and {team2_id_col}.")
        return games_with_kenpom
    
    print(f"Using team ID columns: {team1_id_col} and {team2_id_col}")
    
    # 3. 为获胜队和失败队合并KenPom特征
    # 先为第一支队伍合并
    print(f"Merging KenPom features for {team1_id_col}...")
    wins_with_kp = games_with_kenpom.merge(
        kenpom_features[['TeamID', 'Season'] + numeric_columns],
        left_on=['Season', team1_id_col],
        right_on=['Season', 'TeamID'],
        how='left'
    )
    
    # 计算第一支队伍的匹配率
    team1_match_count = wins_with_kp['TeamID'].notna().sum()
    team1_match_rate = team1_match_count / total_games * 100
    print(f"Team 1 match rate: {team1_match_count}/{total_games} ({team1_match_rate:.1f}%)")
    
    # 重命名第一支队伍的特征列 (使用KP_W_前缀保持一致性)
    for col in numeric_columns:
        if col in wins_with_kp.columns:
            wins_with_kp.rename(columns={col: f'KP_W_{col}'}, inplace=True)
    
    # 从合并结果中删除临时TeamID列
    if 'TeamID' in wins_with_kp.columns:
        wins_with_kp.drop(columns=['TeamID'], inplace=True)
    
    # 再为第二支队伍合并
    print(f"Merging KenPom features for {team2_id_col}...")
    games_with_kp = wins_with_kp.merge(
        kenpom_features[['TeamID', 'Season'] + numeric_columns],
        left_on=['Season', team2_id_col],
        right_on=['Season', 'TeamID'],
        how='left'
    )
    
    # 计算第二支队伍的匹配率
    team2_match_count = games_with_kp['TeamID'].notna().sum()
    team2_match_rate = team2_match_count / total_games * 100
    print(f"Team 2 match rate: {team2_match_count}/{total_games} ({team2_match_rate:.1f}%)")
    
    # 重命名第二支队伍的特征列 (使用KP_L_前缀保持一致性)
    for col in numeric_columns:
        if col in games_with_kp.columns:
            games_with_kp.rename(columns={col: f'KP_L_{col}'}, inplace=True)
    
    # 从合并结果中删除临时TeamID列
    if 'TeamID' in games_with_kp.columns:
        games_with_kp.drop(columns=['TeamID'], inplace=True)
    
    # 4. 高级缺失值处理 - 根据种子排名和类似球队填充
    print("\n执行高级缺失值填充...")
    
    # 按赛季统计缺失值情况
    seasons = sorted(games_with_kp['Season'].unique())
    print("缺失值按赛季统计（合并前）:")
    for season in seasons:
        season_df = games_with_kp[games_with_kp['Season'] == season]
        
        # 选择所有KenPom特征列
        kp_columns = [col for col in games_with_kp.columns if col.startswith(('KP_W_', 'KP_L_'))]
        
        # 计算每个赛季的缺失值百分比
        missing_pct = season_df[kp_columns].isna().mean().mean() * 100
        print(f"  Season {season}: {missing_pct:.1f}% KenPom features missing")
    
    # 确定哪些列需要填充 - 只填充数值列
    kp_w_cols = [f'KP_W_{col}' for col in numeric_columns]
    kp_l_cols = [f'KP_L_{col}' for col in numeric_columns]
    all_kp_cols = kp_w_cols + kp_l_cols
    
    # 总体缺失值统计
    missing_before = games_with_kp[all_kp_cols].isna().mean().mean() * 100
    print(f"Overall missing rate before imputation: {missing_before:.1f}%")
    
    # 修改种子列名检查
    team1_seed_col = 'WSeed' if 'WSeed' in games_with_kp.columns else 'Team1Seed'
    team2_seed_col = 'LSeed' if 'LSeed' in games_with_kp.columns else 'Team2Seed'
    
    # 基于种子排名的填充策略
    if team1_seed_col in games_with_kp.columns and team2_seed_col in games_with_kp.columns:
        print("使用种子排名进行缺失值填充...")
        
        # 对每个赛季分别处理
        for season in seasons:
            season_mask = games_with_kp['Season'] == season
            season_games = games_with_kp[season_mask]
            
            # 为第一支队伍填充
            for col in kp_w_cols:
                # 只处理包含缺失值的数值列
                if col in games_with_kp.columns and games_with_kp[col].isna().any():
                    # 按种子排名分组计算均值
                    seed_means = season_games.groupby(team1_seed_col)[col].mean()
                    
                    # 应用填充 - 使用相同种子的平均值
                    for seed, mean_val in seed_means.items():
                        if not pd.isna(mean_val):
                            seed_mask = (games_with_kp['Season'] == season) & (games_with_kp[team1_seed_col] == seed) & games_with_kp[col].isna()
                            if seed_mask.any():
                                games_with_kp.loc[seed_mask, col] = mean_val
            
            # 为第二支队伍填充
            for col in kp_l_cols:
                # 只处理包含缺失值的数值列
                if col in games_with_kp.columns and games_with_kp[col].isna().any():
                    # 按种子排名分组计算均值
                    seed_means = season_games.groupby(team2_seed_col)[col].mean()
                    
                    # 应用填充 - 使用相同种子的平均值
                    for seed, mean_val in seed_means.items():
                        if not pd.isna(mean_val):
                            seed_mask = (games_with_kp['Season'] == season) & (games_with_kp[team2_seed_col] == seed) & games_with_kp[col].isna()
                            if seed_mask.any():
                                games_with_kp.loc[seed_mask, col] = mean_val
        
        # 记录基于种子填充后的缺失率
        missing_after_seed = games_with_kp[all_kp_cols].isna().mean().mean() * 100
        print(f"Missing rate after seed-based imputation: {missing_after_seed:.1f}%")
    else:
        # 如果没有种子数据，设置一个默认值
        missing_after_seed = missing_before
        print(f"没有找到种子列 {team1_seed_col} 或 {team2_seed_col}，跳过基于种子的填充")
    
    # 5. 最终缺失值处理
    # 对于仍然缺失的值，使用全局均值或中位数填充
    print("对剩余缺失值使用全局方法填充...")
    
    # 先获取每个特征的统计信息做为备份
    feature_stats = {}
    for col in all_kp_cols:
        if col in games_with_kp.columns:
            feature_stats[col] = {
                'mean': games_with_kp[col].mean(),
                'median': games_with_kp[col].median(),
                'min': games_with_kp[col].min(),
                'max': games_with_kp[col].max()
            }
    
    # 对没有填充的缺失值执行最终填充
    for col in all_kp_cols:
        if col in games_with_kp.columns and games_with_kp[col].isna().any():
            # 首先尝试使用当前数据中的均值
            if not pd.isna(feature_stats[col]['mean']):
                # 使用均值填充 - 使用loc方法避免警告
                missing_mask = games_with_kp[col].isna()
                games_with_kp.loc[missing_mask, col] = feature_stats[col]['mean']
            # 如果均值也是NaN，那么使用中位数
            elif not pd.isna(feature_stats[col]['median']):
                missing_mask = games_with_kp[col].isna()
                games_with_kp.loc[missing_mask, col] = feature_stats[col]['median']
            # 如果均值和中位数都是NaN，使用0填充（假设是标准化后的数据）
            else:
                missing_mask = games_with_kp[col].isna()
                games_with_kp.loc[missing_mask, col] = 0
                print(f"  Warning: Using 0 for column {col} as mean and median are NaN")
    
    # 记录最终填充后的缺失率
    missing_after_final = games_with_kp[all_kp_cols].isna().mean().mean() * 100
    print(f"Missing rate after all imputations: {missing_after_final:.1f}%")
    
    # 打印各阶段填充的缺失率变化
    print("\n各阶段填充进度:")
    print(f"  初始缺失率: {missing_before:.1f}%")
    print(f"  种子填充后: {missing_after_seed:.1f}%")
    print(f"  最终填充后: {missing_after_final:.1f}%")
    
    # 6. 创建KenPom特征差异值 - 获胜队与失败队指标差
    print("\n创建KenPom指标差异特征...")
    for base_col in numeric_columns:
        w_col = f'KP_W_{base_col}'
        l_col = f'KP_L_{base_col}'
        
        if w_col in games_with_kp.columns and l_col in games_with_kp.columns:
            # 创建差异列
            diff_col = f'KP_DIFF_{base_col}'
            games_with_kp[diff_col] = games_with_kp[w_col] - games_with_kp[l_col]
            
            # 保持数据类型一致
            if pd.api.types.is_numeric_dtype(games_with_kp[w_col]):
                games_with_kp[diff_col] = pd.to_numeric(games_with_kp[diff_col])
    
    # 7. 为特定重要指标创建比率特征
    key_metrics = ['NetRtg', 'ORtg', 'DRtg', 'AdjT', 'Luck']
    for metric in key_metrics:
        w_col = f'KP_W_{metric}'
        l_col = f'KP_L_{metric}'
        
        if w_col in games_with_kp.columns and l_col in games_with_kp.columns:
            # 处理分母为0或接近0的情况
            eps = 1e-10  # 小的常数，避免除0
            
            # 计算比率（所有比率都大于1，便于解释）
            ratio_col = f'KP_RATIO_{metric}'
            
            # 处理正负值的不同比率计算方法
            is_w_negative = (games_with_kp[w_col] < 0)
            is_l_negative = (games_with_kp[l_col] < 0)
            
            # 初始化比率列
            games_with_kp[ratio_col] = np.nan
            
            # 情况1: 两者都为正 - 直接计算比率
            mask_both_pos = (~is_w_negative) & (~is_l_negative)
            if mask_both_pos.any():
                # 确保W/L比率大于1
                games_with_kp.loc[mask_both_pos, ratio_col] = np.where(
                    games_with_kp.loc[mask_both_pos, w_col] >= games_with_kp.loc[mask_both_pos, l_col],
                    games_with_kp.loc[mask_both_pos, w_col] / (games_with_kp.loc[mask_both_pos, l_col] + eps),
                    games_with_kp.loc[mask_both_pos, l_col] / (games_with_kp.loc[mask_both_pos, w_col] + eps)
                )
            
            # 情况2: 两者都为负 - 计算比率并反转
            mask_both_neg = is_w_negative & is_l_negative
            if mask_both_neg.any():
                # 对于负值，使用绝对值大小，确保比率大于1
                games_with_kp.loc[mask_both_neg, ratio_col] = np.where(
                    abs(games_with_kp.loc[mask_both_neg, w_col]) <= abs(games_with_kp.loc[mask_both_neg, l_col]),
                    abs(games_with_kp.loc[mask_both_neg, l_col]) / (abs(games_with_kp.loc[mask_both_neg, w_col]) + eps),
                    abs(games_with_kp.loc[mask_both_neg, w_col]) / (abs(games_with_kp.loc[mask_both_neg, l_col]) + eps)
                )
            
            # 情况3: 一正一负 - 使用和而不是比率
            mask_mixed = (is_w_negative & ~is_l_negative) | (~is_w_negative & is_l_negative)
            if mask_mixed.any():
                # 对于一正一负的情况，使用绝对值之和作为"比率"
                games_with_kp.loc[mask_mixed, ratio_col] = abs(games_with_kp.loc[mask_mixed, w_col]) + abs(games_with_kp.loc[mask_mixed, l_col])
    
    # 8. 创建综合效率指标
    # 检查关键指标是否可用
    if all(f'KP_DIFF_{m}' in games_with_kp.columns for m in ['NetRtg', 'ORtg', 'DRtg']):
        # 创建KenPom综合效率指标
        games_with_kp['KP_EfficiencyComposite'] = (
            0.5 * games_with_kp['KP_DIFF_NetRtg'] + 
            0.3 * games_with_kp['KP_DIFF_ORtg'] + 
            0.2 * -games_with_kp['KP_DIFF_DRtg']  # 反转DRtg（防守效率），因为低值更好
        )
    
    # 更高级的复合指标
    if all(f'KP_W_{m}' in games_with_kp.columns for m in ['ORtg', 'DRtg', 'Luck']):
        games_with_kp['KP_TeamQualityIndex'] = (
            games_with_kp['KP_W_ORtg'] - games_with_kp['KP_W_DRtg'] + 10 * games_with_kp['KP_W_Luck']
        ) - (
            games_with_kp['KP_L_ORtg'] - games_with_kp['KP_L_DRtg'] + 10 * games_with_kp['KP_L_Luck']
        )
    
    # 打印完成信息
    print(f"Successfully merged KenPom features: Added {len(all_kp_cols)} base features "
          f"and {len(numeric_columns)} difference features, plus composite metrics")
    
    # 识别并标记任何剩余的缺失值
    missing_after = {col: games_with_kp[col].isna().mean() * 100 for col in games_with_kp.columns if games_with_kp[col].isna().any()}
    if missing_after:
        print("\n缺失值情况（最终）:")
        for col, pct in sorted(missing_after.items(), key=lambda x: x[1], reverse=True):
            if pct > 0 and col in all_kp_cols:
                print(f"  {col}: {pct:.2f}% missing")
    else:
        print("最终数据集中没有KenPom特征缺失值！")
    
    return games_with_kp