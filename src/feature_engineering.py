import numpy as np
import pandas as pd
import re

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
    print("计算团队统计特征（仅使用常规赛数据，避免数据泄露）...")
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
        print("GameType列未找到，无法区分常规赛和锦标赛。添加默认GameType列...")
        games['GameType'] = 'Regular'  # 默认为常规赛
    
    # Create a copy with only regular season data
    regular_games = games[games['GameType'] == 'Regular'].copy()
    
    if regular_games.empty:
        print("没有找到常规赛数据，使用所有数据计算团队统计信息...")
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
    
    print(f"团队统计特征计算完成，共处理了 {len(team_stats)} 个团队-赛季组合")
    
    return games, team_stats

def add_head_to_head_features(games):
    """
    For each game, compute head-to-head statistics for Team1 vs Team2 using previous seasons' data.
    If no historical matchup, assign default values.
    
    Args:
        games: DataFrame with game data
        
    Returns:
        DataFrame with head-to-head features added
    """
    print("计算历史交锋特征...")
    
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
    
    print("历史交锋特征计算完成")
    return games

def add_recent_performance_features(games, window=5):
    """
    For each game, calculate recent performance features for each team within the same season.
    Features include recent win rate, average score differential, and momentum indicators.
    
    Args:
        games: DataFrame with game data
        window: Number of recent games to consider
        
    Returns:
        DataFrame with recent performance features added
    """
    if 'DayNum' not in games.columns:
        print("DayNum列未找到，跳过近期表现特征计算。")
        games['Team1_RecentWinRate'] = 0.5
        games['Team2_RecentWinRate'] = 0.5
        games['Team1_RecentScoreDiff'] = 0.0
        games['Team2_RecentScoreDiff'] = 0.0
        return games
    
    print(f"计算近期表现特征（窗口大小={window}场比赛）...")
    
    def recent_stats(row, team, df):
        season = row['Season']
        current_day = row['DayNum']
        
        # Get all games from this season before the current game
        past_games = df[(df['Season'] == season) & (df['DayNum'] < current_day)]
        
        # Filter for games involving this team
        team_games = past_games[(past_games['Team1'] == team) | (past_games['Team2'] == team)]
        
        # Sort by DayNum descending to get most recent first
        team_games = team_games.sort_values('DayNum', ascending=False)
        
        # Take only the last 'window' games
        recent_games = team_games.head(window)
        
        if recent_games.empty:
            return {
                'win_rate': 0.5,
                'score_diff': 0.0,
                'streak': 0,
                'avg_score': 60.0,
                'avg_allowed': 60.0,
                'games_count': 0
            }
        
        # Calculate win rate
        team1_wins = len(recent_games[(recent_games['Team1'] == team) & (recent_games['WinA'] == 1)])
        team2_wins = len(recent_games[(recent_games['Team2'] == team) & (recent_games['WinA'] == 0)])
        total_wins = team1_wins + team2_wins
        win_rate = total_wins / len(recent_games)
        
        # Calculate score differential
        score_diffs = []
        # Also track streak and points
        current_streak = 0
        won_last = None
        total_score = 0
        total_allowed = 0
        
        for _, game in recent_games.iterrows():
            # Determine if team won
            team_won = (game['Team1'] == team and game['WinA'] == 1) or (game['Team2'] == team and game['WinA'] == 0)
            
            # Score differential
            if 'ScoreDiffNorm' in game:
                if game['Team1'] == team:
                    score_diffs.append(game['ScoreDiffNorm'])
                else:
                    score_diffs.append(-game['ScoreDiffNorm'])
            
            # Calculate streak
            if won_last is None:
                # First game in the sequence
                won_last = team_won
                current_streak = 1 if team_won else -1
            elif team_won == won_last:
                # Streak continues
                if team_won:
                    current_streak += 1
                else:
                    current_streak -= 1
            else:
                # Streak broken, reset
                won_last = team_won
                current_streak = 1 if team_won else -1
            
            # Add scores
            if 'WScore' in game and 'LScore' in game:
                if game['Team1'] == team:
                    if game['WinA'] == 1:
                        total_score += game['WScore']
                        total_allowed += game['LScore']
                    else:
                        total_score += game['LScore']
                        total_allowed += game['WScore']
                else:
                    if game['WinA'] == 0:
                        total_score += game['WScore']
                        total_allowed += game['LScore']
                    else:
                        total_score += game['LScore']
                        total_allowed += game['WScore']
        
        avg_score_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0.0
        avg_score = total_score / len(recent_games) if len(recent_games) > 0 else 60.0
        avg_allowed = total_allowed / len(recent_games) if len(recent_games) > 0 else 60.0
        
        return {
            'win_rate': win_rate,
            'score_diff': avg_score_diff,
            'streak': current_streak,
            'avg_score': avg_score,
            'avg_allowed': avg_allowed,
            'games_count': len(recent_games)
        }
    
    # Apply the function to get recent stats for each team
    recent_stats_team1 = games.apply(lambda r: recent_stats(r, r['Team1'], games), axis=1)
    recent_stats_team2 = games.apply(lambda r: recent_stats(r, r['Team2'], games), axis=1)
    
    # Extract the values and set them as new columns
    games['Team1_RecentWinRate'] = recent_stats_team1.apply(lambda x: x['win_rate'])
    games['Team1_RecentScoreDiff'] = recent_stats_team1.apply(lambda x: x['score_diff'])
    games['Team1_Streak'] = recent_stats_team1.apply(lambda x: x['streak'])
    games['Team1_RecentAvgScore'] = recent_stats_team1.apply(lambda x: x['avg_score'])
    games['Team1_RecentAvgAllowed'] = recent_stats_team1.apply(lambda x: x['avg_allowed'])
    games['Team1_RecentGamesCount'] = recent_stats_team1.apply(lambda x: x['games_count'])
    
    games['Team2_RecentWinRate'] = recent_stats_team2.apply(lambda x: x['win_rate'])
    games['Team2_RecentScoreDiff'] = recent_stats_team2.apply(lambda x: x['score_diff'])
    games['Team2_Streak'] = recent_stats_team2.apply(lambda x: x['streak'])
    games['Team2_RecentAvgScore'] = recent_stats_team2.apply(lambda x: x['avg_score'])
    games['Team2_RecentAvgAllowed'] = recent_stats_team2.apply(lambda x: x['avg_allowed'])
    games['Team2_RecentGamesCount'] = recent_stats_team2.apply(lambda x: x['games_count'])
    
    # Calculate difference features
    games['RecentWinRateDiff'] = games['Team1_RecentWinRate'] - games['Team2_RecentWinRate']
    games['RecentScoreDiffDiff'] = games['Team1_RecentScoreDiff'] - games['Team2_RecentScoreDiff']
    games['StreakDiff'] = games['Team1_Streak'] - games['Team2_Streak']
    games['RecentAvgScoreDiff'] = games['Team1_RecentAvgScore'] - games['Team2_RecentAvgScore']
    games['RecentAvgAllowedDiff'] = games['Team1_RecentAvgAllowed'] - games['Team2_RecentAvgAllowed']
    
    # Create momentum features (combinations)
    games['Team1_Momentum'] = games['Team1_RecentWinRate'] * (games['Team1_Streak'] / window)
    games['Team2_Momentum'] = games['Team2_RecentWinRate'] * (games['Team2_Streak'] / window)
    games['MomentumDiff'] = games['Team1_Momentum'] - games['Team2_Momentum']
    
    print("近期表现特征计算完成")
    return games

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
    gb = regular_games.groupby('SeasonIDTeams').agg({col: agg_funcs for col in available_cols}).reset_index()
    
    # Split SeasonIDTeams back into Season and IDTeams
    gb[['Season', 'IDTeams']] = gb['SeasonIDTeams'].str.split('_', n=1, expand=True)
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
    Add Elo rating features to the games dataframe.
    
    Args:
        games: DataFrame with game data
        k_factor: K-factor for Elo calculation (controls how quickly ratings change)
        initial_elo: Starting Elo rating for each team
        reset_each_season: Whether to reset ratings at the start of each season
        
    Returns:
        DataFrame with added Elo rating features
    """
    print("计算Elo评分特征...")
    
    # Initialize team Elo ratings
    team_elos = {}
    
    # Check if 'DayNum' column exists for sorting
    if 'DayNum' in games.columns:
        # Sort games by Season and DayNum to ensure chronological order
        games_sorted = games.sort_values(by=['Season', 'DayNum']).reset_index(drop=True)
    else:
        # For submission data which may not have DayNum, just use Season
        print("  'DayNum' column not found. Assuming pre-sorted or submission data.")
        games_sorted = games.copy()
    
    # Add columns for Elo ratings
    games_sorted['Team1_Elo'] = initial_elo
    games_sorted['Team2_Elo'] = initial_elo
    games_sorted['EloDiff'] = 0
    games_sorted['EloWinProbability'] = 0.5
    
    # Keep track of last season for potential reset
    last_season = None
    
    # Loop through games in chronological order
    for idx, row in games_sorted.iterrows():
        season = row['Season']
        
        # Reset ratings at the start of each season if requested
        if reset_each_season and season != last_season:
            # Determine if we should reset or adjust ratings
            if last_season is not None:
                # Reset but preserve team hierarchy (regress towards mean)
                for team_id in team_elos:
                    team_elos[team_id] = 1500 + 0.75 * (team_elos[team_id] - 1500)
            last_season = season
        
        # Get team IDs
        team1_id = row['Team1']
        team2_id = row['Team2']
        
        # Initialize Elo ratings if not present
        if team1_id not in team_elos:
            team_elos[team1_id] = initial_elo
        if team2_id not in team_elos:
            team_elos[team2_id] = initial_elo
        
        # Record pre-game Elo ratings
        games_sorted.at[idx, 'Team1_Elo'] = team_elos[team1_id]
        games_sorted.at[idx, 'Team2_Elo'] = team_elos[team2_id]
        games_sorted.at[idx, 'EloDiff'] = team_elos[team1_id] - team_elos[team2_id]
        
        # Calculate win probability based on Elo difference
        elo_diff = team_elos[team1_id] - team_elos[team2_id]
        win_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        games_sorted.at[idx, 'EloWinProbability'] = win_prob
        
        # Update Elo ratings based on game outcome if this is historical data
        if 'WinA' in row:
            # Determine actual outcome (1 if Team1 won, 0 if Team2 won)
            actual_result = row['WinA']
            
            # Calculate Elo updates
            # Adjust K-factor based on margin of victory if score data is available
            k = k_factor
            if 'ScoreDiff' in row and 'WinA' in row:
                # Get score difference in absolute terms
                score_diff_abs = abs(row['ScoreDiff'])
                # Add a bonus for dominant victories
                k = k_factor * (1 + 0.1 * (score_diff_abs / 10))
            
            # Calculate Elo updates
            elo_update = k * (actual_result - win_prob)
            
            # Apply updates
            team_elos[team1_id] += elo_update
            team_elos[team2_id] -= elo_update
    
    # Make sure column order is preserved
    result = games.copy()
    result['Team1_Elo'] = games_sorted['Team1_Elo']
    result['Team2_Elo'] = games_sorted['Team2_Elo']
    result['EloDiff'] = games_sorted['EloDiff']
    result['EloWinProbability'] = games_sorted['EloWinProbability']
    
    print(f"Elo评分特征计算完成，共处理了 {len(team_elos)} 支队伍")
    
    return result

def add_strength_of_schedule(games, team_stats, current_season=None):
    """
    Add Strength of Schedule (SoS) features based on opponent quality.
    
    Args:
        games: DataFrame with game data
        team_stats: Dictionary mapping Season_TeamID to team statistics
        current_season: Current season year (if None, use max Season in games)
        
    Returns:
        DataFrame with added Strength of Schedule features
    """
    print("计算赛程强度（Strength of Schedule）特征...")
    
    if current_season is None:
        current_season = games['Season'].max()
    
    # Check if team_stats is properly populated
    if not team_stats:
        print("  WARNING: team_stats dictionary is empty. Using default SoS values.")
        # Set default SoS values
        games['Team1_SOS_WinRate'] = 0.5
        games['Team1_SOS_OffRating'] = 60
        games['Team1_SOS_DefRating'] = 60
        games['Team1_SOS_Combined'] = 0
        
        games['Team2_SOS_WinRate'] = 0.5
        games['Team2_SOS_OffRating'] = 60
        games['Team2_SOS_DefRating'] = 60
        games['Team2_SOS_Combined'] = 0
        
        # Calculate difference features (all zero in this case)
        games['SOS_WinRateDiff'] = 0
        games['SOS_OffRatingDiff'] = 0
        games['SOS_DefRatingDiff'] = 0
        games['SOS_CombinedDiff'] = 0
        
        print("  赛程强度特征设置为默认值")
        return games
    
    # Initialize dictionaries to store SoS metrics
    sos_metrics = {}
    
    # For each team and season, calculate average opponent statistics
    for season in games['Season'].unique():
        season_games = games[games['Season'] == season]
        
        # For each team, collect opponent data
        team_opponents = {}
        
        # Process games in chronological order if DayNum is available
        if 'DayNum' in season_games.columns:
            season_games = season_games.sort_values('DayNum')
        
        # Collect opponent data for each team
        for _, row in season_games.iterrows():
            team1_id = row['Team1']
            team2_id = row['Team2']
            
            # Initialize opponent lists if needed
            if f"{season}_{team1_id}" not in team_opponents:
                team_opponents[f"{season}_{team1_id}"] = []
            if f"{season}_{team2_id}" not in team_opponents:
                team_opponents[f"{season}_{team2_id}"] = []
            
            # Add each team as opponent to the other
            team_opponents[f"{season}_{team1_id}"].append(team2_id)
            team_opponents[f"{season}_{team2_id}"].append(team1_id)
        
        # Calculate SoS metrics for each team
        for team_key, opponents in team_opponents.items():
            # Get opponent team_stats keys
            opponent_keys = [f"{season}_{opponent}" for opponent in opponents]
            
            # Calculate average opponent win rate
            opponent_win_rates = [team_stats.get(key, {}).get('win_rate', 0.5) for key in opponent_keys]
            avg_opp_win_rate = sum(opponent_win_rates) / len(opponent_win_rates) if opponent_win_rates else 0.5
            
            # Calculate average opponent offensive and defensive ratings (if available)
            opponent_off_ratings = [team_stats.get(key, {}).get('avg_score', 60) for key in opponent_keys]
            opponent_def_ratings = [team_stats.get(key, {}).get('avg_allowed', 60) for key in opponent_keys]
            
            avg_opp_off_rating = sum(opponent_off_ratings) / len(opponent_off_ratings) if opponent_off_ratings else 60
            avg_opp_def_rating = sum(opponent_def_ratings) / len(opponent_def_ratings) if opponent_def_ratings else 60
            
            # Store SoS metrics
            sos_metrics[team_key] = {
                'sos_win_rate': avg_opp_win_rate,
                'sos_off_rating': avg_opp_off_rating,
                'sos_def_rating': avg_opp_def_rating,
                'sos_combined': avg_opp_win_rate * (avg_opp_off_rating - avg_opp_def_rating),
                'num_opponents': len(opponents)
            }
    
    # Create a result DataFrame
    result = games.copy()
    
    # Add SoS features to the games dataframe
    result['Team1_SOS_WinRate'] = result['IDTeam1'].map(lambda x: sos_metrics.get(x, {}).get('sos_win_rate', 0.5))
    result['Team1_SOS_OffRating'] = result['IDTeam1'].map(lambda x: sos_metrics.get(x, {}).get('sos_off_rating', 60))
    result['Team1_SOS_DefRating'] = result['IDTeam1'].map(lambda x: sos_metrics.get(x, {}).get('sos_def_rating', 60))
    result['Team1_SOS_Combined'] = result['IDTeam1'].map(lambda x: sos_metrics.get(x, {}).get('sos_combined', 0))
    
    result['Team2_SOS_WinRate'] = result['IDTeam2'].map(lambda x: sos_metrics.get(x, {}).get('sos_win_rate', 0.5))
    result['Team2_SOS_OffRating'] = result['IDTeam2'].map(lambda x: sos_metrics.get(x, {}).get('sos_off_rating', 60))
    result['Team2_SOS_DefRating'] = result['IDTeam2'].map(lambda x: sos_metrics.get(x, {}).get('sos_def_rating', 60))
    result['Team2_SOS_Combined'] = result['IDTeam2'].map(lambda x: sos_metrics.get(x, {}).get('sos_combined', 0))
    
    # Calculate difference features
    result['SOS_WinRateDiff'] = result['Team1_SOS_WinRate'] - result['Team2_SOS_WinRate']
    result['SOS_OffRatingDiff'] = result['Team1_SOS_OffRating'] - result['Team2_SOS_OffRating']
    result['SOS_DefRatingDiff'] = result['Team1_SOS_DefRating'] - result['Team2_SOS_DefRating']
    result['SOS_CombinedDiff'] = result['Team1_SOS_Combined'] - result['Team2_SOS_Combined']
    
    print(f"赛程强度特征计算完成，共处理了 {len(sos_metrics)} 个团队-赛季组合")
    
    return result

def add_key_stat_differentials(games):
    """
    Add key statistical differential features that are predictive of game outcomes.
    These include rebounding rates, turnover rates, shooting percentages, etc.
    
    Args:
        games: DataFrame with game data
        
    Returns:
        DataFrame with added statistical differential features
    """
    print("计算关键统计差异特征...")
    
    # Columns that should be available in the detailed results data
    required_shooting_cols = ['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 
                              'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA']
    required_rebounding_cols = ['WOR', 'WDR', 'LOR', 'LDR']
    required_other_stats = ['WAst', 'WTO', 'WStl', 'WBlk', 'LAst', 'LTO', 'LStl', 'LBlk']
    
    # Check if detailed statistics are available
    has_shooting_stats = all(col in games.columns for col in required_shooting_cols)
    has_rebounding_stats = all(col in games.columns for col in required_rebounding_cols)
    has_other_stats = all(col in games.columns for col in required_other_stats)
    
    result = games.copy()
    
    if not has_shooting_stats and not has_rebounding_stats and not has_other_stats:
        print("  No detailed game statistics found. This is likely submission data.")
        print("  Setting default statistical values for differentials.")
        
        # Set default values for important differentials
        result['FGPctDiff'] = 0
        result['3PPctDiff'] = 0
        result['FTPctDiff'] = 0
        result['TSPctDiff'] = 0
        result['EFGPctDiff'] = 0
        result['OffRebRateDiff'] = 0
        result['DefRebRateDiff'] = 0
        result['TurnoverRateDiff'] = 0
        result['AssistRateDiff'] = 0
        
        # Return early since we don't have the necessary statistics
        return result
    
    # Add shooting percentage features
    if has_shooting_stats:
        print("  添加投篮命中率特征...")
        
        # Initialize stat dictionaries
        team_shooting_stats = {}
        
        # Process historical game data to calculate team shooting stats
        for _, row in games.iterrows():
            season = row['Season']
            
            # Get teams
            wteam_id = row['WTeamID']
            lteam_id = row['LTeamID']
            
            # Process winning team stats
            if f"{season}_{wteam_id}" not in team_shooting_stats:
                team_shooting_stats[f"{season}_{wteam_id}"] = {
                    'fgm': 0, 'fga': 0, 'fg3m': 0, 'fg3a': 0, 'ftm': 0, 'fta': 0,
                    'games': 0
                }
            
            team_shooting_stats[f"{season}_{wteam_id}"]['fgm'] += row['WFGM']
            team_shooting_stats[f"{season}_{wteam_id}"]['fga'] += row['WFGA']
            team_shooting_stats[f"{season}_{wteam_id}"]['fg3m'] += row['WFGM3']
            team_shooting_stats[f"{season}_{wteam_id}"]['fg3a'] += row['WFGA3']
            team_shooting_stats[f"{season}_{wteam_id}"]['ftm'] += row['WFTM']
            team_shooting_stats[f"{season}_{wteam_id}"]['fta'] += row['WFTA']
            team_shooting_stats[f"{season}_{wteam_id}"]['games'] += 1
            
            # Process losing team stats
            if f"{season}_{lteam_id}" not in team_shooting_stats:
                team_shooting_stats[f"{season}_{lteam_id}"] = {
                    'fgm': 0, 'fga': 0, 'fg3m': 0, 'fg3a': 0, 'ftm': 0, 'fta': 0,
                    'games': 0
                }
            
            team_shooting_stats[f"{season}_{lteam_id}"]['fgm'] += row['LFGM']
            team_shooting_stats[f"{season}_{lteam_id}"]['fga'] += row['LFGA']
            team_shooting_stats[f"{season}_{lteam_id}"]['fg3m'] += row['LFGM3']
            team_shooting_stats[f"{season}_{lteam_id}"]['fg3a'] += row['LFGA3']
            team_shooting_stats[f"{season}_{lteam_id}"]['ftm'] += row['LFTM']
            team_shooting_stats[f"{season}_{lteam_id}"]['fta'] += row['LFTA']
            team_shooting_stats[f"{season}_{lteam_id}"]['games'] += 1
        
        # Calculate shooting percentage stats
        for team_key in team_shooting_stats:
            stats = team_shooting_stats[team_key]
            
            # Field goal percentage (overall)
            stats['fg_pct'] = stats['fgm'] / stats['fga'] if stats['fga'] > 0 else 0.45
            
            # 3-point percentage
            stats['fg3_pct'] = stats['fg3m'] / stats['fg3a'] if stats['fg3a'] > 0 else 0.33
            
            # Free throw percentage
            stats['ft_pct'] = stats['ftm'] / stats['fta'] if stats['fta'] > 0 else 0.7
            
            # 2-point percentage (excluding 3-pointers)
            fg2m = stats['fgm'] - stats['fg3m']
            fg2a = stats['fga'] - stats['fg3a']
            stats['fg2_pct'] = fg2m / fg2a if fg2a > 0 else 0.5
            
            # True shooting percentage (accounts for 3-pointers and free throws)
            points = (stats['fgm'] - stats['fg3m']) * 2 + stats['fg3m'] * 3 + stats['ftm']
            true_shooting_attempts = stats['fga'] + 0.44 * stats['fta']
            stats['ts_pct'] = points / (2 * true_shooting_attempts) if true_shooting_attempts > 0 else 0.5
            
            # Effective field goal percentage (accounts for 3-pointers being worth more)
            stats['efg_pct'] = (stats['fgm'] + 0.5 * stats['fg3m']) / stats['fga'] if stats['fga'] > 0 else 0.5
        
        # Add shooting features to games
        result['Team1_FGPct'] = result['IDTeam1'].map(lambda x: team_shooting_stats.get(x, {}).get('fg_pct', 0.45))
        result['Team1_3PPct'] = result['IDTeam1'].map(lambda x: team_shooting_stats.get(x, {}).get('fg3_pct', 0.33))
        result['Team1_FTPct'] = result['IDTeam1'].map(lambda x: team_shooting_stats.get(x, {}).get('ft_pct', 0.7))
        result['Team1_2PPct'] = result['IDTeam1'].map(lambda x: team_shooting_stats.get(x, {}).get('fg2_pct', 0.5))
        result['Team1_TSPct'] = result['IDTeam1'].map(lambda x: team_shooting_stats.get(x, {}).get('ts_pct', 0.5))
        result['Team1_EFGPct'] = result['IDTeam1'].map(lambda x: team_shooting_stats.get(x, {}).get('efg_pct', 0.5))
        
        result['Team2_FGPct'] = result['IDTeam2'].map(lambda x: team_shooting_stats.get(x, {}).get('fg_pct', 0.45))
        result['Team2_3PPct'] = result['IDTeam2'].map(lambda x: team_shooting_stats.get(x, {}).get('fg3_pct', 0.33))
        result['Team2_FTPct'] = result['IDTeam2'].map(lambda x: team_shooting_stats.get(x, {}).get('ft_pct', 0.7))
        result['Team2_2PPct'] = result['IDTeam2'].map(lambda x: team_shooting_stats.get(x, {}).get('fg2_pct', 0.5))
        result['Team2_TSPct'] = result['IDTeam2'].map(lambda x: team_shooting_stats.get(x, {}).get('ts_pct', 0.5))
        result['Team2_EFGPct'] = result['IDTeam2'].map(lambda x: team_shooting_stats.get(x, {}).get('efg_pct', 0.5))
        
        # Calculate differentials
        result['FGPctDiff'] = result['Team1_FGPct'] - result['Team2_FGPct']
        result['3PPctDiff'] = result['Team1_3PPct'] - result['Team2_3PPct']
        result['FTPctDiff'] = result['Team1_FTPct'] - result['Team2_FTPct']
        result['2PPctDiff'] = result['Team1_2PPct'] - result['Team2_2PPct']
        result['TSPctDiff'] = result['Team1_TSPct'] - result['Team2_TSPct']
        result['EFGPctDiff'] = result['Team1_EFGPct'] - result['Team2_EFGPct']
    else:
        print("  没有找到投篮命中率相关列，跳过投篮命中率特征计算")
        # Set default values for shooting differentials
        result['FGPctDiff'] = 0
        result['3PPctDiff'] = 0
        result['FTPctDiff'] = 0
        result['TSPctDiff'] = 0
        result['EFGPctDiff'] = 0
    
    # Add rebounding rate features
    if has_rebounding_stats:
        print("  添加篮板率特征...")
        
        # Initialize stat dictionaries
        team_rebounding_stats = {}
        
        # Process historical game data to calculate team rebounding stats
        for _, row in games.iterrows():
            season = row['Season']
            
            # Get teams
            wteam_id = row['WTeamID']
            lteam_id = row['LTeamID']
            
            # Get rebounding stats from the game
            w_offensive_reb = row['WOR']
            w_defensive_reb = row['WDR']
            l_offensive_reb = row['LOR']
            l_defensive_reb = row['LDR']
            
            # Calculate total rebounds
            w_total_reb = w_offensive_reb + w_defensive_reb
            l_total_reb = l_offensive_reb + l_defensive_reb
            
            # Calculate offensive rebounding opportunities
            w_off_reb_opps = l_defensive_reb + w_offensive_reb
            l_off_reb_opps = w_defensive_reb + l_offensive_reb
            
            # Process winning team stats
            if f"{season}_{wteam_id}" not in team_rebounding_stats:
                team_rebounding_stats[f"{season}_{wteam_id}"] = {
                    'offensive_reb': 0, 'defensive_reb': 0, 'total_reb': 0,
                    'off_reb_opps': 0, 'def_reb_opps': 0, 'games': 0,
                    'opponent_off_reb': 0, 'opponent_def_reb': 0,
                }
            
            team_rebounding_stats[f"{season}_{wteam_id}"]['offensive_reb'] += w_offensive_reb
            team_rebounding_stats[f"{season}_{wteam_id}"]['defensive_reb'] += w_defensive_reb
            team_rebounding_stats[f"{season}_{wteam_id}"]['total_reb'] += w_total_reb
            team_rebounding_stats[f"{season}_{wteam_id}"]['off_reb_opps'] += w_off_reb_opps
            team_rebounding_stats[f"{season}_{wteam_id}"]['def_reb_opps'] += (w_defensive_reb + l_offensive_reb)
            team_rebounding_stats[f"{season}_{wteam_id}"]['opponent_off_reb'] += l_offensive_reb
            team_rebounding_stats[f"{season}_{wteam_id}"]['opponent_def_reb'] += l_defensive_reb
            team_rebounding_stats[f"{season}_{wteam_id}"]['games'] += 1
            
            # Process losing team stats
            if f"{season}_{lteam_id}" not in team_rebounding_stats:
                team_rebounding_stats[f"{season}_{lteam_id}"] = {
                    'offensive_reb': 0, 'defensive_reb': 0, 'total_reb': 0,
                    'off_reb_opps': 0, 'def_reb_opps': 0, 'games': 0,
                    'opponent_off_reb': 0, 'opponent_def_reb': 0,
                }
            
            team_rebounding_stats[f"{season}_{lteam_id}"]['offensive_reb'] += l_offensive_reb
            team_rebounding_stats[f"{season}_{lteam_id}"]['defensive_reb'] += l_defensive_reb
            team_rebounding_stats[f"{season}_{lteam_id}"]['total_reb'] += l_total_reb
            team_rebounding_stats[f"{season}_{lteam_id}"]['off_reb_opps'] += l_off_reb_opps
            team_rebounding_stats[f"{season}_{lteam_id}"]['def_reb_opps'] += (l_defensive_reb + w_offensive_reb)
            team_rebounding_stats[f"{season}_{lteam_id}"]['opponent_off_reb'] += w_offensive_reb
            team_rebounding_stats[f"{season}_{lteam_id}"]['opponent_def_reb'] += w_defensive_reb
            team_rebounding_stats[f"{season}_{lteam_id}"]['games'] += 1
        
        # Calculate rebounding rate stats
        for team_key in team_rebounding_stats:
            stats = team_rebounding_stats[team_key]
            
            # Offensive rebounding rate
            stats['offensive_reb_rate'] = stats['offensive_reb'] / stats['off_reb_opps'] if stats['off_reb_opps'] > 0 else 0.3
            
            # Defensive rebounding rate
            stats['defensive_reb_rate'] = stats['defensive_reb'] / stats['def_reb_opps'] if stats['def_reb_opps'] > 0 else 0.7
            
            # Total rebounding rate
            total_reb_opps = stats['off_reb_opps'] + stats['def_reb_opps']
            stats['total_reb_rate'] = stats['total_reb'] / total_reb_opps if total_reb_opps > 0 else 0.5
            
            # Rebounding differential per game
            total_opponent_reb = stats['opponent_off_reb'] + stats['opponent_def_reb']
            stats['reb_diff_per_game'] = (stats['total_reb'] - total_opponent_reb) / stats['games'] if stats['games'] > 0 else 0
        
        # Add rebounding features to games
        result['Team1_OffRebRate'] = result['IDTeam1'].map(lambda x: team_rebounding_stats.get(x, {}).get('offensive_reb_rate', 0.3))
        result['Team1_DefRebRate'] = result['IDTeam1'].map(lambda x: team_rebounding_stats.get(x, {}).get('defensive_reb_rate', 0.7))
        result['Team1_TotalRebRate'] = result['IDTeam1'].map(lambda x: team_rebounding_stats.get(x, {}).get('total_reb_rate', 0.5))
        result['Team1_RebDiffPerGame'] = result['IDTeam1'].map(lambda x: team_rebounding_stats.get(x, {}).get('reb_diff_per_game', 0))
        
        result['Team2_OffRebRate'] = result['IDTeam2'].map(lambda x: team_rebounding_stats.get(x, {}).get('offensive_reb_rate', 0.3))
        result['Team2_DefRebRate'] = result['IDTeam2'].map(lambda x: team_rebounding_stats.get(x, {}).get('defensive_reb_rate', 0.7))
        result['Team2_TotalRebRate'] = result['IDTeam2'].map(lambda x: team_rebounding_stats.get(x, {}).get('total_reb_rate', 0.5))
        result['Team2_RebDiffPerGame'] = result['IDTeam2'].map(lambda x: team_rebounding_stats.get(x, {}).get('reb_diff_per_game', 0))
        
        # Calculate differentials
        result['OffRebRateDiff'] = result['Team1_OffRebRate'] - result['Team2_OffRebRate']
        result['DefRebRateDiff'] = result['Team1_DefRebRate'] - result['Team2_DefRebRate']
        result['TotalRebRateDiff'] = result['Team1_TotalRebRate'] - result['Team2_TotalRebRate']
        result['RebDiffPerGameDiff'] = result['Team1_RebDiffPerGame'] - result['Team2_RebDiffPerGame']
    else:
        print("  没有找到篮板率相关列，跳过篮板率特征计算")
        # Set default values for rebounding differentials
        result['OffRebRateDiff'] = 0
        result['DefRebRateDiff'] = 0
        result['TotalRebRateDiff'] = 0
    
    # Add turnover rate and other advanced stats
    if has_other_stats:
        print("  添加失误率和其他高级统计特征...")
        
        # Initialize stat dictionaries
        team_advanced_stats = {}
        
        # Process historical game data to calculate advanced team stats
        for _, row in games.iterrows():
            season = row['Season']
            
            # Get teams
            wteam_id = row['WTeamID']
            lteam_id = row['LTeamID']
            
            # Get advanced stats from the game
            w_assists = row['WAst']
            w_turnovers = row['WTO']
            w_steals = row['WStl']
            w_blocks = row['WBlk']
            
            l_assists = row['LAst']
            l_turnovers = row['LTO']
            l_steals = row['LStl']
            l_blocks = row['LBlk']
            
            # Estimate possessions (simple method)
            w_fg_attempts = row['WFGA'] if 'WFGA' in row else 0
            l_fg_attempts = row['LFGA'] if 'LFGA' in row else 0
            w_ft_attempts = row['WFTA'] if 'WFTA' in row else 0
            l_ft_attempts = row['LFTA'] if 'LFTA' in row else 0
            w_offensive_reb = row['WOR'] if 'WOR' in row else 0
            l_offensive_reb = row['LOR'] if 'LOR' in row else 0
            
            # Estimate possessions for each team
            w_possessions = w_fg_attempts + 0.44 * w_ft_attempts - w_offensive_reb + w_turnovers
            l_possessions = l_fg_attempts + 0.44 * l_ft_attempts - l_offensive_reb + l_turnovers
            
            # Process winning team stats
            if f"{season}_{wteam_id}" not in team_advanced_stats:
                team_advanced_stats[f"{season}_{wteam_id}"] = {
                    'assists': 0, 'turnovers': 0, 'steals': 0, 'blocks': 0,
                    'possessions': 0, 'fgm': 0, 'games': 0
                }
            
            team_advanced_stats[f"{season}_{wteam_id}"]['assists'] += w_assists
            team_advanced_stats[f"{season}_{wteam_id}"]['turnovers'] += w_turnovers
            team_advanced_stats[f"{season}_{wteam_id}"]['steals'] += w_steals
            team_advanced_stats[f"{season}_{wteam_id}"]['blocks'] += w_blocks
            team_advanced_stats[f"{season}_{wteam_id}"]['possessions'] += w_possessions
            team_advanced_stats[f"{season}_{wteam_id}"]['fgm'] += row['WFGM'] if 'WFGM' in row else 0
            team_advanced_stats[f"{season}_{wteam_id}"]['games'] += 1
            
            # Process losing team stats
            if f"{season}_{lteam_id}" not in team_advanced_stats:
                team_advanced_stats[f"{season}_{lteam_id}"] = {
                    'assists': 0, 'turnovers': 0, 'steals': 0, 'blocks': 0,
                    'possessions': 0, 'fgm': 0, 'games': 0
                }
            
            team_advanced_stats[f"{season}_{lteam_id}"]['assists'] += l_assists
            team_advanced_stats[f"{season}_{lteam_id}"]['turnovers'] += l_turnovers
            team_advanced_stats[f"{season}_{lteam_id}"]['steals'] += l_steals
            team_advanced_stats[f"{season}_{lteam_id}"]['blocks'] += l_blocks
            team_advanced_stats[f"{season}_{lteam_id}"]['possessions'] += l_possessions
            team_advanced_stats[f"{season}_{lteam_id}"]['fgm'] += row['LFGM'] if 'LFGM' in row else 0
            team_advanced_stats[f"{season}_{lteam_id}"]['games'] += 1
        
        # Calculate advanced stats
        for team_key in team_advanced_stats:
            stats = team_advanced_stats[team_key]
            
            # Turnover rate (turnovers per possession)
            stats['turnover_rate'] = stats['turnovers'] / stats['possessions'] if stats['possessions'] > 0 else 0.15
            
            # Assist rate (assists per field goal made)
            stats['assist_rate'] = stats['assists'] / stats['fgm'] if stats['fgm'] > 0 else 0.5
            
            # Steals per game
            stats['steals_per_game'] = stats['steals'] / stats['games'] if stats['games'] > 0 else 5
            
            # Blocks per game
            stats['blocks_per_game'] = stats['blocks'] / stats['games'] if stats['games'] > 0 else 3
            
            # Assist-to-turnover ratio
            stats['ast_to_ratio'] = stats['assists'] / stats['turnovers'] if stats['turnovers'] > 0 else 1.5
        
        # Add advanced stats features to games
        result['Team1_TurnoverRate'] = result['IDTeam1'].map(lambda x: team_advanced_stats.get(x, {}).get('turnover_rate', 0.15))
        result['Team1_AssistRate'] = result['IDTeam1'].map(lambda x: team_advanced_stats.get(x, {}).get('assist_rate', 0.5))
        result['Team1_StealsPerGame'] = result['IDTeam1'].map(lambda x: team_advanced_stats.get(x, {}).get('steals_per_game', 5))
        result['Team1_BlocksPerGame'] = result['IDTeam1'].map(lambda x: team_advanced_stats.get(x, {}).get('blocks_per_game', 3))
        result['Team1_AstToRatio'] = result['IDTeam1'].map(lambda x: team_advanced_stats.get(x, {}).get('ast_to_ratio', 1.5))
        
        result['Team2_TurnoverRate'] = result['IDTeam2'].map(lambda x: team_advanced_stats.get(x, {}).get('turnover_rate', 0.15))
        result['Team2_AssistRate'] = result['IDTeam2'].map(lambda x: team_advanced_stats.get(x, {}).get('assist_rate', 0.5))
        result['Team2_StealsPerGame'] = result['IDTeam2'].map(lambda x: team_advanced_stats.get(x, {}).get('steals_per_game', 5))
        result['Team2_BlocksPerGame'] = result['IDTeam2'].map(lambda x: team_advanced_stats.get(x, {}).get('blocks_per_game', 3))
        result['Team2_AstToRatio'] = result['IDTeam2'].map(lambda x: team_advanced_stats.get(x, {}).get('ast_to_ratio', 1.5))
        
        # Calculate differentials
        result['TurnoverRateDiff'] = result['Team1_TurnoverRate'] - result['Team2_TurnoverRate']
        result['AssistRateDiff'] = result['Team1_AssistRate'] - result['Team2_AssistRate']
        result['StealsPerGameDiff'] = result['Team1_StealsPerGame'] - result['Team2_StealsPerGame']
        result['BlocksPerGameDiff'] = result['Team1_BlocksPerGame'] - result['Team2_BlocksPerGame']
        result['AstToRatioDiff'] = result['Team1_AstToRatio'] - result['Team2_AstToRatio']
    else:
        print("  没有找到进阶统计相关列，跳过进阶统计特征计算")
        # Set default values for advanced stat differentials
        result['TurnoverRateDiff'] = 0
        result['AssistRateDiff'] = 0
    
    print(f"关键统计差异特征计算完成")
    
    return result

def add_historical_tournament_performance(games, seed_dict, num_years=3):
    """
    Add features based on teams' historical NCAA tournament performance.
    Tracks seed history, tournament success, and progression patterns.
    
    Args:
        games: DataFrame with game data
        seed_dict: Dictionary mapping Season_TeamID to seed value
        num_years: Number of previous years to consider for historical trends
        
    Returns:
        DataFrame with added historical tournament features
    """
    print("计算历史锦标赛表现特征...")
    
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
    Enhance statistical differentials with additional important metrics like free throw rate
    and ensure we have all the key champion indicators.
    
    Args:
        games: DataFrame with game data
        
    Returns:
        DataFrame with additional key statistical differences
    """
    print("增强关键统计差异特征，添加罚球率等冠军指标...")
    
    # Make a copy to avoid modifying the original DataFrame
    result = games.copy()
    
    # Columns to check for detailed stats
    required_shooting_cols = ['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 
                             'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA']
    
    # Check if detailed statistics are available
    has_shooting_stats = all(col in games.columns for col in required_shooting_cols)
    
    if not has_shooting_stats:
        print("  No detailed shooting statistics found. Setting default values.")
        # Set default values for the new indicators
        result['Team1_FTRate'] = 0.25  # Typical FT rate default
        result['Team2_FTRate'] = 0.25
        result['FTRateDiff'] = 0.0
        return result
    
    # Initialize team shooting stats dictionary if we need to build it
    team_shooting_stats = {}
    
    # Process historical game data to calculate team FT rate stats
    for _, row in games.iterrows():
        season = row['Season']
        
        # Get teams
        wteam_id = row['WTeamID']
        lteam_id = row['LTeamID']
        
        # Process winning team stats
        if f"{season}_{wteam_id}" not in team_shooting_stats:
            team_shooting_stats[f"{season}_{wteam_id}"] = {
                'fta': 0, 'fga': 0, 'games': 0
            }
        
        team_shooting_stats[f"{season}_{wteam_id}"]['fta'] += row['WFTA']
        team_shooting_stats[f"{season}_{wteam_id}"]['fga'] += row['WFGA']
        team_shooting_stats[f"{season}_{wteam_id}"]['games'] += 1
        
        # Process losing team stats
        if f"{season}_{lteam_id}" not in team_shooting_stats:
            team_shooting_stats[f"{season}_{lteam_id}"] = {
                'fta': 0, 'fga': 0, 'games': 0
            }
        
        team_shooting_stats[f"{season}_{lteam_id}"]['fta'] += row['LFTA']
        team_shooting_stats[f"{season}_{lteam_id}"]['fga'] += row['LFGA']
        team_shooting_stats[f"{season}_{lteam_id}"]['games'] += 1
    
    # Calculate free throw rate for each team (FTA/FGA)
    for team_key in team_shooting_stats:
        stats = team_shooting_stats[team_key]
        stats['ft_rate'] = stats['fta'] / stats['fga'] if stats['fga'] > 0 else 0.25
    
    # Add free throw rate features
    result['Team1_FTRate'] = result['IDTeam1'].map(lambda x: team_shooting_stats.get(x, {}).get('ft_rate', 0.25))
    result['Team2_FTRate'] = result['IDTeam2'].map(lambda x: team_shooting_stats.get(x, {}).get('ft_rate', 0.25))
    
    # Calculate free throw rate differential
    result['FTRateDiff'] = result['Team1_FTRate'] - result['Team2_FTRate']
    
    # Ensure we have all the critical champion indicators
    critical_indicators = [
        'EFGPctDiff',      # Effective FG% difference
        'TurnoverRateDiff', # Turnover rate difference
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
    result['ChampionComposite'] = (
        0.4 * result['EFGPctDiff'] +      # 40% weight to shooting
        0.25 * -result['TurnoverRateDiff'] +  # 25% weight to not turning it over (negative because lower is better)
        0.20 * result['OffRebRateDiff'] +  # 20% weight to offensive rebounding
        0.15 * result['FTRateDiff']        # 15% weight to getting to the line
    )
    
    print("关键冠军指标增强完成")
    return result