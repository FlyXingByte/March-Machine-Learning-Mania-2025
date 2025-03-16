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
