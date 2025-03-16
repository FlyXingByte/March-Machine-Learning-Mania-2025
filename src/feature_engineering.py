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
    games['Team2_WinRate'] = 0.5
    games['Team2_AvgScore'] = 60
    games['Team2_AvgAllowed'] = 60
    games['Team2_PointDiff'] = 0
    games['Team2_GamesPlayed'] = 0
    
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
                    'wins': 0, 'losses': 0, 'points_for': 0, 'points_against': 0
                }
            
            team_stats[key]['wins'] += 1
            team_stats[key]['points_for'] += row['WScore'] if 'WScore' in row else 0
            team_stats[key]['points_against'] += row['LScore'] if 'LScore' in row else 0
            
            # Process losing team statistics
            opp_key = f"{season}_{opponent_id}"
            if opp_key not in team_stats:
                team_stats[opp_key] = {
                    'wins': 0, 'losses': 0, 'points_for': 0, 'points_against': 0
                }
            
            team_stats[opp_key]['losses'] += 1
            team_stats[opp_key]['points_for'] += row['LScore'] if 'LScore' in row else 0
            team_stats[opp_key]['points_against'] += row['WScore'] if 'WScore' in row else 0
    
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
        else:
            stats['win_rate'] = 0.5
            stats['avg_score'] = 60
            stats['avg_allowed'] = 60
            stats['point_diff'] = 0
            stats['games_played'] = 0
    
    # Map the team statistics back to the original dataframe
    games['Team1_WinRate'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('win_rate', 0.5))
    games['Team1_AvgScore'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('avg_score', 60))
    games['Team1_AvgAllowed'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('avg_allowed', 60))
    games['Team1_PointDiff'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('point_diff', 0))
    games['Team1_GamesPlayed'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('games_played', 0))
    
    games['Team2_WinRate'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('win_rate', 0.5))
    games['Team2_AvgScore'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('avg_score', 60))
    games['Team2_AvgAllowed'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('avg_allowed', 60))
    games['Team2_PointDiff'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('point_diff', 0))
    games['Team2_GamesPlayed'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('games_played', 0))
    
    # Compute difference features
    games['WinRateDiff'] = games['Team1_WinRate'] - games['Team2_WinRate']
    games['AvgScoreDiff'] = games['Team1_AvgScore'] - games['Team2_AvgScore']
    games['AvgAllowedDiff'] = games['Team1_AvgAllowed'] - games['Team2_AvgAllowed']
    games['PointDiffDiff'] = games['Team1_PointDiff'] - games['Team2_PointDiff']
    games['GamesPlayedDiff'] = games['Team1_GamesPlayed'] - games['Team2_GamesPlayed']
    
    print(f"团队统计特征计算完成，共处理了 {len(team_stats)} 个团队-赛季组合")
    
    return games, team_stats

def add_head_to_head_features(games):
    """
    For each game (training only), compute head-to-head win rate for Team1 vs Team2 using previous seasons' data.
    If no historical matchup, assign a default value of 0.5.
    
    Args:
        games: DataFrame with game data
        
    Returns:
        DataFrame with head-to-head features added
    """
    print("Calculating historical head-to-head features...")
    
    def compute_h2h(row, df):
        current_season = row['Season']
        team1 = row['Team1']
        team2 = row['Team2']
        past_games = df[(df['Season'] < current_season) &
                        (((df['Team1'] == team1) & (df['Team2'] == team2)) |
                         ((df['Team1'] == team2) & (df['Team2'] == team1)))]
        if past_games.empty:
            return 0.5
        else:
            win_rate = past_games['WinA'].mean()
            return win_rate
            
    games['H2H_WinRate'] = games.apply(lambda r: compute_h2h(r, games), axis=1)
    return games

def add_recent_performance_features(games, window=5):
    """
    For each game, calculate recent performance features for each team within the same season.
    Features include recent win rate and average score differential.
    
    Args:
        games: DataFrame with game data
        window: Number of recent games to consider
        
    Returns:
        DataFrame with recent performance features added
    """
    if 'DayNum' not in games.columns:
        print("DayNum column not found; skipping recent performance features.")
        games['Team1_RecentWinRate'] = 0.5
        games['Team2_RecentWinRate'] = 0.5
        games['Team1_RecentScoreDiff'] = 0.0
        games['Team2_RecentScoreDiff'] = 0.0
        return games
    
    print(f"Calculating recent performance features (window={window} games)...")
    
    def recent_stats(row, team, df):
        season = row['Season']
        current_day = row['DayNum']
        past_games = df[(df['Season'] == season) & (df['DayNum'] < current_day)]
        team_games = past_games[(past_games['Team1'] == team) | (past_games['Team2'] == team)]
        if team_games.empty:
            return (0.5, 0.0)
        win_rate = team_games['WinA'].mean()
        avg_score_diff = team_games['ScoreDiffNorm'].mean() if 'ScoreDiffNorm' in team_games.columns else 0.0
        return (win_rate, avg_score_diff)
    
    recent_win_rate1 = []
    recent_win_rate2 = []
    recent_score_diff1 = []
    recent_score_diff2 = []
    
    for _, row in games.iterrows():
        win1, score_diff1 = recent_stats(row, row['Team1'], games)
        win2, score_diff2 = recent_stats(row, row['Team2'], games)
        recent_win_rate1.append(win1)
        recent_score_diff1.append(score_diff1)
        recent_win_rate2.append(win2)
        recent_score_diff2.append(score_diff2)
    
    games['Team1_RecentWinRate'] = recent_win_rate1
    games['Team2_RecentWinRate'] = recent_win_rate2
    games['Team1_RecentScoreDiff'] = recent_score_diff1
    games['Team2_RecentScoreDiff'] = recent_score_diff2
    
    games['RecentWinRateDiff'] = games['Team1_RecentWinRate'] - games['Team2_RecentWinRate']
    games['RecentScoreDiffDiff'] = games['Team1_RecentScoreDiff'] - games['Team2_RecentScoreDiff']
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
    print("Preparing submission file features...")
    
    game_info = submission_df['ID'].apply(extract_game_info_func)
    submission_df['Season'] = [info[0] for info in game_info]
    submission_df['Team1'] = [info[1] for info in game_info]
    submission_df['Team2'] = [info[2] for info in game_info]
    
    submission_df['IDTeams'] = submission_df.apply(lambda r: f"{r['Team1']}_{r['Team2']}", axis=1)
    submission_df['IDTeam1'] = submission_df.apply(lambda r: f"{r['Season']}_{r['Team1']}", axis=1)
    submission_df['IDTeam2'] = submission_df.apply(lambda r: f"{r['Season']}_{r['Team2']}", axis=1)
    
    submission_df['Team1Seed'] = submission_df['IDTeam1'].map(seed_dict).fillna(16)
    submission_df['Team2Seed'] = submission_df['IDTeam2'].map(seed_dict).fillna(16)
    submission_df['SeedDiff'] = submission_df['Team1Seed'] - submission_df['Team2Seed']
    
    submission_df['Team1SeedStrength'] = np.exp(-submission_df['Team1Seed'] / 4)
    submission_df['Team2SeedStrength'] = np.exp(-submission_df['Team2Seed'] / 4)
    submission_df['SeedStrengthDiff'] = submission_df['Team1SeedStrength'] - submission_df['Team2SeedStrength']
    
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
    
    submission_df['WinRateDiff'] = submission_df['Team1_WinRate'] - submission_df['Team2_WinRate']
    submission_df['GamesPlayedDiff'] = submission_df['Team1_GamesPlayed'] - submission_df['Team2_GamesPlayed']
    submission_df['AvgScoreDiff'] = submission_df['Team1_AvgScore'] - submission_df['Team2_AvgScore']
    submission_df['AvgAllowedDiff'] = submission_df['Team1_AvgAllowed'] - submission_df['Team2_AvgAllowed']
    submission_df['PointDiffDiff'] = submission_df['Team1_PointDiff'] - submission_df['Team2_PointDiff']
    
    # Force the GameType for submission to "Tournament"
    submission_df['GameType'] = 'Tournament'
    
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
