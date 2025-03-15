import numpy as np
import pandas as pd

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

def add_team_features(games):
    """
    Add historical performance features for each team based on entire season stats

    Args:
        games: DataFrame with game data

    Returns:
        Tuple of (DataFrame with added features, team_stats dictionary)
    """
    print("Calculating team historical statistics...")
    
    team_stats = {}
    
    for season in games['Season'].unique():
        season_games = games[games['Season'] == season]
        for team_id in set(season_games['Team1'].unique()) | set(season_games['Team2'].unique()):
            team_wins = len(season_games[season_games['WTeamID'] == team_id])
            team_losses = len(season_games[season_games['LTeamID'] == team_id])
            
            # Basic statistics
            win_rate = team_wins / (team_wins + team_losses) if (team_wins + team_losses) > 0 else 0.5
            games_played = team_wins + team_losses
            
            # Offensive and defensive efficiency calculations
            team_w_games = season_games[season_games['WTeamID'] == team_id]
            team_l_games = season_games[season_games['LTeamID'] == team_id]
            
            points_scored = team_w_games['WScore'].sum() + team_l_games['LScore'].sum()
            points_allowed = team_w_games['LScore'].sum() + team_l_games['WScore'].sum()
            
            avg_score = points_scored / games_played if games_played > 0 else 0
            avg_allowed = points_allowed / games_played if games_played > 0 else 0
            
            key = f"{season}_{team_id}"
            team_stats[key] = {
                'win_rate': win_rate,
                'games_played': games_played,
                'avg_score': avg_score,
                'avg_allowed': avg_allowed,
                'point_diff': avg_score - avg_allowed
            }
    
    # Map computed statistics to games using IDTeam1 and IDTeam2 fields
    games['Team1_WinRate'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('win_rate', 0.5))
    games['Team2_WinRate'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('win_rate', 0.5))
    games['Team1_GamesPlayed'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('games_played', 0))
    games['Team2_GamesPlayed'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('games_played', 0))
    games['Team1_AvgScore'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('avg_score', 60))
    games['Team2_AvgScore'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('avg_score', 60))
    games['Team1_AvgAllowed'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('avg_allowed', 60))
    games['Team2_AvgAllowed'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('avg_allowed', 60))
    games['Team1_PointDiff'] = games['IDTeam1'].map(lambda x: team_stats.get(x, {}).get('point_diff', 0))
    games['Team2_PointDiff'] = games['IDTeam2'].map(lambda x: team_stats.get(x, {}).get('point_diff', 0))
    
    # Add difference features
    games['WinRateDiff'] = games['Team1_WinRate'] - games['Team2_WinRate']
    games['GamesPlayedDiff'] = games['Team1_GamesPlayed'] - games['Team2_GamesPlayed']
    games['AvgScoreDiff'] = games['Team1_AvgScore'] - games['Team2_AvgScore']
    games['AvgAllowedDiff'] = games['Team1_AvgAllowed'] - games['Team2_AvgAllowed']
    games['PointDiffDiff'] = games['Team1_PointDiff'] - games['Team2_PointDiff']
    
    return games, team_stats

def add_head_to_head_features(games):
    """
    For each game (training only), compute head-to-head win rate for Team1 vs Team2 using previous seasons data.
    If no historical matchup, assign default value 0.5.

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
    For each game, calculate recent performance features for each team in the same season.
    Uses 'DayNum' to determine game order. If 'DayNum' is missing, skip recent performance features.
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
    Aggregate game statistics for later use if available

    Args:
        games: DataFrame with game data

    Returns:
        DataFrame with aggregated features
    """
    available_cols = [col for col in ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR',
                                     'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA',
                                     'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO',
                                     'LStl', 'LBlk', 'LPF'] if col in games.columns]
    
    if not available_cols:
        return pd.DataFrame(columns=['IDTeams_c_score'])
    
    agg_funcs = ['mean', 'median', 'std']
    gb = games.groupby('IDTeams').agg({col: agg_funcs for col in available_cols}).reset_index()
    gb.columns = [''.join(c) + '_c_score' if isinstance(c, tuple) else c for c in gb.columns]
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
    
    # 固定将提交数据的 GameType 设为 "Tournament"
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
    # 首先检查必要的列是否存在
    if kenpom_df.empty:
        print("Warning: KenPom DataFrame is empty. No features will be added.")
        return games
    
    # 检查KenPom数据中的列
    print(f"Available columns in KenPom data: {kenpom_df.columns.tolist()}")
    
    # 检查是否有Season列，如果没有，尝试添加一个默认值
    if 'Season' not in kenpom_df.columns:
        # 如果有Year列，重命名为Season
        if 'Year' in kenpom_df.columns:
            kenpom_df.rename(columns={'Year': 'Season'}, inplace=True)
        else:
            # 如果没有季节信息，使用最近的赛季（2025）作为默认值
            print("Warning: No 'Season' column found in KenPom data. Using 2025 as default.")
            kenpom_df['Season'] = 2025
    
    # 确保Season列是整数类型
    kenpom_df['Season'] = kenpom_df['Season'].astype(int)
    
    # 检查是否有主要效率指标
    efficiency_column = None
    if 'AdjEM' in kenpom_df.columns:
        efficiency_column = 'AdjEM'
    elif 'AdjEff' in kenpom_df.columns:
        efficiency_column = 'AdjEff'
    elif 'NetRtg' in kenpom_df.columns:
        efficiency_column = 'NetRtg'
    else:
        # 尝试找到可能的效率列
        possible_columns = [col for col in kenpom_df.columns if 'Eff' in col or 'Rtg' in col or 'EM' in col]
        if possible_columns:
            efficiency_column = possible_columns[0]
            print(f"Using '{efficiency_column}' as efficiency metric.")
        else:
            print("Warning: No efficiency metric found in KenPom data. Returning games without KenPom features.")
            return games
    
    # 确保效率指标是数值类型
    try:
        kenpom_df[efficiency_column] = pd.to_numeric(kenpom_df[efficiency_column], errors='coerce')
        # 检查是否有NaN值并处理
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
    
    # 确保合并后的列是数值类型
    games['Team1_AdjEM'] = pd.to_numeric(games['Team1_AdjEM'], errors='coerce').fillna(0)
    games['Team2_AdjEM'] = pd.to_numeric(games['Team2_AdjEM'], errors='coerce').fillna(0)
    
    # Calculate difference feature
    games['AdjEM_Diff'] = games['Team1_AdjEM'] - games['Team2_AdjEM']
    print("KenPom features merged.")
    
    return games
