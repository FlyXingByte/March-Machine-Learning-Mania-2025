import os
import pandas as pd
import numpy as np

def load_data(data_path, start_year=2018):
    """
    Load men's and women's regular season and tournament detailed data

    Args:
        data_path: Path to the data directory
        start_year: Only include seasons from this year onwards

    Returns:
        Tuple of DataFrames (season_detail, tourney_detail, seeds, teams, submission)
    """
    print(f"Loading data files from {start_year} and onwards")
    # Load men's and women's regular season and tournament detailed data
    season_detail = pd.concat([
        pd.read_csv(os.path.join(data_path, "MRegularSeasonDetailedResults.csv")),
        pd.read_csv(os.path.join(data_path, "WRegularSeasonDetailedResults.csv"))
    ], ignore_index=True)
    tourney_detail = pd.concat([
        pd.read_csv(os.path.join(data_path, "MNCAATourneyDetailedResults.csv")),
        pd.read_csv(os.path.join(data_path, "WNCAATourneyDetailedResults.csv"))
    ], ignore_index=True)
    
    # Filter data for the seasons after start_year
    original_season_rows = len(season_detail)
    original_tourney_rows = len(tourney_detail)
    
    season_detail = season_detail[season_detail['Season'] >= start_year]
    tourney_detail = tourney_detail[tourney_detail['Season'] >= start_year]
    
    print(f"Regular season data: Reduced from {original_season_rows} rows to {len(season_detail)} rows")
    print(f"Tournament data: Reduced from {original_tourney_rows} rows to {len(tourney_detail)} rows")
    
    seeds = pd.concat([
        pd.read_csv(os.path.join(data_path, "MNCAATourneySeeds.csv")),
        pd.read_csv(os.path.join(data_path, "WNCAATourneySeeds.csv"))
    ], ignore_index=True)
    
    # Filter seed data
    original_seeds_rows = len(seeds)
    seeds = seeds[seeds['Season'] >= start_year]
    print(f"Seed data: Reduced from {original_seeds_rows} rows to {len(seeds)} rows")
    
    # Load team information
    teams = pd.concat([
        pd.read_csv(os.path.join(data_path, "MTeams.csv")),
        pd.read_csv(os.path.join(data_path, "WTeams.csv"))
    ], ignore_index=True)
    
    # Load submission sample file
    try:
        submission = pd.read_csv(os.path.join(data_path, "submission_64.csv"))
        print(f"Submission 64 format loaded with {len(submission)} rows")
    except:
        try:
            submission = pd.read_csv(os.path.join(data_path, "SampleSubmissionStage2.csv"))
            print(f"Sample submission loaded with {len(submission)} rows")
        except:
            submission = pd.DataFrame(columns=['ID', 'Pred'])
            print("No sample submission file found, please provide one")
            
    return season_detail, tourney_detail, seeds, teams, submission

def prepare_seed_dict(seeds_df):
    """
    Create a dictionary where the key is "Season_TeamID" and value is the seed number

    Args:
        seeds_df: DataFrame with seed information

    Returns:
        Dictionary mapping Season_TeamID to seed value
    """
    seed_dict = {
        '_'.join([str(int(row['Season'])), str(int(row['TeamID']))]): int(row['Seed'][1:3])
        for _, row in seeds_df.iterrows()
    }
    return seed_dict

def extract_game_info(id_str):
    """
    Extract year and team IDs from the game ID string

    Args:
        id_str: Game ID string in format "Year_Team1_Team2"

    Returns:
        Tuple of (year, team1, team2)
    """
    parts = id_str.split('_')
    year = int(parts[0])
    team1 = int(parts[1])
    team2 = int(parts[2])
    return year, team1, team2

def merge_and_prepare_games(season_detail, tourney_detail):
    """
    Merge regular season and tournament data into a single DataFrame

    Args:
        season_detail: Regular season game data
        tourney_detail: Tournament game data

    Returns:
        Combined DataFrame with game data
    """
    # Mark data source
    season_detail['ST'] = 'S'
    tourney_detail['ST'] = 'T'
    
    # Merge all game data (regular season and tournament)
    games = pd.concat([season_detail, tourney_detail], ignore_index=True)
    print(f"Merged game data total rows: {len(games)}")
    
    # If game location info is available, map it (e.g., 'A':1, 'H':2, 'N':3)
    if 'WLoc' in games.columns:
        games['WLoc'] = games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})
        
    return games

def load_kenpom_data(data_path, teams):
    """
    Load KenPom external data from 'kenpom-ncaa-2025.csv' and map TeamName to TeamID.

    Args:
        data_path: Path to the data directory.
        teams: DataFrame with teams information (should contain 'TeamID' and 'TeamName').

    Returns:
        kenpom_df: DataFrame with KenPom data and an additional 'TeamID' column.
    """
    kenpom_path = os.path.join(data_path, "kenpom-ncaa-2025.csv")
    
    try:
        kenpom_df = pd.read_csv(kenpom_path)
        print(f"Found KenPom data at {kenpom_path}")
        print(f"KenPom data columns: {kenpom_df.columns.tolist()}")
    except FileNotFoundError:
        print(f"Warning: KenPom data file not found at {kenpom_path}")
        # 返回空的DataFrame，带有基本列结构
        return pd.DataFrame(columns=['TeamName', 'TeamID', 'Season'])
    except Exception as e:
        print(f"Error loading KenPom data: {e}")
        # 返回空的DataFrame，带有基本列结构
        return pd.DataFrame(columns=['TeamName', 'TeamID', 'Season'])
    
    # 检查文件中是否存在 'TeamName' 列，如果没有，则尝试 'School'
    if 'TeamName' not in kenpom_df.columns:
        if 'School' in kenpom_df.columns:
            kenpom_df.rename(columns={'School': 'TeamName'}, inplace=True)
        elif 'Team' in kenpom_df.columns:
            kenpom_df.rename(columns={'Team': 'TeamName'}, inplace=True)
        else:
            print("Warning: Cannot find 'TeamName', 'School', or 'Team' column in KenPom data.")
            possible_team_columns = [col for col in kenpom_df.columns if 'team' in col.lower() or 'school' in col.lower()]
            if possible_team_columns:
                print(f"Using '{possible_team_columns[0]}' as team name column.")
                kenpom_df.rename(columns={possible_team_columns[0]: 'TeamName'}, inplace=True)
            else:
                print("No suitable team name column found. Returning empty DataFrame.")
                return pd.DataFrame(columns=['TeamName', 'TeamID', 'Season'])
    
    # 标准化球队名称以便匹配
    teams['TeamName_clean'] = teams['TeamName'].str.lower().str.strip()
    mapping = teams.set_index('TeamName_clean')['TeamID'].to_dict()
    
    kenpom_df['TeamName_clean'] = kenpom_df['TeamName'].str.lower().str.strip()
    kenpom_df['TeamID'] = kenpom_df['TeamName_clean'].map(mapping)
    
    # 检查映射的有效性
    unmapped_teams = kenpom_df[kenpom_df['TeamID'].isna()]['TeamName'].unique()
    if len(unmapped_teams) > 0:
        print(f"Warning: {len(unmapped_teams)} teams from KenPom data could not be mapped to TeamIDs:")
        for i, team in enumerate(unmapped_teams[:10]):  # 只显示前10个
            print(f"  {i+1}. {team}")
        if len(unmapped_teams) > 10:
            print(f"  ... and {len(unmapped_teams) - 10} more")
    
    # 删除未映射的记录
    original_len = len(kenpom_df)
    kenpom_df = kenpom_df.dropna(subset=['TeamID'])
    if original_len > len(kenpom_df):
        print(f"Dropped {original_len - len(kenpom_df)} rows with unmapped TeamIDs")
    
    if len(kenpom_df) > 0:
        kenpom_df['TeamID'] = kenpom_df['TeamID'].astype(int)
        print(f"Loaded KenPom data: {len(kenpom_df)} valid records")
    else:
        print("Warning: No valid KenPom data records after mapping. Returning empty DataFrame.")
    
    return kenpom_df