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