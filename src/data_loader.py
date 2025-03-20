import os
import pandas as pd
import numpy as np
import re

def load_data(data_path, start_year=2018, stage=2, test_mode=False):
    """
    Load all necessary data files
    
    Parameters:
        data_path: Path to the data directory
        start_year: Only include seasons starting from this year
        stage: Competition stage (1 or 2)
        test_mode: If True, only load 10 games per year for quick testing
    
    Returns:
        Tuple of DataFrames (season_detail, tourney_detail, seeds, teams, submission)
    """
    print(f"Loading data files from {start_year} and later")
    # Load men's and women's regular season and tournament detailed data
    season_detail = pd.concat([
        pd.read_csv(os.path.join(data_path, "MRegularSeasonDetailedResults.csv")),
        pd.read_csv(os.path.join(data_path, "WRegularSeasonDetailedResults.csv"))
    ], ignore_index=True)
    tourney_detail = pd.concat([
        pd.read_csv(os.path.join(data_path, "MNCAATourneyDetailedResults.csv")),
        pd.read_csv(os.path.join(data_path, "WNCAATourneyDetailedResults.csv"))
    ], ignore_index=True)
    
    # Filter seasons data after start_year
    original_season_rows = len(season_detail)
    original_tourney_rows = len(tourney_detail)
    
    season_detail = season_detail[season_detail['Season'] >= start_year]
    tourney_detail = tourney_detail[tourney_detail['Season'] >= start_year]
    
    print(f"Regular season data: reduced from {original_season_rows} rows to {len(season_detail)} rows")
    print(f"Tournament data: reduced from {original_tourney_rows} rows to {len(tourney_detail)} rows")
    
    seeds = pd.concat([
        pd.read_csv(os.path.join(data_path, "MNCAATourneySeeds.csv")),
        pd.read_csv(os.path.join(data_path, "WNCAATourneySeeds.csv"))
    ], ignore_index=True)
    
    # Filter seeds data
    original_seeds_rows = len(seeds)
    seeds = seeds[seeds['Season'] >= start_year]
    print(f"Seeds data: reduced from {original_seeds_rows} rows to {len(seeds)} rows")
    
    # Load team information (combine MTeams.csv and WTeams.csv)
    teams = load_teams(data_path)
    
    # If test mode, only keep a small number of games per year for quick testing
    if test_mode:
        # Group by season and sample 10 games for regular season
        season_detail = season_detail.groupby('Season').apply(
            lambda x: x.sample(min(10, len(x)), random_state=42)
        ).reset_index(drop=True)
        
        # Group by season and sample up to 10 available games for tournament
        tourney_detail = tourney_detail.groupby('Season').apply(
            lambda x: x.sample(min(10, len(x)), random_state=42)
        ).reset_index(drop=True)
        
        print(f"Test mode enabled: Regular season data reduced to {len(season_detail)} rows")
        print(f"Test mode enabled: Tournament data reduced to {len(tourney_detail)} rows")
    
    # Load appropriate submission file based on stage
    # Use correct file name without "M" prefix
    try:
        if stage == 1:
            submission_path = os.path.join(data_path, "SampleSubmissionStage1.csv")
            submission = pd.read_csv(submission_path)
        else:
            submission_path = os.path.join(data_path, "SampleSubmissionStage2.csv")
            submission = pd.read_csv(submission_path)
        print(f"Successfully loaded submission file: {submission_path}")
    except FileNotFoundError:
        print(f"Error: Submission file not found at {submission_path}")
        print("Creating an empty submission DataFrame as backup")
        # Create an empty submission DataFrame as backup
        submission = pd.DataFrame(columns=['ID', 'Pred'])
        
    return season_detail, tourney_detail, seeds, teams, submission

def load_teams(data_path):
    """
    Load and combine men's and women's team information.
    
    Parameters:
        data_path: Path to the data directory.
        
    Returns:
        teams: Combined DataFrame containing 'TeamID' and 'TeamName' columns
    """
    try:
        mteams = pd.read_csv(os.path.join(data_path, "MTeams.csv"))
        wteams = pd.read_csv(os.path.join(data_path, "WTeams.csv"))
        teams = pd.concat([mteams, wteams], ignore_index=True)
        print(f"Loaded {len(teams)} teams from MTeams and WTeams.")
    except Exception as e:
        print("Error loading team data:", e)
        teams = pd.DataFrame()
    return teams

def prepare_seed_dict(seeds_df):
    """
    Create a dictionary where the key is "Season_TeamID" and the value is seed number

    Parameters:
        seeds_df: DataFrame containing seed information

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
    Extract year and teamID from a gameID string

    Parameters:
        id_str: Format of a gameID string as "Year_Team1_Team2"

    Returns:
        Tuple (year, team1, team2)
    """
    parts = id_str.split('_')
    year = int(parts[0])
    team1 = int(parts[1])
    team2 = int(parts[2])
    return year, team1, team2

def merge_and_prepare_games(season_detail, tourney_detail):
    """
    Merge regular season and tournament data into a single DataFrame.
    Add a new column 'GameType' to indicate whether the data is from regular season or tournament.

    Parameters:
        season_detail: Regular season game data
        tourney_detail: Tournament game data

    Returns:
        Merged DataFrame containing game data including 'GameType' column.
    """
    # Create a copy to avoid modifying the original DataFrame
    season_detail = season_detail.copy()
    tourney_detail = tourney_detail.copy()
    
    # Add GameType column to each DataFrame
    season_detail['GameType'] = 'Regular'
    tourney_detail['GameType'] = 'Tournament'
    
    # Merge data
    games = pd.concat([season_detail, tourney_detail], ignore_index=True)
    
    return games

def standardize_team_name(name):
    """
    Standardize team names to improve matching between different datasets.
    
    Args:
        name: Original team name string.
        
    Returns:
        Standardized team name string.
    """
    if name is None:
        return ""
    
    # Convert to string if not already
    name = str(name).strip()
    
    # Remove asterisks, seeds, and trailing numbers used in tournament contexts
    name = re.sub(r'\s*\d+\*?$', '', name)  # Remove trailing numbers and asterisks (e.g., "Duke 2*")
    name = re.sub(r'\s*\([0-9]+\)$', '', name)  # Remove seeds like (12)
    
    # General state abbreviation standardization
    state_mapping = {
        'St.': 'State', 'St': 'State', 
        'N.': 'North', 'N': 'North', 'No.': 'North', 'No': 'North',
        'S.': 'South', 'S': 'South', 'So.': 'South', 'So': 'South',
        'E.': 'East', 'E': 'East',
        'W.': 'West', 'W': 'West',
        'C.': 'Central', 'C': 'Central', 'Cent.': 'Central', 'Cent': 'Central',
    }
    
    for abbr, full in state_mapping.items():
        # Replace abbreviations with spaces on both sides
        name = re.sub(fr'\s{abbr}\s', f' {full} ', name)
        # Replace abbreviations at the start of a word
        name = re.sub(fr'\b{abbr}\s', f'{full} ', name)
        # Replace abbreviations at the end of a word
        name = re.sub(fr'\s{abbr}\b', f' {full}', name)
    
    # Common word standardization
    name = re.sub(r'\bUniv\b', 'University', name)
    name = re.sub(r'\bColl\b', 'College', name)
    name = re.sub(r'\bU$', 'University', name)
    
    # Common team name-specific corrections
    corrections = {
        'NC State': 'North Carolina State',
        'UNC': 'North Carolina',
        'UCF': 'Central Florida',
        'USC': 'Southern California',
        'UTSA': 'Texas San Antonio',
        'UTEP': 'Texas El Paso',
        'UConn': 'Connecticut',
        'Pitt': 'Pittsburgh',
        'LSU': 'Louisiana State',
        'VCU': 'Virginia Commonwealth',
        'BYU': 'Brigham Young',
        'SMU': 'Southern Methodist',
        'UNLV': 'Nevada Las Vegas',
        'Penn': 'Pennsylvania',
        'Ole Miss': 'Mississippi',
        'UMass': 'Massachusetts',
        'Cal': 'California',
        'FGCU': 'Florida Gulf Coast',
        'UAB': 'Alabama Birmingham',
        'FIU': 'Florida International',
        'GA Tech': 'Georgia Tech',
        'UIC': 'Illinois Chicago',
        'LIU': 'Long Island',
        'Middle Tenn': 'Middle Tennessee State',
        'Middle Tennessee': 'Middle Tennessee State',
        'Mount St Mary\'s': 'Mount St Marys',
        'SUNY Albany': 'Albany',
        'St. Joseph\'s': 'Saint Josephs',
        'St Joseph\'s': 'Saint Josephs',
        'Saint Joseph\'s': 'Saint Josephs',
        'St. Mary\'s': 'Saint Marys',
        'St Mary\'s': 'Saint Marys',
        'Saint Mary\'s': 'Saint Marys',
        'Saint Joseph\'s PA': 'Saint Josephs',
        'Long Beach St.': 'Long Beach State',
        'Long Beach St': 'Long Beach State',
        'Cal State Bakersfield': 'CSU Bakersfield',
        'Cal State Fullerton': 'CSU Fullerton',
        'Cal State Northridge': 'CSU Northridge',
        'Miami FL': 'Miami Florida',
        'Miami OH': 'Miami Ohio',
        'Miami (FL)': 'Miami Florida',
        'Miami (OH)': 'Miami Ohio',
        'Loyola Chicago': 'Loyola IL',
        'Loyola Marymount': 'Loyola CA',
        'Loyola MD': 'Loyola Maryland',
        'UL Lafayette': 'Louisiana Lafayette',
        'Louisiana Lafayette': 'Louisiana',
        'UL Monroe': 'Louisiana Monroe',
        'Louisiana Monroe': 'UL Monroe',
        'Texas A&M Corpus Chris': 'Texas A&M Corpus Christi',
        'TX A&M Corpus Chris': 'Texas A&M Corpus Christi',
        'Grambling': 'Grambling State',
        'Prairie View': 'Prairie View A&M',
        'PVAMU': 'Prairie View A&M',
        'Mississippi Valley': 'Mississippi Valley State',
        'MVSU': 'Mississippi Valley State',
        'Missouri KC': 'Missouri Kansas City',
        'UMKC': 'Missouri Kansas City',
        'Kansas City': 'Missouri Kansas City',
        'SE Louisiana': 'Southeastern Louisiana',
        'SE Missouri St': 'Southeast Missouri State',
        'Southeast Missouri': 'Southeast Missouri State',
        'Southern Illinois Edwardsville': 'SIU Edwardsville',
        'Southern Illinois': 'SIU',
        'The Citadel': 'Citadel',
        'Bethune Cookman': 'Bethune-Cookman',
        'Central Connecticut': 'Central Connecticut State',
        'CCSU': 'Central Connecticut State',
        'Detroit': 'Detroit Mercy',
        'Florida A&M': 'Florida AM',
        'FAMU': 'Florida AM',
        'Maryland Eastern Shore': 'MD Eastern Shore',
        'UMES': 'MD Eastern Shore',
        'Mississippi State': 'Mississippi St',
        'Missouri State': 'Missouri St',
        'Montana State': 'Montana St',
        'Morehead State': 'Morehead St',
        'Murray State': 'Murray St',
        'Nicholls': 'Nicholls St',
        'Nicholls State': 'Nicholls St',
        'Portland State': 'Portland St',
        'Sacramento State': 'Sacramento St',
        'San Diego State': 'San Diego St',
        'South Dakota State': 'South Dakota St',
        'Tennessee State': 'Tennessee St',
        'Texas State': 'Texas St',
        'Utah State': 'Utah St',
        'Weber State': 'Weber St',
        'Wichita State': 'Wichita St',
        'Wright State': 'Wright St',
        'Youngstown State': 'Youngstown St',
        'Arkansas Pine Bluff': 'Ark Pine Bluff',
        'Arkansas Little Rock': 'Ark Little Rock',
        'Central Arkansas': 'Cent Arkansas',
        'Georgia Southern': 'GA Southern',
        'Wisconsin Green Bay': 'WI Green Bay',
        'Wisconsin Milwaukee': 'WI Milwaukee',
        'UW Milwaukee': 'WI Milwaukee',
        'Massachusetts Lowell': 'MA Lowell',
        'Nebraska Omaha': 'NE Omaha',
        'New Jersey Tech': 'NJIT',
        'Albany': 'SUNY Albany',
        'Abilene Christian': 'Abilene Chr',
        'American': 'American Univ',
        'Boston University': 'Boston Univ',
        'Charleston': 'Col Charleston',
        'College of Charleston': 'Col Charleston',
        'Houston Baptist': 'Houston Chr',
        'Houston Christian': 'Houston Chr',
        'Green Bay': 'WI Green Bay',
        'Milwaukee': 'WI Milwaukee',
        'Purdue Fort Wayne': 'PFW',
        'Queens': 'Queens NC',
        'Saint Francis PA': 'St Francis PA',
        'Saint Francis NY': 'St Francis NY',
        'Saint Francis Brooklyn': 'St Francis NY',
        'Sam Houston': 'Sam Houston St',
        'SFA': 'Stephen F Austin',
        'SIUE': 'SIU Edwardsville',
        'Tarleton': 'Tarleton St',
        'Tarleton State': 'Tarleton St',
        'UC Davis': 'California Davis',
        'UC Irvine': 'California Irvine',
        'UC Riverside': 'California Riverside',
        'UC San Diego': 'California San Diego',
        'UC Santa Barbara': 'California Santa Barbara',
        'UCSB': 'California Santa Barbara',
        'UNC Asheville': 'North Carolina Asheville',
        'UNC Greensboro': 'North Carolina Greensboro',
        'UNC Wilmington': 'North Carolina Wilmington',
        'UNCG': 'North Carolina Greensboro',
        'UNCW': 'North Carolina Wilmington',
        'UT Arlington': 'Texas Arlington',
        'UT Rio Grande Valley': 'Texas Rio Grande Valley',
        'UTRGV': 'Texas Rio Grande Valley',
        'Western Illinois': 'W Illinois',
        'Western Kentucky': 'W Kentucky',
        'Western Michigan': 'W Michigan',
        'Western Carolina': 'W Carolina',
        'McNeese St.': 'McNeese St',
        'McNeese': 'McNeese St',
        'Coastal Carolina': 'Coastal Car',
        'Washington St.': 'Washington State',
        'Wagner 16*': 'Wagner',
        'Boise St. 10*': 'Boise State',
        'Washington St. 7*': 'Washington State',
        'Long Beach St. 15*': 'Long Beach State',
        'Western Kentucky 15*': 'W Kentucky',
        'Charleston 13*': 'Col Charleston',
        'St. Francis PA': 'St Francis PA',
        'Tennessee Martin': 'TN Martin',
        'Monmouth': 'Monmouth NJ',
        'C Arkansas': 'Cent Arkansas',
    }
    
    # Apply multiple iterations of fixes to handle concatenated corrections
    for _ in range(2):  # Apply fixes twice to handle nested corrections
        # After applying corrections, check if the name is in corrections again
        if name in corrections:
            name = corrections[name]
    
    return name.strip()

def load_kenpom_data(data_path, teams):
    """
    Load KenPom external data from 'kenpom-ncaa-2025.csv' and map TeamName to TeamID.
    Uses standardize_team_name for team name standardization to match with teams data.

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
        return pd.DataFrame(columns=['TeamName', 'TeamID', 'Season'])
    except Exception as e:
        print(f"Error loading KenPom data: {e}")
        return pd.DataFrame(columns=['TeamName', 'TeamID', 'Season'])
    
    # Check if 'TeamName' column exists, if not, try 'School' or 'Team'
    if 'TeamName' not in kenpom_df.columns:
        if 'School' in kenpom_df.columns:
            kenpom_df.rename(columns={'School': 'TeamName'}, inplace=True)
        elif 'Team' in kenpom_df.columns:
            kenpom_df.rename(columns={'Team': 'TeamName'}, inplace=True)
        else:
            raise KeyError("Cannot find 'TeamName', 'School', or 'Team' column in KenPom data, please check CSV file format.")
    
    # Display first few unprocessed team names for debugging
    print("Original KenPom team names (sample):", kenpom_df['TeamName'].head().tolist())
    
    # Standardize team names in KenPom data
    kenpom_df['TeamName_std'] = kenpom_df['TeamName'].apply(standardize_team_name)
    
    # Standardize team names in teams data
    teams['TeamName_std'] = teams['TeamName'].apply(standardize_team_name)
    
    # Create dictionary for debugging
    debug_mapping = {name: std for name, std in zip(teams['TeamName'], teams['TeamName_std'])}
    print("Sample of team name standardization:")
    for i, (orig, std) in enumerate(list(debug_mapping.items())[:5]):
        print(f"  {i+1}. '{orig}' -> '{std}'")
    
    # Create mapping dictionary: standardized name -> TeamID
    mapping = teams.set_index('TeamName_std')['TeamID'].to_dict()
    
    # Lowercase and map standardized names to TeamID
    kenpom_df['TeamName_std_lower'] = kenpom_df['TeamName_std'].str.lower().str.strip()
    teams['TeamName_std_lower'] = teams['TeamName_std'].str.lower().str.strip()
    
    # Rebuild mapping dictionary, using lowercase string
    mapping_lower = teams.set_index('TeamName_std_lower')['TeamID'].to_dict()
    
    # Apply mapping
    kenpom_df['TeamID'] = kenpom_df['TeamName_std_lower'].map(mapping_lower)
    
    # Debug matching issues
    print("\nSample of standardized KenPom team names:")
    for i, row in kenpom_df.head().iterrows():
        mapped_id = row['TeamID']
        status = "✓" if not pd.isna(mapped_id) else "✗"
        print(f"  {status} '{row['TeamName']}' -> '{row['TeamName_std']}' -> ID: {mapped_id}")
    
    # Check mapping validity
    unmapped_teams = kenpom_df[kenpom_df['TeamID'].isna()].copy()
    if len(unmapped_teams) > 0:
        unmapped_count = len(unmapped_teams)
        print(f"Warning: {unmapped_count} teams from KenPom data could not be mapped to TeamIDs initially.")
        
        # Create backup name mapping dictionary (official name->ID)
        team_names_dict = dict(zip(teams['TeamName_std_lower'], teams['TeamID']))
        
        # Try to find unmatched teams through partial matching
        for i, row in unmapped_teams.iterrows():
            orig_name = row['TeamName']
            std_name = row['TeamName_std']
            
            # Try 1: Remove "Univ", "University", "College" etc. words before matching
            simple_name = std_name
            simple_name = re.sub(r'\sUniv(ersity)?(\sof)?', '', simple_name, flags=re.IGNORECASE)
            simple_name = re.sub(r'\sCollege', '', simple_name, flags=re.IGNORECASE)
            simple_name = re.sub(r'\sUniversidad', '', simple_name, flags=re.IGNORECASE)
            simple_name = simple_name.lower().strip()
            
            # Try to match simplified name
            for official_name, team_id in team_names_dict.items():
                official_simple = re.sub(r'\sUniv(ersity)?(\sof)?', '', official_name, flags=re.IGNORECASE)
                official_simple = re.sub(r'\sCollege', '', official_simple, flags=re.IGNORECASE)
                official_simple = re.sub(r'\sUniversidad', '', official_simple, flags=re.IGNORECASE)
                official_simple = official_simple.lower().strip()
                
                # If simplified name matches or one name is a substring of the other, consider it a match
                if simple_name == official_simple or \
                   (len(simple_name) > 5 and simple_name in official_simple) or \
                   (len(official_simple) > 5 and official_simple in simple_name):
                    kenpom_df.loc[i, 'TeamID'] = team_id
                    print(f"  ✓ Found match through simplification: '{orig_name}' -> '{std_name}' -> '{official_name}' -> ID: {team_id}")
                    break
            
            # If still no match found, try manual mapping some common names
            if pd.isna(kenpom_df.loc[i, 'TeamID']):
                manual_matches = {
                    'st marys': 3181,  # For example, if St. Mary's ID is 3181
                    'saint marys': 3181,
                    'st josephs': 3396,
                    'saint josephs': 3396,
                    'st louis': 3391,
                    'saint louis': 3391,
                    'st bonaventure': 3366,
                    'saint bonaventure': 3366,
                    'middle tennessee': 3244,
                    'middle tenn': 3244,
                    'george washington': 3206,
                    'g washington': 3206,
                    'florida atlantic': 3196,
                    'florida atl': 3196,
                    'mcneese': 3243,
                    'mcneese state': 3243,
                    'csun': 3316,
                    'cal st northridge': 3316,
                    
                    # Additional mappings for remaining unmapped teams
                    'kent st': 3245,
                    'kent state': 3245,
                    'st thomas': 3901,  # Assuming this is the ID for St Thomas MN
                    'st thomas mn': 3901,
                    'south carolina st': 3382,
                    'sc state': 3382,
                    'umass lowell': 3415,
                    'massachusetts lowell': 3415,
                    'mount st marys': 3291,
                    'mt st marys': 3291,
                    'bethune cookman': 3127,
                    'bethune-cookman': 3127,
                    'liu': 3415,  # Assuming this is Long Island University's ID
                    'long island university': 3415,
                    'iu indy': 3236,  # Assuming this is IUPUI's ID
                    'iupui': 3236,
                    'fairleigh dickinson': 3191,
                    'f dickinson': 3191,
                    'north carolina central': 3312,
                    'nc central': 3312,
                    'siu edwardsville': 3429,
                    's illinois edwardsville': 3429,
                    'ut arlington': 3399,
                    'texas arlington': 3399,
                    'north florida': 3317,
                    'n florida': 3317,
                    
                    # Final 3 remaining unmapped teams
                    'north carolina a&t': 3113,
                    'nc a&t': 3113,
                    'sacramento st': 3361,
                    'sacramento state': 3361,
                    'cal st fullerton': 3192,
                    'n florida': 3317
                }
                
                # Check if manual mapping exists
                simple_key = std_name.lower().strip()
                if simple_key in manual_matches:
                    kenpom_df.loc[i, 'TeamID'] = manual_matches[simple_key]
                    print(f"  ✓ Found match through manual mapping: '{orig_name}' -> '{std_name}' -> ID: {manual_matches[simple_key]}")
        
        # Re-check unmapped teams
        still_unmapped = kenpom_df[kenpom_df['TeamID'].isna()]['TeamName'].unique()
        if len(still_unmapped) > 0:
            print(f"Still have {len(still_unmapped)} unmapped teams after additional matching attempts:")
            for i, team in enumerate(still_unmapped[:10]):
                std_name = standardize_team_name(team)
                print(f"  {i+1}. {team} -> {std_name}")
            if len(still_unmapped) > 10:
                print(f"  ... and {len(still_unmapped) - 10} more")
    
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

def load_merged_kenpom_data(data_path, teams):
    """
    Load merged KenPom external data from 'merged_kenpom.csv' and map Team_Name to TeamID.
    Uses enhanced multi-level matching strategy to maximize team name matching.

    Args:
        data_path: Path to the data directory.
        teams: DataFrame with teams information (should contain 'TeamID' and 'TeamName').

    Returns:
        merged_kenpom_df: DataFrame with merged KenPom data and an additional 'TeamID' column.
    """
    kenpom_path = os.path.join(data_path, "merged_kenpom.csv")
    try:
        kenpom_df = pd.read_csv(kenpom_path)
        print(f"Found merged KenPom data at {kenpom_path}")
        print(f"Merged KenPom data columns: {kenpom_df.columns.tolist()}")
    except FileNotFoundError:
        # Try alternative path
        alt_path = os.path.join(os.path.dirname(data_path), "merged_kenpom.csv")
        try:
            kenpom_df = pd.read_csv(alt_path)
            print(f"Found merged KenPom data at alternative path: {alt_path}")
            print(f"Merged KenPom data columns: {kenpom_df.columns.tolist()}")
        except FileNotFoundError:
            print(f"Warning: Merged KenPom data file not found at {kenpom_path} or {alt_path}")
            return pd.DataFrame(columns=['Team_Name', 'TeamID', 'Year'])
        except Exception as e:
            print(f"Error loading merged KenPom data from alternative path: {e}")
            return pd.DataFrame(columns=['Team_Name', 'TeamID', 'Year'])
    except Exception as e:
        print(f"Error loading merged KenPom data: {e}")
        return pd.DataFrame(columns=['Team_Name', 'TeamID', 'Year'])
    
    # Record original data statistics
    print(f"Loaded merged KenPom data with {len(kenpom_df)} rows and {len(kenpom_df['Team_Name'].unique())} unique teams.")
    print(f"Data spans seasons: {sorted(kenpom_df['Year'].unique())}")
    
    # Drop the columns we don't want to use
    if 'Team' in kenpom_df.columns:
        kenpom_df = kenpom_df.drop(columns=['Team'])
    if 'Rk' in kenpom_df.columns:
        kenpom_df = kenpom_df.drop(columns=['Rk'])
    if 'W-L' in kenpom_df.columns:
        kenpom_df = kenpom_df.drop(columns=['W-L'])
    
    # Rename Year to Season if needed
    if 'Year' in kenpom_df.columns and 'Season' not in kenpom_df.columns:
        kenpom_df.rename(columns={'Year': 'Season'}, inplace=True)
    
    # Filter for men's data only - we only want male teams with IDs 1000-1999
    men_teams = teams[teams['TeamID'] < 3000].copy()
    print(f"Number of men's teams available for matching: {len(men_teams)}")
    
    # Display sample of team names from the merged KenPom data
    print("Original merged KenPom team names (sample):", kenpom_df['Team_Name'].head().tolist())
    
    # ===== Enhanced standardization processing =====
    # First save original names for reference in debugging
    kenpom_df['Original_Team_Name'] = kenpom_df['Team_Name']
    
    # 1. Standardization processing - use existing function
    kenpom_df['TeamName_std'] = kenpom_df['Team_Name'].apply(standardize_team_name)
    men_teams['TeamName_std'] = men_teams['TeamName'].apply(standardize_team_name)
    
    # 2. Create multiple variations for matching - from simple to complex
    
    # 2.1 Lowercase version
    kenpom_df['TeamName_lower'] = kenpom_df['TeamName_std'].str.lower()
    men_teams['TeamName_lower'] = men_teams['TeamName_std'].str.lower()
    
    # 2.2 Remove common word version (university, college etc.)
    def simplify_name(name):
        if not isinstance(name, str):
            return ""
        name = name.lower()
        name = re.sub(r'\buniversity\b|\bcollege\b|\buniv\b|\bcoll\b|\bstate\b|\bst\b', '', name)
        name = re.sub(r'\bof\b|\bat\b|\bin\b|\band\b|\bthe\b|\ba\b|\ban\b', '', name)
        name = re.sub(r'\s+', ' ', name).strip()  # Clean up extra spaces
        return name
        
    kenpom_df['TeamName_simple'] = kenpom_df['TeamName_std'].apply(simplify_name)
    men_teams['TeamName_simple'] = men_teams['TeamName_std'].apply(simplify_name)
    
    # 2.3 Keep only main name part (usually the first word)
    def extract_main_name(name):
        if not isinstance(name, str):
            return ""
        name_parts = name.lower().split()
        if not name_parts:
            return ""
        # Return the first word, usually the school's main identifier
        return name_parts[0]
        
    kenpom_df['TeamName_main'] = kenpom_df['TeamName_std'].apply(extract_main_name)
    men_teams['TeamName_main'] = men_teams['TeamName_std'].apply(extract_main_name)
    
    # 2.4 Initial name and first letter matching (e.g.: "North Carolina" -> "nc")
    def get_initials(name):
        if not isinstance(name, str):
            return ""
        words = re.findall(r'\b[a-zA-Z]+\b', name.lower())
        return ''.join(word[0] for word in words if word)
        
    kenpom_df['TeamName_initials'] = kenpom_df['TeamName_std'].apply(get_initials)
    men_teams['TeamName_initials'] = men_teams['TeamName_std'].apply(get_initials)
    
    # 2.5 Only alphanumeric characters (remove spaces and punctuation)
    def alphanumeric_only(name):
        if not isinstance(name, str):
            return ""
        return re.sub(r'[^a-zA-Z0-9]', '', name.lower())
        
    kenpom_df['TeamName_alphanum'] = kenpom_df['TeamName_std'].apply(alphanumeric_only)
    men_teams['TeamName_alphanum'] = men_teams['TeamName_std'].apply(alphanumeric_only)
    
    # 2.6 Phonetic encoding (Soundex) - capture similar sounding names
    try:
        from jellyfish import soundex
        has_jellyfish = True
        
        def get_soundex(name):
            if not isinstance(name, str):
                return ""
            words = name.lower().split()
            if not words:
                return ""
            # Apply Soundex only to the first word
            return soundex(words[0])
            
        kenpom_df['TeamName_soundex'] = kenpom_df['TeamName_std'].apply(get_soundex)
        men_teams['TeamName_soundex'] = men_teams['TeamName_std'].apply(get_soundex)
    except ImportError:
        has_jellyfish = False
        print("Jellyfish library not found. Soundex matching will be skipped.")
    
    # Display standardized result sample
    name_changes = [(orig, std) for orig, std in zip(kenpom_df['Team_Name'], kenpom_df['TeamName_std']) if orig != std]
    if name_changes:
        print("\nStandardized name example:")
        for orig, std in name_changes[:5]:
            print(f"  '{orig}' -> '{std}'")
    
    # ===== Multi-stage matching logic =====
    print("\nStarting multi-stage team name matching...")
    
    # Initialize matching tracking
    total_teams = len(kenpom_df)
    match_stats = {
        'initial': 0,
        'std': 0,
        'lower': 0,
        'simple': 0,
        'main': 0,
        'initials': 0,
        'alphanum': 0,
        'soundex': 0,
        'fuzzy': 0,
        'manual': 0
    }
    
    # 1. Create all matching dictionaries
    mapping_dicts = {}
    # Standardized name mapping
    mapping_dicts['std'] = dict(zip(men_teams['TeamName_std'], men_teams['TeamID']))
    # Lowercase mapping
    mapping_dicts['lower'] = dict(zip(men_teams['TeamName_lower'], men_teams['TeamID']))
    # Simplified name mapping
    mapping_dicts['simple'] = dict(zip(men_teams['TeamName_simple'], men_teams['TeamID']))
    # Main name mapping
    mapping_dicts['main'] = dict(zip(men_teams['TeamName_main'], men_teams['TeamID']))
    # Initial letter mapping
    mapping_dicts['initials'] = dict(zip(men_teams['TeamName_initials'], men_teams['TeamID']))
    # Alphanumeric mapping
    mapping_dicts['alphanum'] = dict(zip(men_teams['TeamName_alphanum'], men_teams['TeamID']))
    if has_jellyfish:
        # Soundex mapping
        mapping_dicts['soundex'] = dict(zip(men_teams['TeamName_soundex'], men_teams['TeamID']))
    
    # 2. Multi-stage matching
    # 1st stage: Standard name matching
    kenpom_df['TeamID'] = kenpom_df['TeamName_std'].map(mapping_dicts['std'])
    match_stats['std'] = kenpom_df['TeamID'].notna().sum()
    
    # 2nd stage: Use lowercase name matching for unmatched items
    if match_stats['std'] < total_teams:
        unmatched_mask = kenpom_df['TeamID'].isna()
        kenpom_df.loc[unmatched_mask, 'TeamID'] = kenpom_df.loc[unmatched_mask, 'TeamName_lower'].map(mapping_dicts['lower'])
        match_stats['lower'] = kenpom_df['TeamID'].notna().sum() - match_stats['std']
    
    # 3rd stage: Use simplified name matching for unmatched items
    if match_stats['std'] + match_stats['lower'] < total_teams:
        unmatched_mask = kenpom_df['TeamID'].isna()
        kenpom_df.loc[unmatched_mask, 'TeamID'] = kenpom_df.loc[unmatched_mask, 'TeamName_simple'].map(mapping_dicts['simple'])
        match_stats['simple'] = kenpom_df['TeamID'].notna().sum() - match_stats['std'] - match_stats['lower']
    
    # 4th stage: Use main name part matching for unmatched items
    if sum(match_stats.values()) < total_teams:
        unmatched_mask = kenpom_df['TeamID'].isna()
        kenpom_df.loc[unmatched_mask, 'TeamID'] = kenpom_df.loc[unmatched_mask, 'TeamName_main'].map(mapping_dicts['main'])
        match_stats['main'] = kenpom_df['TeamID'].notna().sum() - sum([match_stats[k] for k in ['std', 'lower', 'simple']])
    
    # 5th stage: Use initial letter matching for unmatched items
    if sum(match_stats.values()) < total_teams:
        unmatched_mask = kenpom_df['TeamID'].isna()
        # Initial letter may have multiple matches, so use only when unique
        unique_initials = {k: v for k, v in mapping_dicts['initials'].items() 
                           if list(mapping_dicts['initials'].values()).count(v) == 1}
        kenpom_df.loc[unmatched_mask, 'TeamID'] = kenpom_df.loc[unmatched_mask, 'TeamName_initials'].map(unique_initials)
        match_stats['initials'] = kenpom_df['TeamID'].notna().sum() - sum([match_stats[k] for k in ['std', 'lower', 'simple', 'main']])
    
    # 6th stage: Use alphanumeric matching for unmatched items
    if sum(match_stats.values()) < total_teams:
        unmatched_mask = kenpom_df['TeamID'].isna()
        unique_alphanum = {k: v for k, v in mapping_dicts['alphanum'].items() 
                           if list(mapping_dicts['alphanum'].values()).count(v) == 1}
        kenpom_df.loc[unmatched_mask, 'TeamID'] = kenpom_df.loc[unmatched_mask, 'TeamName_alphanum'].map(unique_alphanum)
        match_stats['alphanum'] = kenpom_df['TeamID'].notna().sum() - sum([match_stats[k] for k in ['std', 'lower', 'simple', 'main', 'initials']])
    
    # 7th stage: Use Soundex matching for unmatched items
    if has_jellyfish and sum(match_stats.values()) < total_teams:
        unmatched_mask = kenpom_df['TeamID'].isna()
        unique_soundex = {k: v for k, v in mapping_dicts['soundex'].items() 
                           if list(mapping_dicts['soundex'].values()).count(v) == 1}
        kenpom_df.loc[unmatched_mask, 'TeamID'] = kenpom_df.loc[unmatched_mask, 'TeamName_soundex'].map(unique_soundex)
        match_stats['soundex'] = kenpom_df['TeamID'].notna().sum() - sum([match_stats[k] for k in ['std', 'lower', 'simple', 'main', 'initials', 'alphanum']])
    
    # 8th stage: Use fuzzy matching for unmatched items
    if sum(match_stats.values()) < total_teams:
        unmatched_mask = kenpom_df['TeamID'].isna()
        unmatched_rows = kenpom_df[unmatched_mask]
        
        try:
            from rapidfuzz import process, fuzz
            has_rapidfuzz = True
        except ImportError:
            print("  rapidfuzz package not found, using basic similarity matching")
            has_rapidfuzz = False
        
        # Perform fuzzy matching on unmatched names
        for idx, row in unmatched_rows.iterrows():
            team_name = row['TeamName_std']
            
            if has_rapidfuzz:
                # Use Rapidfuzz to calculate similarity
                matches = process.extract(
                    team_name,
                    men_teams['TeamName_std'].unique(),
                    scorer=fuzz.token_sort_ratio,  # Use token_sort_ratio, not case sensitive
                    limit=3
                )
                
                best_match = matches[0] if matches else None
                
                if best_match and best_match[1] >= 85:  # Similarity threshold 85%
                    matched_name = best_match[0]
                    matched_id = mapping_dicts['std'].get(matched_name)
                    
                    if matched_id is not None:
                        kenpom_df.loc[idx, 'TeamID'] = matched_id
                        print(f"  Fuzzy matched: '{team_name}' -> '{matched_name}' (score: {best_match[1]}) -> ID: {matched_id}")
            else:
                # Basic similarity calculation method
                men_team_names = men_teams['TeamName_std'].unique()
                
                # Calculate word overlap rate
                def word_overlap(name1, name2):
                    words1 = set(re.findall(r'\b[a-zA-Z]+\b', name1.lower()))
                    words2 = set(re.findall(r'\b[a-zA-Z]+\b', name2.lower()))
                    if not words1 or not words2:
                        return 0
                    overlap = len(words1 & words2)
                    total = len(words1 | words2)
                    return overlap / total if total > 0 else 0
                
                best_score = 0
                best_match = None
                
                for ref_name in men_team_names:
                    score = word_overlap(team_name, ref_name)
                    if score > best_score and score >= 0.65:  # 65% or more word overlap
                        best_score = score
                        best_match = ref_name
                
                if best_match:
                    matched_id = mapping_dicts['std'].get(best_match)
                    if matched_id is not None:
                        kenpom_df.loc[idx, 'TeamID'] = matched_id
                        print(f"  Fuzzy matched: '{team_name}' -> '{best_match}' (score: {best_score:.2f}) -> ID: {matched_id}")
        
        match_stats['fuzzy'] = kenpom_df['TeamID'].notna().sum() - sum([match_stats[k] for k in match_stats if k != 'fuzzy' and k != 'manual'])
    
    # 9th stage: Manual mapping for common unmatched teams
    if sum(match_stats.values()) < total_teams:
        unmatched_mask = kenpom_df['TeamID'].isna()
        
        # Create manual mapping table - based on common unmatched teams
        manual_mappings = {
            # KenPom name (standardized): TeamID
            'kentucky': 1314,
            'duke': 1181,
            'north carolina': 1314,
            'kansas': 1242,
            'michigan state': 1164,
            'villanova': 1437,
            'gonzaga': 1211,
            'virginia': 1438,
            'wisconsin': 1458,
            'connecticut': 1154,
            'purdue': 1345,
            'baylor': 1124,
            'houston': 1222,
            'indiana': 1232,
            'ohio state': 1326,
            'arizona': 1112,
            'louisville': 1314,
            'memphis': 1266,
            'arkansas': 1116,
            'marquette': 1261,
            'michigan': 1277,
            'texas': 1400,
            'tennessee': 1394,
            'ucla': 1417,
            'kansas state': 1243,
            'iowa state': 1234,
            'creighton': 1158,
            'alabama': 1104,
            'auburn': 1120,
            'florida': 1196,
            'texas tech': 1401,
            'lsu': 1249,
            'indiana': 1232,
            'illinois': 1228,
            'saint marys': 1361,
            'wichita state': 1451,
            'utah state': 1426,
            'dayton': 1163,
            'san diego state': 1365,
            'oklahoma': 1328,
            'cincinnati': 1140,
            'washington state': 1448,
            'byu': 1138,
            'notre dame': 1321,
            'miami florida': 1276,
            'providence': 1344,
            'tcu': 1388,
            'west virginia': 1450,
            'virginia tech': 1439,
            'colorado': 1149,
            'arizona state': 1113,
            'oregon': 1332,
            'xavier': 1462,
            'texas am': 1399,
            'southern california': 1417,
            'seton hall': 1376,
            'clemson': 1147,
            'michigan state': 1278,
            'florida state': 1197,
            'nc state': 1314,
            'utah': 1425,
            'vanderbilt': 1430,
            'butler': 1130,
            'st johns': 1393,
            'st bonaventure': 1366,
            'davidson': 1161,
            'iowa': 1233,
            'northwestern': 1318,
            'wake forest': 1441,
            'mississippi': 1287,
            'mississippi state': 1288,
            'georgia': 1210,
            'missouri': 1294,
            'california': 1132,
            'stanford': 1384,
            'washington': 1447,
            'maryland': 1263,
            'penn state': 1339,
            'rutgers': 1359,
            'nebraska': 1307,
            'south carolina': 1380,
            'oklahoma state': 1329,
            'syracuse': 1393,
            'georgetown': 1207,
            'colorado state': 1150,
            'texas a&m corpus christi': 1397,
            'wagner': 1440,
            'boise state': 1137,
            'south dakota state': 1383,
            'vermont': 1434,
            'akron': 1105,
            'colgate': 1152,
            'montana state': 1296,
            'stetson': 1385,
            'james madison': 1236,
            'toledo': 1404,
            'nc central': 1312,
            'charleston': 1169,
            'grambling state': 1199,
            'eastern washington': 1178,
            'kent state': 1245,
            'iona': 1231,
            'missouri st': 1294,
            'wright st': 1459,
            'bethune cookman': 1127,
            'alcorn state': 1108,
            'jacksonville state': 1235,
            'morehead state': 1297,
            'st thomas': 1410,
            'st francis pa': 1368,
            'liu': 1351,
            'queens nc': 1350,
            'houston chr': 1223,
            'detroit': 1171,
            'detroit mercy': 1171,
            'cal st fullerton': 1133,
            'santa clara': 1373,
            'w kentucky': 1455,
            'mercer': 1274,
            'fiu': 1199,
            'old dominion': 1330,
            'wyoming': 1461,
            'george mason': 1205,
            'yale': 1463,
            'virginia commonwealth': 1438,
            'vcu': 1438,
            'uconn': 1154,
            'unlv': 1316,
            'hofstra': 1221,
            'richmond': 1357,
            'middle tennessee': 1282,
            'fordham': 1199,
            'texas a&m': 1398,
            'indiana state': 1231,
            'north texas': 1322,
            'south florida': 1381,
            'south dakota': 1382,
            'saint louis': 1373,
            'st marys': 1373,
            'boston college': 1125,
            # Add more mappings...
        }
        
        # Lowercase version
        manual_mappings_lower = {k.lower(): v for k, v in manual_mappings.items()}
        
        # Apply manual mapping
        for idx, row in kenpom_df[unmatched_mask].iterrows():
            team_name = row['TeamName_std'].lower()
            
            # Try direct match
            if team_name in manual_mappings_lower:
                kenpom_df.loc[idx, 'TeamID'] = manual_mappings_lower[team_name]
                print(f"  Manual mapping: '{row['TeamName_std']}' -> ID: {manual_mappings_lower[team_name]}")
                continue
            
            # Try substring match
            for map_name, map_id in manual_mappings_lower.items():
                if (len(team_name) > 4 and team_name in map_name) or (len(map_name) > 4 and map_name in team_name):
                    kenpom_df.loc[idx, 'TeamID'] = map_id
                    print(f"  Manual substring mapping: '{row['TeamName_std']}' -> '{map_name}' -> ID: {map_id}")
                    break
        
        match_stats['manual'] = kenpom_df['TeamID'].notna().sum() - sum([match_stats[k] for k in match_stats if k != 'manual'])
    
    # Print matching statistics
    print("\nTeam name matching statistics:")
    for stage, count in match_stats.items():
        if count > 0:
            print(f"  {stage}: {count} teams matched ({count/total_teams*100:.2f}%)")
    
    total_matched = kenpom_df['TeamID'].notna().sum()
    print(f"\nTotal: {total_matched}/{total_teams} team names successfully matched ({total_matched/total_teams*100:.2f}%)")
    
    # Print examples of unmatched teams
    unmatched_mask = kenpom_df['TeamID'].isna()
    if unmatched_mask.any():
        unmatched = kenpom_df[unmatched_mask][['Original_Team_Name', 'TeamName_std']].drop_duplicates()
        print(f"\nStill {len(unmatched)} unique unmatched team names:")
        
        # Display some examples of unmatched names
        if len(unmatched) > 0:
            print("Unmatched name example:")
            for i, (idx, row) in enumerate(unmatched.head(10).iterrows()):
                print(f"  {i+1}. Original: '{row['Original_Team_Name']}', Standardized: '{row['TeamName_std']}'")
    
    # Team name matching rate by season
    if 'Season' in kenpom_df.columns:
        print("\nTeam name matching rate by season:")
        for season in sorted(kenpom_df['Season'].unique()):
            season_df = kenpom_df[kenpom_df['Season'] == season]
            season_matched = season_df['TeamID'].notna().sum()
            season_total = len(season_df)
            print(f"  Season {season}: {season_matched}/{season_total} teams matched ({season_matched/season_total*100:.1f}%)")
            
            # If there are unmatched, display the first few
            if season_matched < season_total:
                unmatched_season = season_df[season_df['TeamID'].isna()]['Original_Team_Name'].unique()[:5]
                unmatched_teams = ", ".join([f'"{t}"' for t in unmatched_season])
                print(f"    Examples of unmatched teams: {unmatched_teams}")
    
    # Drop temporary columns and convert TeamID to integer
    kenpom_df = kenpom_df.drop(columns=[col for col in kenpom_df.columns if col.startswith('TeamName_') and col != 'TeamName_std'])
    
    # Filter unmatched rows, ensure TeamID is integer
    matched_kenpom_df = kenpom_df.dropna(subset=['TeamID']).copy()
    if len(matched_kenpom_df) > 0:
        matched_kenpom_df['TeamID'] = matched_kenpom_df['TeamID'].astype(int)
    
    print(f"\nFinal result: {len(matched_kenpom_df)}/{len(kenpom_df)} teams matched ({len(matched_kenpom_df)/len(kenpom_df)*100:.1f}%)")
    
    return matched_kenpom_df
