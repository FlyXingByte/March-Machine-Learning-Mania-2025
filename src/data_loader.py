import os
import pandas as pd
import numpy as np
import re

def load_data(data_path, start_year=2018, stage=2, test_mode=False):
    """
    Load all necessary data files
    
    Args:
        data_path: Path to the data directory
        start_year: Only include seasons from this year onwards
        stage: Competition stage (1 or 2)
        test_mode: If True, only load 10 games per year for quick testing
    
    Returns:
        Tuple of dataframes (season_detail, tourney_detail, seeds, teams, submission)
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
    
    # Load team information（合并 MTeams.csv 和 WTeams.csv）
    teams = load_teams(data_path)
    
    # If test mode is enabled, only keep a small sample of games per year for quick testing
    if test_mode:
        # Group by Season and sample 10 games per season for regular season
        season_detail = season_detail.groupby('Season').apply(
            lambda x: x.sample(min(10, len(x)), random_state=42)
        ).reset_index(drop=True)
        
        # Group by Season and sample min(10, available games) per season for tournament
        tourney_detail = tourney_detail.groupby('Season').apply(
            lambda x: x.sample(min(10, len(x)), random_state=42)
        ).reset_index(drop=True)
        
        print(f"Test mode enabled: Reduced regular season data to {len(season_detail)} rows")
        print(f"Test mode enabled: Reduced tournament data to {len(tourney_detail)} rows")
    
    # Load the appropriate submission file based on stage
    # Using the correct file names without the "M" prefix
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
        print("Creating an empty submission DataFrame as fallback")
        # Create an empty submission dataframe as fallback
        submission = pd.DataFrame(columns=['ID', 'Pred'])
        
    return season_detail, tourney_detail, seeds, teams, submission

def load_teams(data_path):
    """
    Load and merge men's and women's teams information.
    
    Args:
        data_path: Path to the data directory.
        
    Returns:
        teams: 合并后的 DataFrame，包含 'TeamID' 和 'TeamName' 列
    """
    try:
        mteams = pd.read_csv(os.path.join(data_path, "MTeams.csv"))
        wteams = pd.read_csv(os.path.join(data_path, "WTeams.csv"))
        teams = pd.concat([mteams, wteams], ignore_index=True)
        print(f"Loaded {len(teams)} teams from MTeams and WTeams.")
    except Exception as e:
        print("Error loading teams data:", e)
        teams = pd.DataFrame()
    return teams

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
    Merge regular season and tournament data into a single DataFrame.
    Adds a new column 'GameType' to indicate whether the game is from the regular season or tournament.

    Args:
        season_detail: Regular season game data
        tourney_detail: Tournament game data

    Returns:
        Combined DataFrame with game data, including the 'GameType' column.
    """
    # Make copies to avoid modifying original DataFrames
    season_detail = season_detail.copy()
    tourney_detail = tourney_detail.copy()
    
    # Add a descriptive column for game type
    season_detail['GameType'] = 'Regular'
    tourney_detail['GameType'] = 'Tournament'
    
    # Merge all game data (regular season and tournament)
    games = pd.concat([season_detail, tourney_detail], ignore_index=True)
    print(f"Merged game data total rows: {len(games)}")
    
    # If game location info is available, map it (e.g., 'A':1, 'H':2, 'N':3)
    if 'WLoc' in games.columns:
        games['WLoc'] = games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})
        
    return games

def standardize_team_name(name):
    """
    对球队名称进行标准化处理，处理常见的缩写问题，并补充完整名称。
    """
    # 定义手动映射字典，根据实际情况扩充
    corrections = {
        # Common abbreviations to full names
        "Michigan St": "Michigan State",
        "Michigan St.": "Michigan State",
        "Iowa St": "Iowa State",
        "Iowa St.": "Iowa State",
        "St. John's": "St John's",
        "Saint Mary's": "St Mary's",
        "St. Mary's": "St Mary's",
        "Mississippi St": "Mississippi State",
        "Mississippi St.": "Mississippi State",
        "Ohio St": "Ohio State",
        "Ohio St.": "Ohio State",
        "Boise St": "Boise State",
        "Boise St.": "Boise State",
        "Utah St": "Utah State",
        "Utah St.": "Utah State",
        "Colorado St": "Colorado State",
        "Colorado St.": "Colorado State",
        "McNeese": "McNeese State",
        "McNeese St.": "McNeese State",
        "Washington St": "Washington State",
        "Washington St.": "Washington State",
        "Penn St": "Penn State",
        "Penn St.": "Penn State",
        "Kent St": "Kent State",
        "Kent St.": "Kent State",
        "Fresno St": "Fresno State",
        "Fresno St.": "Fresno State",
        "San Diego St": "San Diego State",
        "San Diego St.": "San Diego State",
        "Oregon St": "Oregon State",
        "Oregon St.": "Oregon State",
        "Kansas St": "Kansas State",
        "Kansas St.": "Kansas State",
        "Montana St": "Montana State",
        "Montana St.": "Montana State",
        "San Jose St": "San Jose State",
        "San Jose St.": "San Jose State",
        "Illinois St": "Illinois State",
        "Illinois St.": "Illinois State",
        "Florida St": "Florida State",
        "Florida St.": "Florida State",
        "Appalachian St": "Appalachian State",
        "Appalachian St.": "Appalachian State",
        "Arizona St": "Arizona State",
        "Arizona St.": "Arizona State",
        "Alabama St": "Alabama State",
        "Alabama St.": "Alabama State",
        "Arkansas St": "Arkansas State",
        "Arkansas St.": "Arkansas State",
        "Ball St": "Ball State",
        "Ball St.": "Ball State",
        "Cleveland St": "Cleveland State",
        "Cleveland St.": "Cleveland State",
        "Coppin St": "Coppin State",
        "Coppin St.": "Coppin State",
        "Georgia St": "Georgia State",
        "Georgia St.": "Georgia State",
        "Idaho St": "Idaho State",
        "Idaho St.": "Idaho State",
        "Indiana St": "Indiana State",
        "Indiana St.": "Indiana State",
        "Jacksonville St": "Jacksonville State",
        "Jacksonville St.": "Jacksonville State",
        "Kennesaw St": "Kennesaw State",
        "Kennesaw St.": "Kennesaw State",
        "Morehead St": "Morehead State",
        "Morehead St.": "Morehead State",
        "Mississippi Val St": "Mississippi Valley State",
        "Mississippi Val St.": "Mississippi Valley State",
        "Mississippi Valley St": "Mississippi Valley State",
        "Mississippi Valley St.": "Mississippi Valley State",
        "Norfolk St": "Norfolk State",
        "Norfolk St.": "Norfolk State",
        "Sacramento St": "Sacramento State",
        "Sacramento St.": "Sacramento State",
        "Southern Utah": "S Utah",
        "Southern Illinois": "S Illinois", 
        "Southern California": "USC",
        "Southern Mississippi": "Southern Miss",
        "Texas A&M-CC": "Texas A&M Corpus Chris",
        "Loyola Chicago": "Loyola-Chicago",
        "Loyola Marymount": "Loy Marymount",
        "N.C. State": "NC State",
        "UConn": "Connecticut",
        "UCF": "Central Florida",
        "UNLV": "Nevada Las Vegas",
        "UTEP": "Texas El Paso",
        "UTSA": "UT San Antonio",
        "LSU": "Louisiana State",
        "VCU": "Virginia Commonwealth",
        "SMU": "Southern Methodist",
        "BYU": "Brigham Young",
        "Ole Miss": "Mississippi",
        "UNC": "North Carolina",
        "UNC Greensboro": "NC Greensboro",
        "UNC Wilmington": "NC Wilmington",
        "UNC Asheville": "NC Asheville",
        "UNC Charlotte": "Charlotte",
        "UC Davis": "California Davis",
        "UC Irvine": "Cal Irvine",
        "UC Riverside": "Cal Riverside",
        "UC San Diego": "Cal San Diego",
        "UC Santa Barbara": "Cal Santa Barbara",
        "CSU Bakersfield": "Cal St Bakersfield",
        "CSU Fullerton": "Cal St Fullerton",
        "CSU Northridge": "Cal St Northridge",
        
        # Add new mappings from the error message
        "Saint Joseph's": "St Joseph's",
        "Oklahoma St.": "Oklahoma State",
        "CSUN": "Cal St Northridge",
        "St. Bonaventure": "St Bonaventure",
        "Saint Louis": "St Louis",
        "Florida Atlantic": "Florida Atl",
        "Middle Tennessee": "Middle Tenn",
        "George Washington": "G Washington",
        
        # Additional common variations
        "North Carolina St": "NC State",
        "North Carolina State": "NC State",
        "TCU": "Texas Christian",
        "USC": "Southern California",
        "USC Upstate": "SC Upstate",
        "UTRGV": "UT Rio Grande Valley",
        "UMBC": "MD Baltimore County",
        "UMKC": "Missouri KC",
        "UAB": "Alabama Birmingham",
        "Saint Peter's": "St Peter's",
        "Saint Francis PA": "St Francis PA",
        "Saint Francis NY": "St Francis NY",
        "Saint Francis Brooklyn": "St Francis NY",
        "Saint Joseph's": "St Joseph's",
        "Saint John's": "St John's",
        "Saint Thomas": "St Thomas",
        "Saint Francis": "St Francis PA",
        "Long Island": "Long Island University",
        "LIU": "Long Island University",
        "UTEP": "Texas El Paso",
        "San Francisco": "San Francisco",
        "Stephen F. Austin": "SF Austin",
        "Miami FL": "Miami (FL)",
        "Miami OH": "Miami (OH)",
        "Miami (Florida)": "Miami (FL)",
        "Miami (Ohio)": "Miami (OH)",
        "FIU": "Florida Intl",
        "Florida Int'l": "Florida Intl",
        "East Tennessee St": "E Tennessee St",
        "East Tennessee State": "E Tennessee St",
        "Western Kentucky": "W Kentucky",
        "Western Michigan": "W Michigan",
        "Western Carolina": "W Carolina",
        "Western Illinois": "W Illinois",
        "Eastern Kentucky": "E Kentucky",
        "Eastern Michigan": "E Michigan",
        "Eastern Illinois": "E Illinois",
        "Eastern Washington": "E Washington",
        "Central Michigan": "C Michigan",
        "Central Arkansas": "C Arkansas",
        "Central Connecticut": "C Connecticut St",
        "Central Connecticut St": "C Connecticut St",
        "UNC-Wilmington": "NC Wilmington",
        "UNC-Greensboro": "NC Greensboro",
        "UNC-Asheville": "NC Asheville",
        "College of Charleston": "Charleston",
        "Charleston Southern": "Charleston So",
        "Cal State Bakersfield": "Cal St Bakersfield",
        "Cal State Fullerton": "Cal St Fullerton",
        "Cal State Northridge": "Cal St Northridge",
        "Cal Baptist": "California Baptist",
        "Seattle": "Seattle U",
        "Bowling Green": "Bowling Green State",
        "Georgia Tech": "Georgia Tech",
        "Virginia Tech": "Virginia Tech",
        "Louisiana Tech": "Louisiana Tech",
        "Texas Tech": "Texas Tech",
        "Texas A&M": "Texas A&M",
        "Florida A&M": "Florida A&M",
        "Alabama A&M": "Alabama A&M",
        "NC A&T": "NC A&T",
        "Grambling": "Grambling State",
        "FGCU": "Florida Gulf Coast",
        
        # Additional mappings for remaining unmapped teams
        "Kent St.": "Kent State",
        "St. Thomas": "St Thomas MN",
        "South Carolina St.": "SC State",
        "UMass Lowell": "Massachusetts Lowell",
        "Mount St. Mary's": "Mt St Mary's",
        "Bethune Cookman": "Bethune-Cookman",
        "IU Indy": "IUPUI",
        "Fairleigh Dickinson": "F Dickinson",
        "North Carolina Central": "NC Central",
        "SIU Edwardsville": "S Illinois Edwardsville",
        "UT Arlington": "Texas Arlington",
        "North Florida": "N Florida",
        
        # Final 3 remaining unmapped teams
        "North Carolina A&T": "NC A&T",
        "Sacramento St.": "Sacramento State",
        "Cal St. Fullerton": "Cal St Fullerton",
    }
    
    # First check for exact matches in our corrections dictionary
    if name in corrections:
        return corrections[name]
    
    # Remove common suffixes for standardization
    name = name.replace(" Spartans", "")
    name = name.replace(" Cyclones", "")
    name = name.replace(" Red Storm", "")
    name = name.replace(" Gaels", "")
    name = name.replace(" Bulldogs", "")
    name = name.replace(" Buckeyes", "")
    name = name.replace(" Broncos", "")
    name = name.replace(" Aggies", "")
    name = name.replace(" Rams", "")
    name = name.replace(" Cowboys", "")
    name = name.replace(" Eagles", "")
    name = name.replace(" Wildcats", "")
    name = name.replace(" Wolverines", "")
    name = name.replace(" Hoosiers", "")
    name = name.replace(" Boilermakers", "")
    name = name.replace(" Mountaineers", "")
    name = name.replace(" Blue Devils", "")
    name = name.replace(" Tigers", "")
    name = name.replace(" Cardinal", "")
    name = name.replace(" Crimson Tide", "")
    name = name.replace(" Razorbacks", "")
    name = name.replace(" Jayhawks", "")
    name = name.replace(" Tar Heels", "")
    name = name.replace(" Huskies", "")
    name = name.replace(" Volunteers", "")
    name = name.replace(" Demon Deacons", "")
    
    # 如果名称以 " St." 结尾且不是已知的特殊情况，则替换为 " State"
    if re.search(r'\sSt\.$', name):
        name = re.sub(r'\sSt\.$', ' State', name)
        
    # 标准化 "Saint" 为 "St"
    name = re.sub(r'^Saint\s', 'St ', name)
        
    # 处理特殊的缩写情况
    name = re.sub(r'^North Carolina$', 'NC', name)
    name = re.sub(r'^South Carolina$', 'SC', name)
    name = re.sub(r'^North Carolina A&T$', 'NC A&T', name)
    
    # After applying corrections, check if the name is in corrections again
    if name in corrections:
        return corrections[name]
    
    return name.strip()

def load_kenpom_data(data_path, teams):
    """
    Load KenPom external data from 'kenpom-ncaa-2025.csv' and map TeamName to TeamID.
    使用 standardize_team_name 对球队名称进行标准化，以便与 teams 数据匹配。

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
    
    # 检查文件中是否存在 'TeamName' 列，如果没有，则尝试 'School' 或 'Team'
    if 'TeamName' not in kenpom_df.columns:
        if 'School' in kenpom_df.columns:
            kenpom_df.rename(columns={'School': 'TeamName'}, inplace=True)
        elif 'Team' in kenpom_df.columns:
            kenpom_df.rename(columns={'Team': 'TeamName'}, inplace=True)
        else:
            raise KeyError("无法在 KenPom 数据中找到 'TeamName'、'School' 或 'Team' 列，请检查 CSV 文件格式。")
    
    # 显示前几个未处理的球队名称，帮助调试
    print("Original KenPom team names (sample):", kenpom_df['TeamName'].head().tolist())
    
    # 对 KenPom 数据中的球队名称进行标准化处理
    kenpom_df['TeamName_std'] = kenpom_df['TeamName'].apply(standardize_team_name)
    
    # 对 teams 数据中的球队名称进行标准化处理
    teams['TeamName_std'] = teams['TeamName'].apply(standardize_team_name)
    
    # 创建字典用于调试
    debug_mapping = {name: std for name, std in zip(teams['TeamName'], teams['TeamName_std'])}
    print("Sample of team name standardization:")
    for i, (orig, std) in enumerate(list(debug_mapping.items())[:5]):
        print(f"  {i+1}. '{orig}' -> '{std}'")
    
    # 创建映射字典：标准化名称 -> TeamID
    mapping = teams.set_index('TeamName_std')['TeamID'].to_dict()
    
    # 对标准化后的名称进行小写处理并映射到TeamID
    kenpom_df['TeamName_std_lower'] = kenpom_df['TeamName_std'].str.lower().str.strip()
    teams['TeamName_std_lower'] = teams['TeamName_std'].str.lower().str.strip()
    
    # 重建映射字典，使用小写字符串
    mapping_lower = teams.set_index('TeamName_std_lower')['TeamID'].to_dict()
    
    # 应用映射
    kenpom_df['TeamID'] = kenpom_df['TeamName_std_lower'].map(mapping_lower)
    
    # 调试匹配的问题
    print("\nSample of standardized KenPom team names:")
    for i, row in kenpom_df.head().iterrows():
        mapped_id = row['TeamID']
        status = "✓" if not pd.isna(mapped_id) else "✗"
        print(f"  {status} '{row['TeamName']}' -> '{row['TeamName_std']}' -> ID: {mapped_id}")
    
    # 检查映射的有效性
    unmapped_teams = kenpom_df[kenpom_df['TeamID'].isna()].copy()
    if len(unmapped_teams) > 0:
        unmapped_count = len(unmapped_teams)
        print(f"Warning: {unmapped_count} teams from KenPom data could not be mapped to TeamIDs initially.")
        
        # 创建备用名称映射字典（正式名称->ID）
        team_names_dict = dict(zip(teams['TeamName_std_lower'], teams['TeamID']))
        
        # 尝试通过部分匹配找到未匹配的团队
        for i, row in unmapped_teams.iterrows():
            orig_name = row['TeamName']
            std_name = row['TeamName_std']
            
            # 尝试1: 移除"Univ"，"University"，"College"等词汇后再匹配
            simple_name = std_name
            simple_name = re.sub(r'\sUniv(ersity)?(\sof)?', '', simple_name, flags=re.IGNORECASE)
            simple_name = re.sub(r'\sCollege', '', simple_name, flags=re.IGNORECASE)
            simple_name = re.sub(r'\sUniversidad', '', simple_name, flags=re.IGNORECASE)
            simple_name = simple_name.lower().strip()
            
            # 尝试匹配简化的名称
            for official_name, team_id in team_names_dict.items():
                official_simple = re.sub(r'\sUniv(ersity)?(\sof)?', '', official_name, flags=re.IGNORECASE)
                official_simple = re.sub(r'\sCollege', '', official_simple, flags=re.IGNORECASE)
                official_simple = re.sub(r'\sUniversidad', '', official_simple, flags=re.IGNORECASE)
                official_simple = official_simple.lower().strip()
                
                # 如果简化后的名称匹配，或者一个名称是另一个名称的子串，则认为是匹配的
                if simple_name == official_simple or \
                   (len(simple_name) > 5 and simple_name in official_simple) or \
                   (len(official_simple) > 5 and official_simple in simple_name):
                    kenpom_df.loc[i, 'TeamID'] = team_id
                    print(f"  ✓ Found match through simplification: '{orig_name}' -> '{std_name}' -> '{official_name}' -> ID: {team_id}")
                    break
            
            # 如果仍未找到匹配项，尝试手动映射一些常见的名称
            if pd.isna(kenpom_df.loc[i, 'TeamID']):
                manual_matches = {
                    'st marys': 3181,  # 例如，如果 St. Mary's 对应的ID是 3181
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
                
                # 检查是否有手动映射
                simple_key = std_name.lower().strip()
                if simple_key in manual_matches:
                    kenpom_df.loc[i, 'TeamID'] = manual_matches[simple_key]
                    print(f"  ✓ Found match through manual mapping: '{orig_name}' -> '{std_name}' -> ID: {manual_matches[simple_key]}")
        
        # 重新检查未映射的团队
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
