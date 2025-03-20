import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import re
import copy


def parse_tournament_structure(submission_df):
    """
    Parse tournament structure from the sample submission file.
    
    Parameters:
        submission_df: DataFrame with 'ID' column containing matchup IDs
        
    Returns:
        Dictionary containing tournament structure
    """
    print("Parsing tournament structure from submission data...")
    
    # Check if ID column exists
    id_column = 'original_ID' if 'original_ID' in submission_df.columns else 'ID'
    
    if id_column not in submission_df.columns:
        raise ValueError(f"ID column not found (tried 'ID' and 'original_ID')")
        
    # ID format example: "2025_1101_1106" represents a game between TeamID 1101 and 1106 in 2025
    seasons = submission_df[id_column].str.split('_').str[0].unique()
    print(f"Found {len(seasons)} tournament seasons: {seasons}")
    
    tournament_structure = {}
    
    for season in seasons:
        season_games = submission_df[submission_df[id_column].str.startswith(f"{season}_")]
        
        # Extract all team IDs that appear in the tournament
        team_ids = set()
        for id_str in season_games[id_column]:
            parts = id_str.split('_')
            if len(parts) >= 3:
                team_ids.add(int(parts[1]))
                team_ids.add(int(parts[2]))
        
        tournament_structure[season] = {
            'team_ids': sorted(list(team_ids)),
            'games': season_games[id_column].tolist()
        }
        
        print(f"Season {season}: Found {len(team_ids)} teams and {len(season_games)} potential matchups")
    
    return tournament_structure


def build_bracket(tournament_structure, seed_dict, season):
    """
    Build bracket structure for a tournament simulation.
    
    Parameters:
        tournament_structure: Dictionary containing tournament structure
        seed_dict: Dictionary mapping Season_TeamID to seed value
        season: Season for which to build the bracket
        
    Returns:
        Dictionary containing bracket structure
    """
    teams = tournament_structure[season]['team_ids']
    bracket = {
        'teams': {},
        'rounds': {},
        'current_round': 1
    }
    
    # Fill in team information
    for team_id in teams:
        seed = seed_dict.get(f"{season}_{team_id}", 16)  # If not found, default to seed 16
        bracket['teams'][team_id] = {
            'id': team_id,
            'seed': seed,
            'eliminated': False
        }
    
    # Try to infer bracket structure from game IDs
    # This is a simplified approach and may need to be adjusted for specific tournament formats
    all_matchups = tournament_structure[season]['games']
    
    # Calculate unique team pairings
    team_pairs = set()
    for matchup in all_matchups:
        parts = matchup.split('_')
        if len(parts) >= 3:
            team_pair = f"{min(parts[1], parts[2])}_{max(parts[1], parts[2])}"
            team_pairs.add(team_pair)
    
    # Estimate team count based on unique pairings
    estimated_teams = len(team_pairs) + 1  # n teams need n-1 games
    if estimated_teams != len(teams):
        print(f"Warning: Estimated {estimated_teams} teams, but found {len(teams)} team IDs")
    
    # Determine number of rounds (assuming power of 2)
    num_rounds = 1
    while 2**num_rounds < len(teams):
        num_rounds += 1
    
    print(f"Building a bracket with {len(teams)} teams and {num_rounds} rounds")
    
    # Initialize rounds
    for r in range(1, num_rounds + 1):
        bracket['rounds'][r] = {
            'games': [],
            'completed': False
        }
    
    # For simplicity in this implementation, we won't try to fully reconstruct the actual bracket
    # Instead, we'll note which teams could potentially play each other in each round
    # This is sufficient for most Monte Carlo simulation purposes
    
    return bracket


def run_tournament_simulation(bracket, win_probability_lookup, num_simulations=10000):
    """
    Run Monte Carlo simulation of the tournament.
    
    Args:
        bracket: Tournament bracket structure
        win_probability_lookup: Function to lookup win probability for a matchup
        num_simulations: Number of simulations to run
        
    Returns:
        Dictionary with simulation results
    """
    print(f"Running {num_simulations} tournament simulations...")
    
    # Initialize results tracking
    results = {
        'champion_counts': {},        # How many times each team won
        'final_four_counts': {},      # How many times each team reached final four
        'elite_eight_counts': {},     # How many times each team reached elite eight
        'sweet_sixteen_counts': {},   # How many times each team reached sweet sixteen
        'round_probabilities': {},    # Probability of each team reaching each round
        'matchup_probabilities': {}   # Probability of each potential matchup occurring
    }
    
    # Initialize counts
    for team_id in bracket['teams']:
        results['champion_counts'][team_id] = 0
        results['final_four_counts'][team_id] = 0
        results['elite_eight_counts'][team_id] = 0
        results['sweet_sixteen_counts'][team_id] = 0
        results['round_probabilities'][team_id] = {}
    
    # Get total number of rounds
    num_rounds = max(bracket['rounds'].keys())
    
    # For the first few simulations, print detailed logs
    verbose_simulations = min(3, num_simulations)
    
    # Run simulations
    from tqdm import tqdm
    for sim in tqdm(range(num_simulations), desc="Simulating tournaments"):
        # Reset bracket for this simulation
        sim_bracket = reset_bracket(bracket)
        
        # More detailed logging for the first few simulations
        verbose = sim < verbose_simulations
        if verbose:
            print(f"\n=== Simulation {sim+1} ===")
        
        # Run this tournament simulation
        champion = simulate_tournament(sim_bracket, win_probability_lookup)
        
        # Track results
        if champion in results['champion_counts']:
            results['champion_counts'][champion] += 1
        else:
            print(f"Warning: Champion {champion} not found in results tracking. Adding now.")
            results['champion_counts'][champion] = 1
            # Initialize other counters for this team
            results['final_four_counts'][champion] = 0
            results['elite_eight_counts'][champion] = 0
            results['sweet_sixteen_counts'][champion] = 0
            results['round_probabilities'][champion] = {}
        
        # Track which teams made it to various rounds
        for team_id, team_info in sim_bracket['teams'].items():
            if 'eliminated_in_round' in team_info:
                eliminated_round = team_info['eliminated_in_round']
                
                # Count teams that made it to specific milestone rounds
                if eliminated_round > num_rounds - 1 or team_id == champion:
                    results['final_four_counts'][team_id] += 1
                if eliminated_round > num_rounds - 2 or team_id == champion:
                    results['elite_eight_counts'][team_id] += 1
                if eliminated_round > num_rounds - 3 or team_id == champion:
                    results['sweet_sixteen_counts'][team_id] += 1
                
                # Track round probabilities
                for round_num in range(1, num_rounds + 1):
                    if round_num not in results['round_probabilities'][team_id]:
                        results['round_probabilities'][team_id][round_num] = 0
                    
                    if eliminated_round >= round_num or team_id == champion:
                        results['round_probabilities'][team_id][round_num] += 1
        
        # After first simulation, check for potential issues
        if sim == 0:
            # Check if champion has received a count
            if results['champion_counts'].get(champion, 0) != 1:
                print(f"WARNING: Champion tracking failed for {champion}")
    
    # Convert counts to probabilities
    for team_id in results['round_probabilities']:
        for round_num in results['round_probabilities'][team_id]:
            results['round_probabilities'][team_id][round_num] /= num_simulations
    
    # Process matchup probabilities
    if 'matchup_counts' in results:
        for matchup, count in results['matchup_counts'].items():
            results['matchup_probabilities'][matchup] = count / num_simulations
    
    # Calculate champion probabilities
    results['champion_probabilities'] = {
        team_id: count / num_simulations 
        for team_id, count in results['champion_counts'].items()
    }
    
    # Calculate final four probabilities
    results['final_four_probabilities'] = {
        team_id: count / num_simulations 
        for team_id, count in results['final_four_counts'].items()
    }
    
    return results


def reset_bracket(bracket):
    """
    Reset the bracket for a new simulation.
    
    Args:
        bracket: Tournament bracket structure
        
    Returns:
        Reset bracket structure
    """
    # Create a deep copy of the bracket
    new_bracket = copy.deepcopy(bracket)
    
    # Reset team elimination status
    for team_id in new_bracket['teams']:
        new_bracket['teams'][team_id]['eliminated'] = False
        if 'eliminated_in_round' in new_bracket['teams'][team_id]:
            del new_bracket['teams'][team_id]['eliminated_in_round']
    
    # Reset round completion status
    for round_num in new_bracket['rounds']:
        new_bracket['rounds'][round_num]['completed'] = False
        new_bracket['rounds'][round_num]['games'] = []
    
    return new_bracket


def simulate_tournament(bracket, win_probability_lookup):
    """
    Simulate a single tournament run.
    
    Args:
        bracket: Tournament bracket structure for this simulation
        win_probability_lookup: Function to lookup win probability for a matchup
        
    Returns:
        ID of tournament champion
    """
    # Get the teams and their seeds
    teams = [(team_id, info['seed']) for team_id, info in bracket['teams'].items()]
    # Sort by seed for initial matchups
    teams.sort(key=lambda x: x[1])
    
    # Number of rounds
    num_rounds = max(bracket['rounds'].keys())
    
    # Simulate each round
    remaining_teams = [team_id for team_id, _ in teams]
    
    # Safety check: ensure we have at least one team
    if not remaining_teams:
        print("Warning: No teams found in bracket. Returning dummy champion.")
        return 0  # Return a dummy champion ID
    
    # Log initial number of teams
    print(f"Starting tournament simulation with {len(remaining_teams)} teams and {num_rounds} rounds")
    
    for current_round in range(1, num_rounds + 1):
        # Safety check: make sure we have teams remaining
        if not remaining_teams:
            print(f"Warning: No teams left before round {current_round}. Using last known teams.")
            # If we somehow lost all teams, use the teams from last round
            teams_by_seed = sorted([(team_id, bracket['teams'][team_id]['seed']) 
                                   for team_id in bracket['teams'] if not bracket['teams'][team_id]['eliminated']], 
                                  key=lambda x: x[1])
            if teams_by_seed:
                remaining_teams = [team_id for team_id, _ in teams_by_seed]
            else:
                # If all teams are marked as eliminated, just pick a random team as champion
                import random
                champion = random.choice(list(bracket['teams'].keys()))
                print(f"All teams were eliminated. Randomly selected Team {champion} as champion.")
                return champion
        
        next_round_teams = []
        
        # Ensure even number of teams by adding a "bye" if necessary
        if len(remaining_teams) % 2 != 0:
            print(f"Odd number of teams ({len(remaining_teams)}) in round {current_round}. Adding a bye.")
            # Add a placeholder team that will always lose
            remaining_teams.append(-1)  # Use -1 as dummy team ID
        
        # Pair teams for this round
        num_games = len(remaining_teams) // 2
        print(f"Round {current_round}: Simulating {num_games} games with {len(remaining_teams)} teams")
        
        for i in range(num_games):
            # In a typical bracket, we'd pair 1 vs 16, 2 vs 15, etc.
            # But for simplicity, we'll just pair sequentially from the remaining teams
            team1 = remaining_teams[i * 2]
            team2 = remaining_teams[i * 2 + 1]
            
            # Skip if either team is a dummy "bye" team
            if team1 == -1:
                winner = team2
                loser = team1
            elif team2 == -1:
                winner = team1
                loser = team2
            else:
                # Get win probability for this matchup
                try:
                    prob = win_probability_lookup(team1, team2)
                except Exception as e:
                    print(f"Error getting win probability for {team1} vs {team2}: {e}")
                    # Default to 50% if there's an error
                    prob = 0.5
                
                # Simulate the game
                import random
                if random.random() < prob:
                    winner = team1
                    loser = team2
                else:
                    winner = team2
                    loser = team1
            
            # If winner or loser is a dummy team, handle appropriately
            if winner != -1:
                # Only add real teams to next round
                next_round_teams.append(winner)
                
            # Only update real teams in the bracket
            if loser != -1:
                # Update bracket with game result
                bracket['teams'][loser]['eliminated'] = True
                bracket['teams'][loser]['eliminated_in_round'] = current_round
            
            # Record the game in the bracket (skip dummy teams)
            if team1 != -1 and team2 != -1:
                bracket['rounds'][current_round]['games'].append({
                    'team1': team1,
                    'team2': team2,
                    'winner': winner
                })
        
        # Mark this round as completed
        bracket['rounds'][current_round]['completed'] = True
        
        # Log number of teams advancing
        print(f"Round {current_round} complete. {len(next_round_teams)} teams advancing to next round.")
        
        # Update remaining teams for next round
        remaining_teams = next_round_teams
    
    # Safety check: ensure we have a champion
    if not remaining_teams:
        print("Warning: No champion found after all rounds. Selecting a random non-eliminated team.")
        # Find any team that's not eliminated
        non_eliminated = [team_id for team_id, info in bracket['teams'].items() 
                         if not info.get('eliminated', False)]
        
        if non_eliminated:
            champion = non_eliminated[0]
        else:
            # If all teams are somehow eliminated, just pick the first team
            champion = list(bracket['teams'].keys())[0]
            print(f"All teams were eliminated. Selected Team {champion} as champion.")
    else:
        # The last team standing is the champion
        champion = remaining_teams[0]
        print(f"Team {champion} is the champion!")
    
    return champion


def create_win_probability_lookup(submission_df, team_stats=None):
    """
    Create a function to look up win probabilities between teams.
    
    Args:
        submission_df: DataFrame with game predictions
        team_stats: Optional dictionary with team statistics
        
    Returns:
        Function that returns win probability for team1 vs team2
    """
    # Create lookup dictionary from submission data
    prob_dict = {}
    
    # Check the appropriate ID column
    id_column = 'original_ID' if 'original_ID' in submission_df.columns else 'ID'
    
    for _, row in submission_df.iterrows():
        id_parts = row[id_column].split('_')
        if len(id_parts) >= 3:
            season = id_parts[0]
            team1 = int(id_parts[1])
            team2 = int(id_parts[2])
            
            # Store in both directions for easier lookup
            key_forward = f"{season}_{team1}_{team2}"
            key_reverse = f"{season}_{team2}_{team1}"
            
            # If Team1 is sorted first in the ID, probability is directly from Pred
            # Otherwise, probability is 1 - Pred
            if team1 < team2:
                prob_dict[key_forward] = row['Pred']
                prob_dict[key_reverse] = 1 - row['Pred']
            else:
                prob_dict[key_forward] = 1 - row['Pred']
                prob_dict[key_reverse] = row['Pred']
    
    def get_win_probability(team1, team2):
        """
        Look up win probability for team1 vs team2.
        Returns probability that team1 wins.
        """
        # Handle dummy teams used for "byes"
        if team1 == -1:
            return 0.0  # Dummy team always loses
        if team2 == -1:
            return 1.0  # Opponent of dummy team always wins
            
        # Get current season (could be passed as a parameter if needed)
        current_season = "2025"  # hardcoded for simplicity, adapt as needed
        
        # Try to find a direct matchup in our predictions
        key = f"{current_season}_{team1}_{team2}"
        if key in prob_dict:
            return prob_dict[key]
        
        # Try the reverse matchup
        key = f"{current_season}_{team2}_{team1}"
        if key in prob_dict:
            return 1 - prob_dict[key]
        
        # If no direct matchup found, use team statistics to estimate
        # This is a simple fallback and could be improved
        if team_stats is not None:
            team1_key = f"{current_season}_{team1}"
            team2_key = f"{current_season}_{team2}"
            
            team1_win_rate = team_stats.get(team1_key, {}).get('win_rate', 0.5)
            team2_win_rate = team_stats.get(team2_key, {}).get('win_rate', 0.5)
            
            # Estimate based on relative win rates
            if team1_win_rate + team2_win_rate > 0:
                return team1_win_rate / (team1_win_rate + team2_win_rate)
        
        # Default to 50% if no better information available
        return 0.5
    
    return get_win_probability


def optimize_predictions(submission_df, simulation_results, weight=0.3):
    """
    Optimize predictions based on Monte Carlo simulation results.
    
    Args:
        submission_df: Original submission DataFrame with predictions
        simulation_results: Results from Monte Carlo simulation
        weight: Weight to give to simulation results (0-1)
        
    Returns:
        DataFrame with optimized predictions
    """
    print(f"Optimizing predictions with simulation weight: {weight}")
    
    # Create a copy to avoid modifying the original
    optimized_df = submission_df.copy()
    
    # Initialize dictionaries to store our simulation-based estimates for each matchup
    matchup_estimates = {}
    if 'matchup_probabilities' in simulation_results:
        matchup_estimates = simulation_results['matchup_probabilities']
    
    # Check if we need to access IDs differently due to original_ID column
    id_column = 'original_ID' if 'original_ID' in optimized_df.columns else 'ID'
    
    # Index for looking up the predictions
    pred_lookup = dict(zip(optimized_df[id_column], optimized_df['Pred']))
    
    # Track how many predictions were modified
    modified_count = 0
    
    # Process each row
    for i, row in optimized_df.iterrows():
        game_id = row[id_column]
        original_pred = row['Pred']
        
        # Get the teams from the game ID
        match = re.match(r'(\d+)_(\d+)_(\d+)', game_id)
        if match:
            season, team1, team2 = match.groups()
            team1, team2 = int(team1), int(team2)
            
            # Get round advancement probabilities for both teams
            team1_advancement = {}
            team2_advancement = {}
            
            if 'round_probabilities' in simulation_results:
                team1_advancement = simulation_results['round_probabilities'].get(team1, {})
                team2_advancement = simulation_results['round_probabilities'].get(team2, {})
            
            # Calculate a simulation-based estimate, if available
            sim_estimate = None
            if team1 in simulation_results.get('champion_probabilities', {}) and team2 in simulation_results.get('champion_probabilities', {}):
                # If we have champion probabilities, use them to inform our estimate
                team1_champ = simulation_results['champion_probabilities'][team1]
                team2_champ = simulation_results['champion_probabilities'][team2]
                
                if team1_champ + team2_champ > 0:
                    sim_estimate = team1_champ / (team1_champ + team2_champ)
            
            # If we didn't get an estimate from champion probabilities, try matchup probabilities
            if sim_estimate is None and f"{team1}_{team2}" in matchup_estimates:
                sim_estimate = matchup_estimates[f"{team1}_{team2}"]
            
            # If we have a simulation-based estimate, blend it with the original prediction
            if sim_estimate is not None:
                # Blend original prediction with simulation-based estimate
                optimized_pred = (1 - weight) * original_pred + weight * sim_estimate
                
                # Ensure probabilities are in valid range
                optimized_pred = max(0.01, min(0.99, optimized_pred))
                
                # Update the prediction
                optimized_df.at[i, 'Pred'] = optimized_pred
                
                if abs(optimized_pred - original_pred) > 0.01:  # Only count significant changes
                    modified_count += 1
    
    print(f"Modified {modified_count} predictions based on simulation results")
    
    return optimized_df


def simulate_and_optimize(submission_df, seed_dict, team_stats=None, num_simulations=10000, simulation_weight=0.3):
    """
    Main function to run the Monte Carlo simulation and optimize predictions.
    
    Args:
        submission_df: DataFrame with original predictions
        seed_dict: Dictionary mapping Season_TeamID to seed value
        team_stats: Optional dictionary with team statistics
        num_simulations: Number of simulations to run
        simulation_weight: Weight to give to simulation results (0-1)
        
    Returns:
        DataFrame with optimized predictions
    """
    print(f"Starting Monte Carlo simulation with {num_simulations} iterations")
    
    # Check simulation parameters
    if num_simulations < 1:
        print("Warning: Invalid number of simulations. Setting to 1000.")
        num_simulations = 1000
    
    if simulation_weight < 0 or simulation_weight > 1:
        print("Warning: Invalid simulation weight. Clamping to range 0-1.")
        simulation_weight = max(0, min(1, simulation_weight))
    
    # Parse tournament structure
    try:
        tournament_structure = parse_tournament_structure(submission_df)
    except Exception as e:
        print(f"Error parsing tournament structure: {e}")
        print("Returning original predictions without Monte Carlo optimization.")
        return submission_df
    
    # Create win probability lookup function
    try:
        win_prob_lookup = create_win_probability_lookup(submission_df, team_stats)
    except Exception as e:
        print(f"Error creating win probability function: {e}")
        print("Returning original predictions without Monte Carlo optimization.")
        return submission_df
    
    all_simulation_results = {}
    
    # Process each tournament season
    for season in tournament_structure:
        print(f"\nProcessing tournament for season {season}")
        
        # Check if we have enough teams
        if len(tournament_structure[season]['team_ids']) < 2:
            print(f"Not enough teams for season {season}. Skipping simulation.")
            continue
            
        # Build bracket structure
        try:
            bracket = build_bracket(tournament_structure, seed_dict, season)
        except Exception as e:
            print(f"Error building bracket for season {season}: {e}")
            print(f"Skipping simulation for season {season}.")
            continue
        
        # Run simulation
        try:
            simulation_results = run_tournament_simulation(
                bracket, 
                win_prob_lookup,
                num_simulations=num_simulations
            )
            
            all_simulation_results[season] = simulation_results
            
            # Print some simulation insights
            print("\nChampion Probabilities (Top 5):")
            top_champions = sorted(simulation_results['champion_probabilities'].items(), 
                                  key=lambda x: x[1], reverse=True)[:5]
            for team_id, prob in top_champions:
                seed = seed_dict.get(f"{season}_{team_id}", "Unknown")
                print(f"  Team {team_id} (Seed {seed}): {prob:.1%}")
            
            print("\nFinal Four Probabilities (Top 5):")
            top_final_four = sorted(simulation_results['final_four_probabilities'].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
            for team_id, prob in top_final_four:
                seed = seed_dict.get(f"{season}_{team_id}", "Unknown")
                print(f"  Team {team_id} (Seed {seed}): {prob:.1%}")
                
        except Exception as e:
            print(f"Error running simulation for season {season}: {e}")
            print(f"Skipping optimization for season {season}.")
    
    # Check if we have any simulation results
    if not all_simulation_results:
        print("No successful simulations completed. Returning original predictions.")
        return submission_df
    
    # Optimize predictions based on simulation results
    try:
        optimized_df = optimize_predictions(
            submission_df, 
            all_simulation_results.get("2025", {}),  # Assuming 2025 is the current season
            weight=simulation_weight
        )
        return optimized_df
    except Exception as e:
        print(f"Error optimizing predictions: {e}")
        print("Returning original predictions.")
        return submission_df