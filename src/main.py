import os
import argparse
import data_loader
import feature_engineering
import models
import evaluation
import numpy as np
from sklearn.metrics import brier_score_loss
import pandas as pd

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='March Mania Prediction')
    parser.add_argument('--data_path', type=str, default="Z:\\kaggle\\MMLM2025\\March-Machine-Learning-Mania-2025\\input\\march-machine-learning-mania-2025",
                        help='数据目录的路径')
    parser.add_argument('--start_year', type=int, default=2021,
                        help='只包含从该年份开始的赛季数据')
    parser.add_argument('--output_file', type=str, default="submission.csv",
                        help='预测结果输出文件名')
    parser.add_argument('--verbose', type=int, default=1,
                        help='详细程度级别 (0=最小, 1=正常)')
    parser.add_argument('--stage', type=int, default=2,
                        help='比赛阶段 (1 或 2)')
    parser.add_argument('--test_mode', action='store_true',
                        help='启用测试模式，每年仅使用10场比赛进行快速测试')
    parser.add_argument('--simulation_mode', action='store_true',
                        help='启用模拟模式，在2021-2023数据 + 2024常规赛数据上训练，在2024锦标赛数据上评估')
    parser.add_argument('--use_extended_models', action='store_true',
                        help='使用扩展模型集，包含更多模型变体以增加多样性')
    # 新增蒙特卡罗模拟参数
    parser.add_argument('--use_monte_carlo', action='store_true',
                        help='启用蒙特卡罗模拟以优化预测')
    parser.add_argument('--num_simulations', type=int, default=10000,
                        help='要运行的蒙特卡罗模拟次数')
    parser.add_argument('--simulation_weight', type=float, default=0.3,
                        help='给予模拟结果的权重 (0-1)')
    return parser.parse_args()

def run_pipeline(data_path, start_year=2021, output_file="submission.csv", verbose=1, stage=2, test_mode=False, 
                simulation_mode=False, use_extended_models=False, use_monte_carlo=False, num_simulations=10000, 
                simulation_weight=0.3):
    """
    运行完整的March Mania预测流程

    参数:
        data_path: 数据目录路径
        start_year: 仅包含从该年份开始的赛季
        output_file: 预测输出文件名
        verbose: 详细程度级别 (0=最小, 1=正常)
        stage: 比赛阶段 (1 或 2)
        test_mode: 如为True，每年仅使用10场比赛进行快速测试
        simulation_mode: 如为True，在2021-2023数据 + 2024常规赛数据上训练，在2024锦标赛数据上评估
        use_extended_models: 如为True，使用扩展模型集，包含更多模型变体以增加多样性
        use_monte_carlo: 如为True，使用蒙特卡罗模拟优化预测
        num_simulations: 要运行的蒙特卡罗模拟次数
        simulation_weight: 给予模拟结果的权重 (0-1)
    """
    print(f"从{start_year}年开始运行March Mania预测流程，用于第{stage}阶段")
    if test_mode:
        print("测试模式已启用：每年仅使用10场比赛进行快速测试")
        print("这将显著减少数据集大小并加速流程")
        print("注意：结果将不太准确，但对调试完整流程很有用")
    
    if simulation_mode:
        print("模拟模式已启用：在2021-2023数据 + 2024常规赛数据上训练")
        print("将在2024锦标赛数据上评估模型性能")
    
    if use_extended_models:
        print("扩展模型已启用：使用具有多样超参数的更大模型集")
        print("这将显著增加训练时间但可能提高预测准确性")
    
    if use_monte_carlo:
        print("蒙特卡罗模拟已启用：将使用锦标赛模拟优化预测")
        print(f"模拟次数：{num_simulations}，模拟权重：{simulation_weight}")
    
    # 第1步：加载数据
    season_detail, tourney_detail, seeds, teams, submission = data_loader.load_data(data_path, start_year, stage, test_mode)
    
    # 加载合并的KenPom数据
    merged_kenpom_df = data_loader.load_merged_kenpom_data(data_path, teams)
    
    # 为原始KenPom数据创建空DataFrame，因为它已被禁用
    kenpom_df = pd.DataFrame()
    print("原始KenPom数据加载已禁用，改用merged_kenpom.csv。")
    
    # 第2步：合并并准备比赛数据
    games = data_loader.merge_and_prepare_games(season_detail, tourney_detail)
    
    # 第3步：创建种子字典
    seed_dict = data_loader.prepare_seed_dict(seeds)
    
    # 第4步：基础特征工程
    print("执行特征工程...")
    games = feature_engineering.feature_engineering(games, seed_dict)
    
    # 第5步：添加团队统计特征（累积，避免目标泄漏）
    games, team_stats_cum = feature_engineering.add_team_features(games)
    
    # 第6步：添加高级特征：对抗历史和最近表现
    games = feature_engineering.add_head_to_head_features(games)
    games = feature_engineering.add_recent_performance_features(games, window=5)
    
    # 第6.5步：添加新特征：Elo评级、赛程强度和关键统计差异
    print("添加高级预测特征...")
    games = feature_engineering.add_elo_ratings(games, k_factor=20, initial_elo=1500, reset_each_season=True)
    games = feature_engineering.add_strength_of_schedule(games, team_stats_cum)
    games = feature_engineering.add_key_stat_differentials(games)
    games = feature_engineering.enhance_key_stat_differentials(games)
    games = feature_engineering.add_historical_tournament_performance(games, seed_dict)
    
    # --- 将KenPom特征合并到训练数据中 ---
    if not merged_kenpom_df.empty:
        print("添加合并的KenPom特征...")
        games = feature_engineering.merge_merged_kenpom_features(games, merged_kenpom_df)
    else:
        print("没有可用的合并KenPom数据。")
    

    
    # 第7步：聚合特征
    agg_features = feature_engineering.aggregate_features(games)
    
    # --- 根据模式筛选训练数据 ---
    current_season = games['Season'].max()
    print(f"当前赛季: {current_season}")
    
    # 在模拟模式下，为评估保留2024锦标赛数据
    eval_data = None
    if simulation_mode and current_season >= 2024:
        print("提取2024锦标赛数据用于评估...")
        eval_data = games[(games['Season'] == 2024) & (games['GameType'] == 'Tournament')].copy()
        print(f"评估数据大小: {len(eval_data)} 场比赛")
    
    # 筛选训练数据
    if simulation_mode:
        # 模拟模式：在2021-2023所有数据 + 2024常规赛数据上训练
        games = games[
            ((games['Season'] < 2024)) |
            ((games['Season'] == 2024) & (games['GameType'] == 'Regular'))
        ]
    else:
        # 正常模式：使用当前赛季常规赛和之前赛季锦标赛数据
        games = games[
            ((games['Season'] == current_season) & (games['GameType'] == 'Regular')) |
            ((games['Season'] < current_season) & (games['GameType'].isin(['Regular', 'Tournament'])))
        ]
    
    print(f"筛选后的训练数据行数: {len(games)}")
    
    # 第8步：准备提交特征（提交比赛保持不变）
    submission_df = feature_engineering.prepare_submission_features(
        submission, seed_dict, team_stats_cum, data_loader.extract_game_info
    )
    
    # 处理前验证提交数据完整性
    if 'original_index' in submission_df.columns:
        expected_rows = len(submission_df['original_index'].unique())
        actual_rows = len(submission_df)
        if expected_rows != actual_rows:
            print(f"警告：检测到提交数据完整性问题！")
            print(f"预期 {expected_rows} 个唯一行，但发现 {actual_rows} 个总行")
            print("这表明有重复或缺失行，这将导致错位")
    
    # 将相同的新特征应用到提交数据
    print("向提交数据添加高级预测特征...")
    submission_df = feature_engineering.add_elo_ratings(submission_df, k_factor=20, initial_elo=1500, reset_each_season=True)
    submission_df = feature_engineering.add_strength_of_schedule(submission_df, team_stats_cum)
    # 注意：关键统计差异需要比赛详情，这些可能不适用于未来比赛
    # 但如果它们在提交数据中，就会被使用
    submission_df = feature_engineering.add_key_stat_differentials(submission_df)
    submission_df = feature_engineering.enhance_key_stat_differentials(submission_df)
    submission_df = feature_engineering.add_historical_tournament_performance(submission_df, seed_dict)
    
    # 尝试将KenPom特征添加到提交数据
    if not merged_kenpom_df.empty:
        print("向提交数据添加合并的KenPom特征...")
        submission_df = feature_engineering.merge_merged_kenpom_features(submission_df, merged_kenpom_df)
    else:
        print("提交数据没有可用的合并KenPom数据。")
    
    # 如果在模拟模式下，将新特征应用到评估数据
    if simulation_mode and eval_data is not None and not eval_data.empty:
        print("向评估数据添加高级预测特征...")
        eval_data = feature_engineering.add_elo_ratings(eval_data, k_factor=20, initial_elo=1500, reset_each_season=True)
        eval_data = feature_engineering.add_strength_of_schedule(eval_data, team_stats_cum)
        eval_data = feature_engineering.add_key_stat_differentials(eval_data)
        eval_data = feature_engineering.enhance_key_stat_differentials(eval_data)
        eval_data = feature_engineering.add_historical_tournament_performance(eval_data, seed_dict)
        
        # 尝试将KenPom特征添加到评估数据
        if not merged_kenpom_df.empty and simulation_mode:
            print("向评估数据添加合并的KenPom特征...")
            eval_data = feature_engineering.merge_merged_kenpom_features(eval_data, merged_kenpom_df)
        

    
    # 在所有特征工程后验证提交数据完整性
    if 'original_index' in submission_df.columns:
        expected_rows = len(submission_df['original_index'].unique())
        actual_rows = len(submission_df)
        if expected_rows != actual_rows:
            print(f"警告：处理后检测到提交数据完整性问题！")
            print(f"预期 {expected_rows} 个唯一行，但发现 {actual_rows} 个总行")
            print("这表明有重复或缺失行，可能会导致错位")
            # 在继续前试图修复这个问题
            print("尝试通过删除可能的重复行来修复...")
            # 保持原始索引顺序
            submission_df = submission_df.drop_duplicates(subset=['original_index'], keep='first')
            # 重新索引以确保顺序正确
            submission_df = submission_df.sort_values('original_index').reset_index(drop=True)
            print(f"修复后的行数: {len(submission_df)}")
    
    # 第9步：准备训练和评估数据
    # 这里添加新的特征选择或工程步骤
    combined_features = feature_engineering.select_best_features(games, is_tournament=False)
    print(f"选定 {len(combined_features)} 个特征用于模型训练")
    
    X_train = games[combined_features].copy()
    y_train = games['WTeamWins'].copy()
    
    # 如果可用，获取提交数据的相同特征
    X_submission = submission_df[combined_features].copy()
    
    if verbose >= 1:
        print(f"训练数据形状: {X_train.shape}")
        print(f"提交数据形状: {X_submission.shape}")
    
    # 显示训练集中缺失值的百分比
    missing_train = (X_train.isnull().sum() / len(X_train)) * 100
    missing_train = missing_train[missing_train > 0].sort_values(ascending=False)
    if len(missing_train) > 0 and verbose >= 1:
        print("\n训练集中缺失值的特征（百分比）:")
        for feature, pct in missing_train.items():
            print(f"{feature}: {pct:.2f}%")
    
    # 第10步：处理缺失值
    X_train = feature_engineering.handle_missing_values(X_train)
    X_submission = feature_engineering.handle_missing_values(X_submission)
    
    # 将同样的缺失值处理应用到评估数据（如果在模拟模式下）
    if simulation_mode and eval_data is not None and not eval_data.empty:
        X_eval = eval_data[combined_features].copy()
        y_eval = eval_data['WTeamWins'].copy()
        X_eval = feature_engineering.handle_missing_values(X_eval)
        print(f"评估数据形状: {X_eval.shape}")
        
        # 第11.5步：如果处于模拟模式，评估模型性能
        evaluate_model(X_train, y_train, X_eval, y_eval, combined_features, use_extended_models)
    
    # 第11步：训练模型并生成预测
    # 使用模型集合进行训练
    print("训练模型集合...")
    
    # 选择最佳阈值（如果不处于模拟模式）
    best_threshold = 0.5  # 默认值
    if not simulation_mode:
        best_threshold = find_best_threshold(X_train, y_train, combined_features, use_extended_models)
        print(f"使用最佳阈值: {best_threshold:.4f}")
    
    # 根据配置选择基本或扩展模型集
    if use_extended_models:
        print("使用扩展模型集...")
        models_list = models.get_extended_models()
    else:
        models_list = models.get_base_models()
    
    # 训练所有模型
    trained_models = models.train_all_models(X_train, y_train, models_list, verbose=verbose)
    
    # 生成预测
    predictions = models.predict_with_all_models(trained_models, X_submission, verbose=verbose)
    
    # 第12步：如果启用，使用蒙特卡洛模拟优化
    if use_monte_carlo:
        print(f"运行 {num_simulations} 次蒙特卡洛模拟...")
        import monte_carlo
        
        # 对提交数据使用未调整的模型预测作为基准
        base_predictions = predictions.copy()
        
        # 执行蒙特卡洛模拟
        simulation_predictions = monte_carlo.run_monte_carlo_simulations(
            submission, seed_dict, trained_models, X_submission, 
            num_simulations=num_simulations,
            verbose=verbose
        )
        
        # 融合基本预测和模拟预测
        for i in range(len(predictions)):
            predictions[i] = (1 - simulation_weight) * base_predictions[i] + simulation_weight * simulation_predictions[i]
        
        print(f"已完成蒙特卡洛模拟，使用 {simulation_weight:.2f} 权重融合")
    
    # 修复边缘情况：确保预测在[0.01, 0.99]范围内
    predictions = np.clip(predictions, 0.01, 0.99)
    
    # 准备最终提交
    submission_output = pd.read_csv(os.path.join(data_path, "SampleSubmissionStage2.csv" if stage == 2 else "SampleSubmissionStage1.csv"))
    submission_output['Pred'] = predictions
    
    # 保存预测
    submission_output.to_csv(output_file, index=False)
    print(f"预测已保存至 {output_file}")
    
    return submission_output

def evaluate_model(X_train, y_train, X_eval, y_eval, features, use_extended_models=False):
    """
    在真实评估数据上评估模型性能
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        X_eval: 评估特征
        y_eval: 评估标签
        features: 特征列表
        use_extended_models: 是否使用扩展模型集
    """
    print("\n在2024锦标赛数据上进行模型评估...")
    
    # 根据配置选择基本或扩展模型集
    if use_extended_models:
        print("使用扩展模型集进行评估...")
        models_list = models.get_extended_models()
    else:
        models_list = models.get_base_models()
    
    # 训练所有模型
    trained_models = models.train_all_models(X_train, y_train, models_list, verbose=1)
    
    # 在评估集上进行预测
    eval_predictions = models.predict_with_all_models(trained_models, X_eval, verbose=1)
    
    # 评估指标：准确度和对数损失
    accuracy = (eval_predictions > 0.5).astype(int) == y_eval
    accuracy_pct = accuracy.mean() * 100
    
    # 计算对数损失
    from sklearn.metrics import log_loss
    log_loss_score = log_loss(y_eval, eval_predictions)
    
    # 计算Brier分数
    brier_score = brier_score_loss(y_eval, eval_predictions)
    
    print(f"评估集大小: {len(X_eval)} 场比赛")
    print(f"准确度: {accuracy_pct:.2f}%")
    print(f"对数损失: {log_loss_score:.4f}")
    print(f"Brier分数: {brier_score:.4f}")
    print("注意：锦标赛预测通常比常规赛预测更具挑战性")

def find_best_threshold(X_train, y_train, features, use_extended_models=False):
    """
    找到将概率转换为二元预测的最佳阈值
    
    参数:
        X_train: 训练特征
        y_train: 训练标签
        features: 特征列表
        use_extended_models: 是否使用扩展模型集
        
    返回:
        最佳阈值值
    """
    from sklearn.model_selection import StratifiedKFold
    
    print("寻找最佳预测阈值...")
    
    # 根据配置选择基本或扩展模型集
    if use_extended_models:
        models_list = models.get_extended_models()
    else:
        models_list = models.get_base_models()
    
    # 使用交叉验证找到最佳阈值
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = []
    all_targets = []
    
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # 训练所有模型
        trained_models = models.train_all_models(X_train_fold, y_train_fold, models_list, verbose=0)
        
        # 获取验证集预测
        val_preds = models.predict_with_all_models(trained_models, X_val_fold, verbose=0)
        
        all_preds.extend(val_preds)
        all_targets.extend(y_val_fold)
    
    # 将列表转换为数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 尝试不同的阈值
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_accuracy = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        accuracy = np.mean((all_preds > threshold) == all_targets)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    print(f"最佳阈值: {best_threshold:.4f}，交叉验证准确度: {best_accuracy*100:.2f}%")
    return best_threshold

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        args.data_path,
        args.start_year,
        args.output_file,
        args.verbose,
        args.stage,
        args.test_mode,
        args.simulation_mode,
        args.use_extended_models,
        args.use_monte_carlo,
        args.num_simulations,
        args.simulation_weight
    )

# 方案1：
# python main.py --use_monte_carlo --num_simulations=20000 --simulation_weight=0.4