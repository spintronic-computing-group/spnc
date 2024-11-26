import optuna

'''
this function is used to control the termination of the study under certain conditions.

'''


def callback(targets):
    """
    设置多目标优化的callback函数
    
    Args:
        targets: 包含两个目标值的元组 (target1, target2)
    """
    def callback(study, trial):
        # 获取当前的Pareto前沿解
        pareto_front = study.best_trials

        # 检查是否有解同时满足两个目标
        for t in pareto_front:
            values = t.values  # 对于多目标优化，values是一个元组
            if values[0] <= targets[0] and values[1] <= targets[1]:
                print(f"找到满足目标的解: {values}")
                study.stop()
                return
            
    return callback

