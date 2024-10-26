import subprocess

def run_dashboard():
    """启动Optuna仪表盘,便于实时监控优化过程"""
    subprocess.run(["optuna-dashboard", "sqlite:///db.sqlite3"])