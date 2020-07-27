import warnings
warnings.simplefilter('ignore')

from lib import best_arm_policies as bp
from lib.arms import NormalArm

# armのオブジェクト生成。
arm1 = NormalArm('arm1', 3.0, 1.0)  # X1 ~ N(μ1, σ1)
arm2 = NormalArm('arm2', 10.0, 1.0)  # X2 ~ N(μ2, σ2)
arm3 = NormalArm('arm3', 10.1, 1.0)  # X3 ~ N(μ3, σ3)
arm4 = NormalArm('arm4', 10.15, 1.0)  # X3 ~ N(μ4, σ4)

# アーム指定
target_arms = [arm1, arm2, arm3, arm4]

# 実行
if __name__ == "__main__":
    df_se = bp.successive_elimination_policy(target_arms, 0.04, 0.01, 10000)
    print(df_se.tail())
