import random


# 参考リポジトリの処理を流用(arms/normal.py。正規乱数を返す処理→アームとして使用)
class NormalArm():
    def __init__(self, name, mu, sigma):
        self.name = name  # 最終結果を返す時のために、armの名前を設定できるようにしておく。
        self.mu = mu
        self.sigma = sigma

    def draw(self):
        return random.gauss(self.mu, self.sigma)
