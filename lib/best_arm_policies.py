# coding=utf-8
import numpy as np
import pandas as pd


# 逐次削除方策用の関数
def successive_elimination_policy(arms_: list,
                                  epsilon: float,
                                  delta: float,
                                  max_iter: int
                                  ):
    """
    # 入力
    arms_: armを格納したlist
    epsilon: 許容幅 ε >= 0
    delta: 誤識別率 δ > 0
    max_iter: 試行回数の上限

    # パラメータ
    beta: 信頼度 β(n, δ): N × (0,1) → (0, ∞)

    # 出力
    printで表示: 終了時の試行回数(n)、ε-最適腕(i*)、各アームの選択回数
    return: ε-最適腕(i*)、報酬の標本平均(mu)、ucb, lcbの推移

    # 補足
    試行回数はアームを引いた回数(t)ではなく、「全アームを引く」を繰り返した回数(n)を使用
    アーム数をKとおくと、以下の関係になる。
      t = K*n
    betaは以下の計算で求める(教科書p.105, 定理6.1より)
      log(4Kn^2/delta)
    """

    # 値の初期化
    arm_names = [arm_.__dict__['name'] for arm_ in arms_]  # アーム名の一覧, 出力のカラム名に使用
    arms = np.array(arms_)  # 後の処理のためにnumpy.arrayへ変換
    num_arms = len(arms)  # 総アーム数(初期値)
    n = 1

    best_arm_lst = []
    rw_mean_lst = []
    ucb_lst = []
    lcb_lst = []

    # ループスタート
    for n in range(1, max_iter+1):  # nは[1, max_iter]の範囲をとる。

        # 対象リストに含まれるすべてのアームを1回引く(無効化済みのアームはnp.nanを返す)
        if n == 1:
            reword_init = [arm_.draw() if arm_ is not None else np.nan for arm_ in arms]
            rewords = np.array(reword_init).reshape(1, num_arms)
        else:
            reword_latest = [arm_.draw() if arm_ is not None else np.nan for arm_ in arms]
            rewords = np.vstack([rewords, reword_latest])

        # 各アームiのUCB, LCBスコア(6.5)を計算
        # TODO: 無効化したアームの標本平均は明示的にnanにしといた方が良いか。
        rw_mean = rewords.mean(axis=0)  # 各アームの報酬の標本平均(無効化したアームはnp.nanが返る)
        arm_selected_cnt = np.sum(~np.isnan(rewords), axis=0)  # 各アームが選択された回数Ni(t)
        beta = np.log(4 * num_arms * n ** 2 / delta)
        ucb = rw_mean + np.sqrt(beta / (2 * n))
        lcb = rw_mean - np.sqrt(beta / (2 * n))

        # その時点で有効なアームのうち、標本平均最大のものを選択
        best_arm = np.nanargmax(rw_mean, axis=0)

        # 結果蓄積用のリストに結果を貯めていく
        best_arm_lst.append(arms[best_arm].__dict__['name'])
        rw_mean_lst.append(rw_mean)
        ucb_lst.append(ucb)
        lcb_lst.append(lcb)

        # ucb,lcbの最適腕のindexをnp.nanに置換したarrayを生成(終了判定に使用)
        ucb_oth = ucb.copy()
        ucb_oth[best_arm] = np.nan

        # 終了判定: 最適腕（最新）のLCB + εが他の有効な全てのアームのUCBより大きい場合、その腕を解として終了。
        # TODO: max_iterまでに終わらなかった場合の分岐をいれる
        if lcb[best_arm] + epsilon > np.nanmax(ucb_oth):
            print(f'{n}, {arms[best_arm].__dict__}')
            print(f'{arm_selected_cnt}')
            print('finish')
            break

        else:
            # アーム除外判定: 処理が継続される場合、UCBが最適腕（最新）のLCBよりも小さいアームを無効化する。
            arms[ucb < lcb[best_arm]] = None

            # 処理終了、nを+1して冒頭へ戻る
            n += 1

    # ループ終了後、ε-最適腕(i*)、報酬の標本平均(mu)、ucb, lcbの推移をDataFrameとして出力
    # TODO: リストではなくndarrayで貯めていった方が楽な気がしないでもない
    df_best_arm = pd.DataFrame(best_arm_lst, columns=['best_arm'])
    df_mu_sample = pd.DataFrame(rw_mean_lst, columns=[f'{i}_mean' for i in arm_names])
    df_ucb = pd.DataFrame(ucb_lst, columns=[f'{i}_ucb' for i in arm_names])
    df_lcb = pd.DataFrame(lcb_lst, columns=[f'{i}_lcb' for i in arm_names])
    df_result = pd.concat([df_best_arm, df_mu_sample, df_ucb, df_lcb], axis=1)

    return df_result


# LUCB方策用の関数
def lucb_policy(arms_: list,
                epsilon: float,
                delta: float,
                max_iter: int
                ):
    """
    # 入力
    arms_: armを格納したlist
    epsilon: 許容幅 ε >= 0
    delta: 誤識別率 δ > 0
    max_iter: 試行回数の上限

    # パラメータ
    beta: 信頼度 β(t, δ): N × (0,1) → (0, ∞)

    # 出力
    printで表示: 終了時の試行回数(t)、ε-最適腕(i*)、各アームの選択回数
    return: ε-最適腕(i*)、報酬の標本平均(mu)、ucb, lcbの推移

    # 補足
    betaは以下の計算で求める(教科書p.108, 定理6.2より)
    log(5Kt^4/4delta)
    """

    # 値の初期化
    arm_names = [arm_.__dict__['name'] for arm_ in arms_]  # アーム名の一覧, 出力のカラム名に使用
    arms = np.array(arms_)  # 後の処理のためにnumpy.arrayへ変換
    num_arms = len(arms)  # 総アーム数(初期値)
    t = 1

    best_arm_lst = []
    next_pull_arm_lst = []
    rw_mean_lst = []
    ucb_lst = []
    lcb_lst = []

    # 処理スタート
    # 対象リストに含まれるすべてのアームを1回引く
    reword_init = [arm_.draw() if arm_ is not None else np.nan for arm_ in arms]
    rewords = np.array(reword_init).reshape(1, num_arms)

    # ループスタート
    for t in range(1, max_iter+1):  # tは[1, max_iter]の範囲をとる。

        # 各アームiのUCB, LCBスコア(6.5)を計算
        rw_mean = np.nanmean(rewords, axis=0)  # 全アームの標本平均(前回引かれなかったアームについても平均を出力)
        arm_selected_cnt = np.sum(~np.isnan(rewords), axis=0)  # 各アームが選択された回数Ni(t)
        beta = np.log(5 * num_arms * t ** 4 / (4 * delta))  # betaの式は逐次削除と異なる(定理6.2)
        ucb = rw_mean + np.sqrt(beta / (2 * arm_selected_cnt))
        lcb = rw_mean - np.sqrt(beta / (2 * arm_selected_cnt))

        # 標本平均最大のアームを選択
        best_arm = np.nanargmax(rw_mean, axis=0)

        # ucb,lcbの最適腕のindexをnp.nanに置換したarrayを生成(終了判定に使用)
        ucb_oth = ucb.copy()
        ucb_oth[best_arm] = np.nan

        # 最適腕でないアームのうち、UCBが最大のものを選択
        next_pull_arm = np.nanargmax(ucb_oth)

        # 結果蓄積用のリストに結果を貯めていく
        best_arm_lst.append(arms[best_arm].__dict__['name'])
        next_pull_arm_lst.append(arms[next_pull_arm].__dict__['name'])
        rw_mean_lst.append(rw_mean)
        ucb_lst.append(ucb)
        lcb_lst.append(lcb)

        # 終了判定: 最適腕でないアームの最大UCBが、最適腕のLCB + ε より小さい場合、探索を終了。
        # TODO: max_iterまでに終わらなかった場合の分岐をいれる
        if ucb_oth[next_pull_arm] < (lcb[best_arm] + epsilon):
            print(f'{t}, {arms[best_arm].__dict__}')
            print(f'{arm_selected_cnt}')
            print('finish')
            break

        else:
            # 継続する場合は、i*(最適腕)、i**(最適腕でなく、最大UCBのアーム)をそれぞれ1回引く
            reword_latest = [arms[i].draw() if i in [best_arm, next_pull_arm] else np.nan for i in range(num_arms)]
            rewords = np.vstack([rewords, reword_latest])

            # 処理終了、tを+2(2回引いているため)して冒頭へ戻る
            t += 2

    # ループ終了後、ε-最適腕(i*)、報酬の標本平均(mu)、ucb, lcbの推移をDataFrameとして出力
    # TODO: リストではなくndarrayで貯めていった方が楽な気がしないでもない
    # TODO: 試行回数tの列を別途作る必要がある。(indexはtの1/2になっているため)
    df_best_arm = pd.DataFrame(best_arm_lst, columns=['best_arm'])
    df_next_pull_arm = pd.DataFrame(next_pull_arm_lst, columns=['next_pull_arm'])
    df_mu_sample = pd.DataFrame(rw_mean_lst, columns=[f'{i}_mean' for i in arm_names])
    df_ucb = pd.DataFrame(ucb_lst, columns=[f'{i}_ucb' for i in arm_names])
    df_lcb = pd.DataFrame(lcb_lst, columns=[f'{i}_lcb' for i in arm_names])
    df_result = pd.concat([df_best_arm, df_next_pull_arm, df_mu_sample, df_ucb, df_lcb], axis=1)

    return df_result
