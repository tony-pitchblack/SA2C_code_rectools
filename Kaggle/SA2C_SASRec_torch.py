import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from SASRecModules_torch import SASRecQNetworkTorch


def parse_args():
    parser = argparse.ArgumentParser(description="Run nive double q learning.")

    parser.add_argument("--epoch", type=int, default=50, help="Number of max epochs.")
    parser.add_argument("--data", nargs="?", default="data", help="data directory")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument(
        "--hidden_factor",
        type=int,
        default=64,
        help="Number of hidden factors, i.e., embedding size.",
    )
    parser.add_argument("--r_click", type=float, default=0.2, help="reward for the click behavior.")
    parser.add_argument("--r_buy", type=float, default=1.0, help="reward for the purchase behavior.")
    parser.add_argument("--r_negative", type=float, default=-0.0, help="reward for the negative behavior.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--discount", type=float, default=0.5, help="Discount factor for RL.")
    parser.add_argument("--neg", type=int, default=10, help="number of negative samples.")
    parser.add_argument("--weight", type=float, default=1.0, help="number of negative samples.")
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="smooth factor for off-policy correction,smooth=0 equals no correction",
    )
    parser.add_argument("--clip", type=float, default=0.0, help="clip value for advantage")
    parser.add_argument("--lr_2", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--max_steps", type=int, default=0, help="If > 0, stop after this many update steps.")
    parser.add_argument(
        "--model",
        type=str,
        default="NItNet",
        help="the base recommendation models, including GRU,Caser,NItNet and SASRec",
    )
    parser.add_argument(
        "--num_filters",
        type=int,
        default=16,
        help="Number of filters per filter size (default: 16) (for Caser)",
    )
    parser.add_argument("--filter_sizes", nargs="?", default="[2,3,4]", help="Specify the filter_size (for Caser)")
    parser.add_argument("--num_heads", default=1, type=int, help="number heads (for SASRec)")
    parser.add_argument("--num_blocks", default=1, type=int, help="number heads (for SASRec)")
    parser.add_argument("--dropout_rate", default=0.1, type=float)

    return parser.parse_args()


def pad_history(itemlist, length, pad_item):
    if len(itemlist) >= length:
        return itemlist[-length:]
    if len(itemlist) < length:
        return itemlist + [pad_item] * (length - len(itemlist))
    return itemlist


def calculate_hit(
    sorted_list,
    topk,
    true_items,
    rewards,
    r_click,
    total_reward,
    hit_click,
    ndcg_click,
    hit_purchase,
    ndcg_purchase,
):
    true_items = np.asarray(true_items)
    rewards = np.asarray(rewards)
    for i, k in enumerate(topk):
        rec_list = sorted_list[:, -k:]
        hits = (rec_list == true_items[:, None]).any(axis=1)
        hit_idx = np.where(hits)[0]
        if hit_idx.size == 0:
            continue
        pos = rec_list[hit_idx] == true_items[hit_idx, None]
        rank = k - pos.argmax(axis=1)
        total_reward[i] += rewards[hit_idx].sum()
        is_click = rewards[hit_idx] == r_click
        is_purchase = ~is_click
        if is_click.any():
            hit_click[i] += float(is_click.sum())
            ndcg_click[i] += float((1.0 / np.log2(rank[is_click] + 1)).sum())
        if is_purchase.any():
            hit_purchase[i] += float(is_purchase.sum())
            ndcg_purchase[i] += float((1.0 / np.log2(rank[is_purchase] + 1)).sum())


def calculate_off(
    sorted_list,
    true_items,
    rewards,
    reward_click,
    off_click_ng,
    off_purchase_ng,
    off_prob_click,
    off_prob_purchase,
    pop_dict,
    topk=10,
):
    true_items = np.asarray(true_items)
    rewards = np.asarray(rewards)
    rec_list = sorted_list[:, -topk:]
    for j in range(len(true_items)):
        prob = float(pop_dict[true_items[j]])
        if rewards[j] == reward_click:
            off_prob_click[0] += 1.0 / prob
        else:
            off_prob_purchase[0] += 1.0 / prob
        if true_items[j] in rec_list[j]:
            rank = topk - int(np.argwhere(rec_list[j] == true_items[j])[0][0])
            if rewards[j] == reward_click:
                off_click_ng[0] += (1.0 / np.log2(rank + 1)) / prob
            else:
                off_purchase_ng[0] += (1.0 / np.log2(rank + 1)) / prob


def _sample_negative_actions(item_num, actions, neg, device):
    bsz = actions.shape[0]
    neg_actions = torch.randint(0, item_num, size=(bsz, neg), device=device)
    bad = neg_actions.eq(actions[:, None])
    while bad.any():
        neg_actions[bad] = torch.randint(0, item_num, size=(int(bad.sum().item()),), device=device)
        bad = neg_actions.eq(actions[:, None])
    return neg_actions


@torch.no_grad()
def evaluate(model, data_directory, state_size, item_num, reward_click, reward_buy, pop_dict, device):
    eval_sessions = pd.read_pickle(os.path.join(data_directory, "sampled_val.df"))
    eval_ids = eval_sessions.session_id.unique()
    groups = eval_sessions.groupby("session_id")

    batch_sessions = 100
    evaluated = 0
    total_clicks = 0.0
    total_purchase = 0.0
    topk = [5, 10, 15, 20]

    total_reward = [0.0, 0.0, 0.0, 0.0]
    hit_clicks = [0.0, 0.0, 0.0, 0.0]
    ndcg_clicks = [0.0, 0.0, 0.0, 0.0]
    hit_purchase = [0.0, 0.0, 0.0, 0.0]
    ndcg_purchase = [0.0, 0.0, 0.0, 0.0]

    off_prob_click = [0.0]
    off_prob_purchase = [0.0]
    off_click_ng = [0.0]
    off_purchase_ng = [0.0]

    model.eval()
    while evaluated < len(eval_ids):
        states, len_states, actions, rewards = [], [], [], []
        for _ in range(batch_sessions):
            if evaluated == len(eval_ids):
                break
            sid = eval_ids[evaluated]
            group = groups.get_group(sid)
            history = []
            for _, row in group.iterrows():
                state = list(history)
                len_states.append(state_size if len(state) >= state_size else 1 if len(state) == 0 else len(state))
                states.append(pad_history(state, state_size, item_num))
                action = int(row["item_id"])
                is_buy = int(row["is_buy"])
                reward = reward_buy if is_buy == 1 else reward_click
                if is_buy == 1:
                    total_purchase += 1.0
                else:
                    total_clicks += 1.0
                actions.append(action)
                rewards.append(reward)
                history.append(action)
            evaluated += 1

        inputs = torch.as_tensor(np.asarray(states, dtype=np.int64), device=device)
        lens = torch.as_tensor(np.asarray(len_states, dtype=np.int64), device=device)
        _, ce_logits = model(inputs, lens)
        prediction = ce_logits.detach().cpu().numpy()

        sorted_list = np.argsort(prediction, axis=1)
        calculate_hit(
            sorted_list,
            topk,
            actions,
            rewards,
            reward_click,
            total_reward,
            hit_clicks,
            ndcg_clicks,
            hit_purchase,
            ndcg_purchase,
        )
        calculate_off(
            sorted_list,
            actions,
            rewards,
            reward_click,
            off_click_ng,
            off_purchase_ng,
            off_prob_click,
            off_prob_purchase,
            pop_dict,
        )

    off_click_ng_val = off_click_ng[0] / off_prob_click[0] if off_prob_click[0] > 0 else 0.0
    off_purchase_ng_val = off_purchase_ng[0] / off_prob_purchase[0] if off_prob_purchase[0] > 0 else 0.0

    print("#############################################################")
    print(f"total clicks: {int(total_clicks)}, total purchase:{int(total_purchase)}")
    for i, k in enumerate(topk):
        hr_click = hit_clicks[i] / total_clicks if total_clicks > 0 else 0.0
        hr_purchase = hit_purchase[i] / total_purchase if total_purchase > 0 else 0.0
        ng_click = ndcg_clicks[i] / total_clicks if total_clicks > 0 else 0.0
        ng_purchase = ndcg_purchase[i] / total_purchase if total_purchase > 0 else 0.0
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"cumulative reward @ {k}: {total_reward[i]:f}")
        print(f"clicks hr ndcg @ {k} : {hr_click:f}, {ng_click:f}")
        print(f"purchase hr and ndcg @{k} : {hr_purchase:f}, {ng_purchase:f}")
    print(f"off-line corrected evaluation (click_ng,purchase_ng)@10: {off_click_ng_val:f}, {off_purchase_ng_val:f}")
    print("#############################################################")


def main():
    args = parse_args()
    if args.model != "SASRec":
        raise ValueError("This torch script only supports --model SASRec")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_directory = args.data
    data_statis = pd.read_pickle(os.path.join(data_directory, "data_statis.df"))
    state_size = int(data_statis["state_size"][0])
    item_num = int(data_statis["item_num"][0])
    reward_click = float(args.r_click)
    reward_buy = float(args.r_buy)
    reward_negative = float(args.r_negative)

    replay_buffer = pd.read_pickle(os.path.join(data_directory, "replay_buffer.df"))
    with open(os.path.join(data_directory, "pop_dict.txt"), "r") as f:
        pop_dict = eval(f.read())

    qn1 = SASRecQNetworkTorch(
        item_num=item_num,
        state_size=state_size,
        hidden_size=args.hidden_factor,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout_rate=args.dropout_rate,
    ).to(device)
    qn2 = SASRecQNetworkTorch(
        item_num=item_num,
        state_size=state_size,
        hidden_size=args.hidden_factor,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        dropout_rate=args.dropout_rate,
    ).to(device)

    opt1_qn1 = torch.optim.Adam(qn1.parameters(), lr=args.lr)
    opt2_qn1 = torch.optim.Adam(qn1.parameters(), lr=args.lr_2)
    opt1_qn2 = torch.optim.Adam(qn2.parameters(), lr=args.lr)
    opt2_qn2 = torch.optim.Adam(qn2.parameters(), lr=args.lr_2)

    total_step = 0
    num_rows = replay_buffer.shape[0]
    num_batches = int(num_rows / args.batch_size)

    for _ in range(args.epoch):
        for _ in range(num_batches):
            if args.max_steps > 0 and total_step >= args.max_steps:
                return
            batch = replay_buffer.sample(n=args.batch_size)

            state = np.asarray(list(batch["state"].values), dtype=np.int64)
            len_state = np.asarray(list(batch["len_state"].values), dtype=np.int64)
            next_state = np.asarray(list(batch["next_state"].values), dtype=np.int64)
            len_next_state = np.asarray(list(batch["len_next_states"].values), dtype=np.int64)
            action = np.asarray(list(batch["action"].values), dtype=np.int64)
            is_buy = np.asarray(list(batch["is_buy"].values), dtype=np.int64)
            is_done = np.asarray(list(batch["is_done"].values), dtype=np.bool_)

            reward = np.where(is_buy == 1, reward_buy, reward_click).astype(np.float32)
            discount = np.full(shape=(action.shape[0],), fill_value=float(args.discount), dtype=np.float32)

            pointer = np.random.randint(0, 2)
            if pointer == 0:
                main_qn, target_qn = qn1, qn2
                opt1, opt2 = opt1_qn1, opt2_qn1
            else:
                main_qn, target_qn = qn2, qn1
                opt1, opt2 = opt1_qn2, opt2_qn2

            main_qn.train()
            target_qn.train()

            state_t = torch.as_tensor(state, device=device)
            len_state_t = torch.as_tensor(len_state, device=device)
            next_state_t = torch.as_tensor(next_state, device=device)
            len_next_state_t = torch.as_tensor(len_next_state, device=device)
            action_t = torch.as_tensor(action, device=device, dtype=torch.long)
            reward_t = torch.as_tensor(reward, device=device)
            discount_t = torch.as_tensor(discount, device=device)
            done_t = torch.as_tensor(is_done, device=device, dtype=torch.float32)

            neg_actions_t = _sample_negative_actions(item_num, action_t, args.neg, device=device)

            with torch.no_grad():
                q_next_target, _ = target_qn(next_state_t, len_next_state_t)
                q_next_selector, _ = main_qn(next_state_t, len_next_state_t)
                q_curr_target, _ = target_qn(state_t, len_state_t)
                q_curr_selector, _ = main_qn(state_t, len_state_t)

            q_values, ce_logits = main_qn(state_t, len_state_t)

            a_star = q_next_selector.argmax(dim=1)
            q_tp1 = q_next_target.gather(1, a_star[:, None]).squeeze(1)
            target_pos = reward_t + discount_t * q_tp1 * (1.0 - done_t)
            q_sa = q_values.gather(1, action_t[:, None]).squeeze(1)
            qloss_pos = ((q_sa - target_pos.detach()) ** 2).mean()

            a_star_curr = q_curr_selector.argmax(dim=1)
            q_t_star = q_curr_target.gather(1, a_star_curr[:, None]).squeeze(1)
            target_neg = reward_negative + discount_t * q_t_star
            q_sneg = q_values.gather(1, neg_actions_t)
            qloss_neg = ((q_sneg - target_neg.detach()[:, None]) ** 2).sum(dim=1).mean()

            ce_loss_pre = F.cross_entropy(ce_logits, action_t, reduction="none")

            if total_step < 15000:
                loss = qloss_pos + qloss_neg + ce_loss_pre.mean()
                opt1.zero_grad(set_to_none=True)
                loss.backward()
                opt1.step()
                total_step += 1
                if total_step % 4000 == 0:
                    evaluate(main_qn, data_directory, state_size, item_num, reward_click, reward_buy, pop_dict, device)
            else:
                with torch.no_grad():
                    prob = F.softmax(ce_logits, dim=1).gather(1, action_t[:, None]).squeeze(1)
                behavior_prob = torch.as_tensor([float(pop_dict[int(a)]) for a in action], device=device)
                ips = (prob / behavior_prob).clamp(0.1, 10.0).pow(float(args.smooth))

                with torch.no_grad():
                    q_pos_det = q_values.gather(1, action_t[:, None]).squeeze(1)
                    q_neg_det = q_values.gather(1, neg_actions_t).sum(dim=1)
                    q_avg = (q_pos_det + q_neg_det) / float(1 + args.neg)
                    advantage = q_pos_det - q_avg
                    if float(args.clip) > 0:
                        advantage = advantage.clamp(-float(args.clip), float(args.clip))

                ce_loss_post = ips * ce_loss_pre * advantage
                loss = float(args.weight) * (qloss_pos + qloss_neg) + ce_loss_post.mean()
                opt2.zero_grad(set_to_none=True)
                loss.backward()
                opt2.step()
                total_step += 1
                if total_step % 4000 == 0:
                    evaluate(main_qn, data_directory, state_size, item_num, reward_click, reward_buy, pop_dict, device)


if __name__ == "__main__":
    main()


