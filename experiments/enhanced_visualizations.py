import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def infer_dataset(file_path, cfg):
    v = cfg.get('dataset_type')
    if v == 'h':
        return 'HotpotQA', 'hotpot'
    if v == 't':
        return 'TriviaQA', 'triviaqa'
    if v == 's':
        return 'SQuAD', 'squad'
    name = Path(file_path).name
    if name.startswith('hotpot_'):
        return 'HotpotQA', 'hotpot'
    if name.startswith('squad_'):
        return 'SQuAD', 'squad'
    if name.startswith('triviaqa_'):
        return 'TriviaQA', 'triviaqa'
    return 'Unknown', 'unknown'

def load_results(results_dir):
    items = []
    for p in Path(results_dir).glob('*.json'):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        cfg = data.get('config', {})
        dataset_name, dataset_prefix = infer_dataset(p, cfg)
        model = cfg.get('model', 'unknown')
        react_rate = data.get('react_success_rate')
        lats_rate = data.get('lats_success_rate')
        use_tree = cfg.get('use_tree_search', False)
        mode = cfg.get('tree_expand_mode', 'none') if use_tree else ('basic' if use_tree is False else 'none')
        max_iters = cfg.get('max_iterations')
        react_results = data.get('react_results', [])
        lats_results = data.get('lats_results', [])
        items.append({
            'path': str(p),
            'dataset_name': dataset_name,
            'dataset_prefix': dataset_prefix,
            'model': model,
            'react_success_rate': react_rate,
            'lats_success_rate': lats_rate,
            'use_tree': use_tree,
            'mode': mode,
            'max_iterations': max_iters,
            'react_results': react_results,
            'lats_results': lats_results,
        })
    return items

def mean(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None

def aggregate_heatmap(items, key):
    datasets = sorted({it['dataset_name'] for it in items})
    models = sorted({it['model'] for it in items})
    grid = [[None for _ in models] for _ in datasets]
    for di, d in enumerate(datasets):
        for mi, m in enumerate(models):
            rates = [it[key] for it in items if it['dataset_name'] == d and it['model'] == m]
            grid[di][mi] = mean(rates)
    return datasets, models, grid

def draw_heatmap(datasets, models, grid, title, output_file):
    arr = [[(v if v is not None else 0.0) for v in row] for row in grid]
    fig, ax = plt.subplots(figsize=(1.8 * max(4, len(models)), 1.2 * max(3, len(datasets))))
    im = ax.imshow(arr, cmap='cividis', vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(datasets)))
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.set_yticklabels(datasets)
    ax.set_title(title, fontsize=14)
    for i in range(len(datasets)):
        for j in range(len(models)):
            v = grid[i][j]
            s = '' if v is None else f"{v*100:.1f}%"
            ax.text(j, i, s, ha='center', va='center', color='black', fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='成功率')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

def aggregate_modes(items):
    out = {}
    for it in items:
        d = it['dataset_name']
        out.setdefault(d, {})
        out[d].setdefault('ReAct', [])
        if it['react_success_rate'] is not None:
            out[d]['ReAct'].append(it['react_success_rate'])
        mode = it['mode']
        out[d].setdefault(mode, [])
        if it['lats_success_rate'] is not None:
            out[d][mode].append(it['lats_success_rate'])
    return out

def draw_mode_bars(dataset_name, mode_rates, output_file):
    labels = list(mode_rates.keys())
    values = [mean(mode_rates[k]) if mode_rates[k] else None for k in labels]
    fig, ax = plt.subplots(figsize=(12, 6))
    vals = [v * 100 if v is not None else 0 for v in values]
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(len(labels))]
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor='black', linewidth=1.2)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f"{v:.1f}%", ha='center', va='bottom', fontsize=11)
    ax.set_ylabel('成功率 (%)', fontsize=12)
    ax.set_title(f'{dataset_name}: 模式对比', fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

def aggregate_iterations_rates(items):
    agg = {}
    for it in items:
        d = it['dataset_name']
        mi = it['max_iterations']
        if mi is None:
            continue
        agg.setdefault(d, {})
        agg[d].setdefault(mi, {'react': [], 'lats': []})
        if it['react_success_rate'] is not None:
            agg[d][mi]['react'].append(it['react_success_rate'])
        if it['lats_success_rate'] is not None:
            agg[d][mi]['lats'].append(it['lats_success_rate'])
    return agg

def draw_iteration_lines(dataset_name, rates_by_iter, output_file):
    xs = sorted(rates_by_iter.keys())
    react_means = [mean(rates_by_iter[x]['react']) for x in xs]
    lats_means = [mean(rates_by_iter[x]['lats']) for x in xs]
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap('tab10')
    ax.plot(xs, [v * 100 if v is not None else 0 for v in react_means], marker='o', color=cmap(0), label='ReAct', linewidth=2)
    ax.plot(xs, [v * 100 if v is not None else 0 for v in lats_means], marker='s', color=cmap(2), label='LATS', linewidth=2)
    ax.set_xlabel('最大迭代次数', fontsize=12)
    ax.set_ylabel('成功率 (%)', fontsize=12)
    ax.set_title(f'{dataset_name}: 迭代次数与成功率', fontsize=14)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

def count_iterations(history):
    if not history:
        return 0
    if isinstance(history[0], dict):
        has_iter = [h for h in history if 'iteration' in h]
        if has_iter:
            try:
                return max(int(h.get('iteration', 0)) for h in has_iter)
            except Exception:
                pass
        steps = [h for h in history if 'step' in h and isinstance(h.get('step'), str)]
        if steps:
            try:
                return sum(1 for h in steps if h['step'].startswith('expansion_iter_'))
            except Exception:
                pass
        return len(history)
    if isinstance(history[0], str):
        try:
            return sum(1 for s in history if s.startswith('迭代'))
        except Exception:
            return len(history)
    return len(history)

def aggregate_iteration_distributions(items):
    out = {}
    for it in items:
        d = it['dataset_name']
        out.setdefault(d, {'react': [], 'lats': []})
        for r in it.get('react_results', []) or []:
            h = r.get('history', [])
            out[d]['react'].append(count_iterations(h))
        for r in it.get('lats_results', []) or []:
            h = r.get('history', [])
            out[d]['lats'].append(count_iterations(h))
    return out

def draw_iteration_boxplot(dataset_name, distributions, output_file):
    data = [distributions.get('react', []), distributions.get('lats', [])]
    labels = ['ReAct', 'LATS']
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(data, tick_labels=labels, notch=False, patch_artist=True)
    for i, b in enumerate(bp['boxes']):
        b.set_facecolor('#FF6B6B' if i == 0 else '#4ECDC4')
    ax.set_ylabel('迭代次数', fontsize=12)
    ax.set_title(f'{dataset_name}: 迭代分布', fontsize=14)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

def extract_success_iteration(history):
    if not history:
        return None
    if isinstance(history[0], dict):
        for h in history:
            s = h.get('status')
            if s in ('success', 'success_found_in_final_check', 'success_manual_eval'):
                if 'iteration' in h:
                    try:
                        return int(h.get('iteration'))
                    except Exception:
                        pass
                step = h.get('step')
                if isinstance(step, str):
                    if step == 'initial_expansion':
                        return 0
                    if step.startswith('expansion_iter_'):
                        try:
                            return int(step.split('_')[-1])
                        except Exception:
                            return None
        for h in history:
            if h.get('reward') == 1:
                if 'iteration' in h:
                    try:
                        return int(h.get('iteration'))
                    except Exception:
                        return None
        return None
    if isinstance(history[0], str):
        for i, s in enumerate(history):
            if ('成功' in s) or ('找到解决方案' in s):
                if '迭代 ' in s:
                    try:
                        t = s.split('迭代 ')[1].split(':')[0]
                        return int(t)
                    except Exception:
                        return i + 1
                return i + 1
        return None
    return None

def collect_success_iterations(items, dataset_name):
    budget_max = 1
    react_iters = []
    lats_iters = []
    total_react = 0
    total_lats = 0
    for it in items:
        if it['dataset_name'] != dataset_name:
            continue
        for r in it.get('react_results', []) or []:
            total_react += 1
            h = r.get('history', [])
            budget_max = max(budget_max, count_iterations(h))
            t = extract_success_iteration(h)
            if t is not None:
                react_iters.append(t)
        for r in it.get('lats_results', []) or []:
            total_lats += 1
            h = r.get('history', [])
            budget_max = max(budget_max, count_iterations(h))
            t = extract_success_iteration(h)
            if t is not None:
                lats_iters.append(t)
    return budget_max, total_react, total_lats, react_iters, lats_iters

def build_curve(success_iters, total_count, max_budget):
    xs = list(range(1, max_budget + 1))
    ys = []
    for b in xs:
        c = sum(1 for t in success_iters if t is not None and t <= b)
        ys.append(c / total_count if total_count else 0)
    return xs, ys

def draw_budget_trend_lines(dataset_name, xs, ys_react, ys_lats, output_file):
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap('tab10')
    ax.plot(xs, [v * 100 for v in ys_react], marker='o', linewidth=2.2, color=cmap(0), label='ReAct')
    ax.plot(xs, [v * 100 for v in ys_lats], marker='s', linewidth=2.2, color=cmap(2), label='LATS')
    ax.set_xlabel('迭代预算', fontsize=12)
    ax.set_ylabel('成功率 (%)', fontsize=12)
    ax.set_title(f'{dataset_name}: 迭代预算-成功率趋势', fontsize=14)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

def mode_label(use_tree, mode):
    if use_tree:
        m = (mode or 'unknown').lower()
        return f'Tree-GRPO-{m}'
    return 'LATS-basic'

def collect_success_iterations_by_mode(items, dataset_name):
    data = {'ReAct': {'iters': [], 'total': 0, 'budget': 1}}
    for it in items:
        if it['dataset_name'] != dataset_name:
            continue
        # ReAct
        for r in it.get('react_results', []) or []:
            data['ReAct']['total'] += 1
            h = r.get('history', [])
            b = count_iterations(h)
            data['ReAct']['budget'] = max(data['ReAct']['budget'], b)
            t = extract_success_iteration(h)
            if t is not None:
                data['ReAct']['iters'].append(t)
        # LATS variants
        label = mode_label(it.get('use_tree'), it.get('mode'))
        if label not in data:
            data[label] = {'iters': [], 'total': 0, 'budget': 1}
        for r in it.get('lats_results', []) or []:
            data[label]['total'] += 1
            h = r.get('history', [])
            b = count_iterations(h)
            data[label]['budget'] = max(data[label]['budget'], b)
            t = extract_success_iteration(h)
            if t is not None:
                data[label]['iters'].append(t)
    # unify budget
    max_budget = max(v['budget'] for v in data.values()) if data else 1
    curves = {}
    for k, v in data.items():
        xs, ys = build_curve(v['iters'], v['total'], max_budget)
        curves[k] = {'xs': xs, 'ys': ys}
    return max_budget, curves

def draw_budget_trend_lines_multi(dataset_name, curves, output_file):
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = list(curves.keys())
    cmap = plt.get_cmap('tab10') if len(labels) <= 10 else plt.get_cmap('tab20')
    markers = ['o', 's', '^', 'D', 'v', 'X', 'P', 'h', '>', '<']
    for i, (label, cur) in enumerate(curves.items()):
        color = cmap(i % cmap.N)
        mk = markers[i % len(markers)]
        ax.plot(cur['xs'], [v * 100 for v in cur['ys']], marker=mk, linewidth=2.2, color=color, label=label)
    ax.set_xlabel('迭代预算', fontsize=12)
    ax.set_ylabel('成功率 (%)', fontsize=12)
    ax.set_title(f'{dataset_name}: 迭代预算-成功率（方法变体）', fontsize=14)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--output_dir', type=str, default='visualization')
    args = parser.parse_args()
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.output_dir, exist_ok=True)
    items = load_results(args.results_dir)
    ds, ms, grid_r = aggregate_heatmap(items, 'react_success_rate')
    ds2, ms2, grid_l = aggregate_heatmap(items, 'lats_success_rate')
    if ds and ms:
        out_react = os.path.join(args.output_dir, f'summary_heatmap_react_{ts}.png')
        draw_heatmap(ds, ms, grid_r, '模型-数据集成功率热力图 (ReAct)', out_react)
    if ds2 and ms2:
        out_lats = os.path.join(args.output_dir, f'summary_heatmap_lats_{ts}.png')
        draw_heatmap(ds2, ms2, grid_l, '模型-数据集成功率热力图 (LATS)', out_lats)
    modes = aggregate_modes(items)
    for d, rates in modes.items():
        out_file = os.path.join(args.output_dir, f"{(d.lower() if d else 'dataset')}_mode_comparison_{ts}.png")
        draw_mode_bars(d, rates, out_file)
    by_iter = aggregate_iterations_rates(items)
    for d, v in by_iter.items():
        out_file = os.path.join(args.output_dir, f"{(d.lower() if d else 'dataset')}_iteration_success_{ts}.png")
        draw_iteration_lines(d, v, out_file)
    dists = aggregate_iteration_distributions(items)
    for d, dist in dists.items():
        out_file = os.path.join(args.output_dir, f"{(d.lower() if d else 'dataset')}_iteration_distribution_{ts}.png")
        draw_iteration_boxplot(d, dist, out_file)
    datasets = sorted({it['dataset_name'] for it in items})
    for d in datasets:
        mb, tr, tl, ri, li = collect_success_iterations(items, d)
        xs_r, ys_r = build_curve(ri, tr, mb)
        xs_l, ys_l = build_curve(li, tl, mb)
        out_file = os.path.join(args.output_dir, f"{(d.lower() if d else 'dataset')}_budget_success_trend_{ts}.png")
        draw_budget_trend_lines(d, xs_r, ys_r, ys_l, out_file)
        mb2, curves = collect_success_iterations_by_mode(items, d)
        out_file2 = os.path.join(args.output_dir, f"{(d.lower() if d else 'dataset')}_budget_success_trend_modes_{ts}.png")
        draw_budget_trend_lines_multi(d, curves, out_file2)

if __name__ == '__main__':
    main()

