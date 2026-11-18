"""Build report-sized figures from frozen test JSON and render Report.md.

Requires the project Python environment, Pandoc and XeLaTeX. It never trains,
selects or scores a model. PDF figures are public; PNG previews/TeX stay in data/.
"""
import json
from pathlib import Path
import re
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'data/report_build'
FIGURES = ROOT / 'results/figures'
COLORS = ['#C35136', '#236B8E', '#50866A', '#777777']


def read(name):
    return json.loads((ROOT / 'results/tables' / (name + '.json')).read_text())


def budget(metric, fraction=.01):
    return next(r for r in metric['budgets'] if r['review_fraction'] == fraction)


def finish(fig, name):
    fig.tight_layout(pad=.7)
    fig.savefig(FIGURES / (name + '.pdf'), metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(BUILD / (name + '.png'), dpi=300)
    plt.close(fig)


def figures():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
                         'legend.fontsize': 8, 'legend.frameon': False,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .15, 'lines.linewidth': 1.6,
                         'lines.markersize': 4, 'pdf.fonttype': 42})
    u, b, c, h = (read(n) for n in ['ulb_test', 'b_extensions_test', 'c_extensions_test', 'handbook_test'])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.65))
    for i, (name, label) in enumerate([('logistic_regression_class_weight', 'Weighted LR'),
                                       ('if_sample4096', 'IF4096'), ('km_base_raw', 'Cluster distance')]):
        values = u['metrics'][name]['budgets']
        for ax, key in zip(axes, ['recall_at_k', 'precision_at_k']):
            ax.plot([r['review_fraction']*100 for r in values], [r[key]*100 for r in values],
                    marker=['o','s','^'][i], color=COLORS[i], label=label)
            ax.set(xlabel='Review budget (%)', ylabel=key.split('_')[0].title()+' (%)', ylim=(0,100))
    axes[0].legend(loc='lower right');finish(fig, 'report_budget')

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.65))
    fractions = b['E03']['fractions']
    for i, (key, label) in enumerate([('AP','AP'), ('recall','Recall at 1%')]):
        values = [[r['metrics']['AP'] if key=='AP' else budget(r['metrics'])['recall_at_k']
                   for r in b['E03']['conditions'] if r['fraction']==f] for f in fractions]
        x = np.array(fractions)*100
        axes[0].plot(x, [np.mean(v) for v in values], marker=['o','s'][i], color=COLORS[i], label=label)
        axes[0].fill_between(x, [min(v) for v in values], [max(v) for v in values], color=COLORS[i], alpha=.12)
    axes[0].axhline(b['E03']['IF_reference']['metrics']['AP'], color=COLORS[2], ls='--', label='Fixed IF256 AP')
    axes[0].set(xlabel='Labeled training rows (%)', ylabel='Test metric', ylim=(0,1));axes[0].legend(loc='lower right')
    rows=sorted(b['E05']['importances'], key=lambda r:-r['mean'])[:8][::-1]
    axes[1].barh([r['feature'] for r in rows], [r['mean'] for r in rows], xerr=[r['std'] for r in rows], color=COLORS[0], capsize=2)
    axes[1].set(xlabel='Test AP decrease after permutation', title='Fixed LR; 5 repeats')
    finish(fig, 'report_supervision')

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.65))
    names=['logistic_regression_class_weight','if_sample4096','km_base_raw','amount_descending']
    labels=['LR','IF4096','Distance','Amount'];x=np.arange(4)
    axes[0].bar(x-.17,[budget(c['metrics'][n])['recall_at_k']*100 for n in names], .34,label='Fraud count',color=COLORS[0])
    axes[0].bar(x+.17,[budget(c['metrics'][n])['amount_coverage']*100 for n in names],.34,label='Fraud amount',color=COLORS[1])
    axes[0].set(xticks=x,xticklabels=labels,ylabel='Coverage at 1% (%)',ylim=(0,100));axes[0].legend()
    for i,n in enumerate(names[:2]):
        rows=[r for r in c['cost_sensitivity'] if r['method']==n and r['c_FN']==50]
        axes[1].plot([r['review_fraction']*100 for r in rows],[r['hypothetical_loss'] for r in rows],marker=['o','s'][i],color=COLORS[i],label=['LR','IF4096'][i])
    axes[1].set(xlabel='Review budget (%)',ylabel='Hypothetical loss',title='FN:FP cost = 50:1');axes[1].legend()
    finish(fig,'report_amount')

    fig,axes=plt.subplots(1,2,figsize=(6.6,2.65));views=['transaction_only','customer_history','terminal_history','both_history','both_history_graph'];x=np.arange(5)
    for i,family in enumerate(['lr','if']):
        for ax,key in zip(axes,['AP','recall']):
            vals=[h['metrics'][family+'_'+v]['AP'] if key=='AP' else budget(h['metrics'][family+'_'+v])['recall_at_k'] for v in views]
            ax.bar(x+(i-.5)*.36,vals,.36,color=COLORS[i],label=family.upper())
            ax.set(xticks=x,xticklabels=['Current','Customer','Terminal','Both','+Relation'],ylabel='AP' if key=='AP' else 'Recall at 1%',ylim=(0,.35))
            ax.tick_params(axis='x',rotation=25)
    axes[0].legend();finish(fig,'report_context')

    fig,axes=plt.subplots(1,2,figsize=(6.6,2.65));rates=h['history_missing']['rates']
    for i,name in enumerate(['lr_both_history','if_both_history']):
        for ax,key in zip(axes,['AP','recall']):
            values=[[r['metrics'][name]['AP'] if key=='AP' else budget(r['metrics'][name])['recall_at_k'] for r in h['history_missing']['records'] if r['missing_fraction']==rate] for rate in rates]
            x=np.array(rates)*100;ax.plot(x,[np.mean(v) for v in values],marker=['o','s'][i],color=COLORS[i],label=['LR','IF'][i])
            ax.fill_between(x,[min(v) for v in values],[max(v) for v in values],color=COLORS[i],alpha=.14)
            ax.set(xlabel='History events omitted (%)',ylabel='AP' if key=='AP' else 'Recall at 1%',ylim=(0,.30))
    axes[0].legend();finish(fig,'report_stress')


def audit_tables():
    """Check every numerical report table against canonical experiment outputs."""
    tables = {}
    current = None
    for line in (ROOT / 'report/Report.md').read_text().splitlines():
        if not line.startswith('|'):
            current = None
            continue
        cells = [v.strip() for v in line.strip('|').split('|')]
        if all(re.fullmatch(r':?-+:?', v) for v in cells):
            continue
        if current is None:
            current = tuple(cells)
            tables[current] = []
        else:
            tables[current].append(cells)
    checked = []

    def check(name, expected, second_header=None):
        expected = [[str(v) for v in row] for row in expected]
        matches = [rows for header, rows in tables.items()
                   if header[0] == name and
                   (second_header is None or header[1] == second_header)]
        if expected not in matches:
            raise ValueError('Report numerical table differs from source: ' + name)
        checked.append(name if second_header is None else name + ' / ' + second_header)

    u, c, b, e, h, d, p = (read(n) for n in [
        'ulb_test', 'c_test', 'b_extensions_test', 'c_extensions_test',
        'handbook_test', 'handbook_d_test', 'paysim_test'])
    if any(v['phase'] != 'test' for v in [u, c, b, e, h, d, p]):
        raise ValueError('Report figures/tables require formal test results')
    freeze = read('group_freeze')
    rows = []
    for key, label in [('ulb','ULB, cleaned'),('handbook','Handbook'),('paysim','PaySim sample')]:
        parts = freeze['data'][key]['partitions']
        rows.append([label, *[f"{parts[n]['rows']:,} / {parts[n]['frauds']:,}" for n in ['train','validation','test']]])
    check('Dataset', rows)
    model_names = [('logistic_regression_original','LR'),('decision_tree_original','Decision tree'),
                   ('gaussian_nb_original','Gaussian NB'),('random_forest_original','Random forest'),
                   ('logistic_regression_class_weight','Weighted LR (selected)'),
                   ('logistic_regression_random_undersample','Undersampled LR'),
                   ('km_base_raw','Cluster distance'),('if_sample4096','IF4096 (selected)')]
    rows = []
    for name, label in model_names:
        m = u['metrics'][name]; q = budget(m)
        rows.append([label, f"{m['AP']:.4f}", q['TP'], q['FP'],
                     f"{q['recall_at_k']*100:.2f}", f"{q['precision_at_k']*100:.2f}"])
    check('Method', rows)
    rows = []
    for lr, iff in zip(c['groups']['logistic_regression_class_weight']['groups'],
                       c['groups']['if_sample4096']['groups']):
        a, z = lr['global_budget_allocation'][2], iff['global_budget_allocation'][2]
        rows.append([lr['cluster_id'],lr['within_group']['n_samples'],lr['within_group']['n_positive'],
                     f"{a['reviewed_in_group']} / {a['frauds_detected_in_group']}",
                     f"{z['reviewed_in_group']} / {z['frauds_detected_in_group']}"])
    check('Cluster',rows)
    rows=[]
    for name,label in [('logistic_regression_class_weight','Weighted LR'),('if_sample4096','IF4096')]:
        t=u['threshold_policies'][name]['overall']['fixed_threshold']
        rows.append([label,f"{t['value']:.4f}",t['K'],t['TP'],f"{t['actual_review_fraction']*100:.2f}"])
    check('Method', rows, 'Frozen threshold')
    rows=[]
    for f in b['E03']['fractions']:
        ms=[r['metrics'] for r in b['E03']['conditions'] if r['fraction']==f]
        aps=[r['AP'] for r in ms];tps=[budget(r)['TP'] for r in ms]
        rows.append([f'{100*f:g}%',f'{np.mean(aps):.4f}',f'{min(aps):.4f}–{max(aps):.4f}',f'{min(tps)}–{max(tps)}'])
    check('Labels visible',rows)
    rows=[]
    for scheme,label in [('paired_rows','Paired rows (500)'),('paired_time_blocks','Time blocks (200)')]:
        z=e['uncertainty']['schemes'][scheme]['differences']['logistic_regression_class_weight minus if_sample4096']
        rows.append([label,f"{z['AP']['point']:.4f}",f"[{z['AP']['lower']:.4f}, {z['AP']['upper']:.4f}]",
                     f"{z['recall']['point']*100:.2f}",f"[{z['recall']['lower']*100:.2f}, {z['recall']['upper']*100:.2f}]"])
    check('Resampling',rows)
    rows=[]
    for v,label in zip(['transaction_only','customer_history','terminal_history','both_history','both_history_graph'],
                       ['Current transaction','+ Customer history','+ Terminal history','+ Both histories','Both + relations']):
        l,r=h['metrics']['lr_'+v],h['metrics']['if_'+v]
        rows.append([label,f"{l['AP']:.4f}",budget(l)['TP'],f"{r['AP']:.4f}",budget(r)['TP']])
    check('Input view',rows)
    rows=[]
    for name,label in [('rolling_static','Static'),('rolling_weekly_delay_0d','Weekly, delay 0 days'),
                       ('rolling_weekly_delay_1d','Weekly, delay 1 day'),('rolling_weekly_delay_7d','Weekly, delay 7 days')]:
        m=h['metrics'][name];q=budget(m)
        rows.append([label,f"{m['AP']:.4f}",q['TP'],f"{q['recall_at_k']*100:.2f}"])
    check('History LR snapshot',rows)
    check('Final scenario',[[r['scenario'],r['frauds'],r['models']['classifier']['TP'],r['models']['IF']['TP'],r['rule_covered_frauds']]
                            for r in d['mechanisms'] if r['frauds']])
    rows=[]
    for name,label in [('amount','Amount'),('weighted_lr','Weighted LR'),('fixed_if','Fixed IF')]:
        m=p['metrics'][name];q=budget(m)
        rows.append([label,f"{m['AP']:.4f}",q['TP'],f"{q['precision_at_k']*100:.2f}",f"{q['recall_at_k']*100:.2f}"])
    check('PaySim method',rows)
    structure=read('c_baseline')['training_structure'];rows=[]
    for k in [3,5,8]:
        items=[r for r in structure if r['k']==k]
        rows.append([k,f"{np.mean([r['inertia'] for r in items]):,.0f}",f"{np.mean([r['sample_silhouette'] for r in items]):.4f}",f"{items[0]['seed_stability_ARI']:.4f}"])
    check('k',rows)
    rows=[]
    for name,source in [('ULB R001',u['E04']),('Handbook R001',d['E15'])]:
        first=source['cumulative'][0];a,j=first['transactions'],first['frauds'];n,f=source['n_transactions'],source['n_fraud']
        rows.append([name,f'{100*a/n:.4f}',f'{100*j/n:.4f}',f'{100*j/a:.4f}',f'{j*n/(a*f):.4f}',f'{.5*(j/a+j/f):.4f}'])
    check('Rule',rows)
    ulb=json.loads((ROOT/'report/d_validation/manifest.json').read_text())['schemes']
    handbook=read('handbook_d')['frozen_rules']['schemes'];rows=[]
    for name,schemes in [('ULB',ulb),('Handbook',handbook)]:
        for scheme in ['equal_width','equal_frequency']:
            m=schemes[scheme];rows.append([name,scheme.replace('_','-'),m['candidate_count'] if name=='ULB' else len(m['candidates']),len(m['retained_rule_ids']) if name=='ULB' else len(m['rules'])])
    check('Dataset', rows, 'Binning')
    return checked


def main():
    BUILD.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    checked_tables = audit_tables()
    figures()
    for executable in ['pandoc','xelatex']:
        if shutil.which(executable) is None:raise RuntimeError('Install '+executable+' to build the PDF')
    tex=BUILD/'Report.tex'
    subprocess.run(['pandoc',str(ROOT/'report/Report.md'),'--standalone','--resource-path='+str(ROOT/'report')+':'+str(ROOT),'-o',str(tex)],check=True,cwd=ROOT)
    for _ in range(2):
        subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(BUILD),str(tex)],check=True,cwd=ROOT/'report',stdout=subprocess.DEVNULL)
    aux=(BUILD/'Report.aux').read_text();match=re.search(r'\\newlabel\{maintextend\}\{\{[^}]*\}\{(\d+)\}',aux)
    if match is None:raise RuntimeError('Missing main-text pagination label')
    pages=int(match.group(1))
    if not 6<=pages<=10:raise RuntimeError(f'Main text has {pages} pages; required 6-10')
    shutil.copyfile(BUILD/'Report.pdf',ROOT/'report/Report.pdf')
    print(json.dumps({'output':'report/Report.pdf','main_text_pages':pages,'source':'report/Report.md','model_training_or_scoring':False,'numeric_tables_checked':len(checked_tables)}))


if __name__=='__main__':main()
