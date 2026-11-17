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


def main():
    BUILD.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
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
    print(json.dumps({'output':'report/Report.pdf','main_text_pages':pages,'source':'report/Report.md','model_training_or_scoring':False}))


if __name__=='__main__':main()
