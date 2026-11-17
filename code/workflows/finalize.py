"""Shared final workflow: validation preparation -> recorded freeze -> scoring.

No report/poster editing. Models, row predictions and cases stay in data/;
versioned aggregate evidence and figures use the existing results layout.
"""
import copy
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from classification import score_model, train_model, cold_start_groups, fixed_error_cases
from evaluation import (DEFAULT_BUDGETS, partition_description, make_prediction_frame,
                        evaluate_amounts, fit_amount_bins, cost_sensitivity, evaluate_groups, evaluate_scores,
                        save_figures, save_c_extension_figures, save_handbook_c_figures,
                        save_review_policy_figure, evaluate_c_extensions,
                        evaluate_improved_frozen, publish_frozen_c_results)
from handbook import load_handbook_development, METADATA_COLUMNS, CONTEXT_COLUMNS
from pattern_mining import fit_handbook_rules, evaluate_handbook_rules
from paysim import load_paysim_sample, run_paysim_validation, _scores as paysim_scores
from preprocessing import DEFAULT_DATA_PATH, get_features, load_data, split_data, data_sha256, source_commit
from validation.run_b_validation import run_handbook_b, compact_metrics, save_b_extension_figure
from validation.run_d_validation import evaluate_rule_extensions, rule_training_metrics
from temporal import (handbook_history_builder, run_handbook_rolling,
                      select_review_threshold, evaluate_review_threshold)
from unsupervised import (run_handbook_c, score_handbook_c, build_relation_features,
                          evaluate_history_missing, fit_configuration, score_configuration)

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def source_hashes():
    paths = [ROOT/"requirements.txt", *sorted((ROOT/"code").rglob("*.py"))]
    paths += [ROOT/p for p in ["results/tables/b_freeze.json", "results/tables/b_validation.json",
                              "results/tables/c_improved.json", "results/tables/c_extensions.json",
                              "results/tables/d_freeze.json", "results/tables/handbook_a.json",
                              "report/d_validation/train_validation_rules.csv"]]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths}


def provenance():
    return {"source_commit": source_commit(ROOT),
            "source_sha256": source_hashes(), "python": platform.python_version(),
            "versions": {n:importlib.metadata.version(n) for n in ["numpy","pandas","scikit-learn","scipy","joblib"]}}


def stripped(value):
    """Remove private row identities/PR arrays from public aggregate dictionaries."""
    if isinstance(value, dict):
        return {k:stripped(v) for k,v in value.items() if k not in {
            "covered_row_ids","classifier_ids","anomaly_ids","combined_ids","pr_curve","cases"}}
    if isinstance(value,list):return [stripped(v) for v in value]
    return value


def rule_frame(rows, view):
    frame=rows[["row_id","Class","Time","TX_FRAUD_SCENARIO"]].copy()
    for col in ("Amount","customer_amount_ratio_7d","customer_count_7d","terminal_count_7d"):
        frame[col]=view.loc[rows.row_id,col].to_numpy()
    return frame


def thresholds(predictions):
    return {name:select_review_threshold(frame,.01) for name,frame in predictions.items()}


def save_rule_figure(result, path):
    from matplotlib.figure import Figure
    fig=Figure(figsize=(11,4),layout="constrained");axes=fig.subplots(1,2)
    cumulative=result["E15"]["cumulative"]
    axes[0].plot(range(1,len(cumulative)+1),[r["frauds"] for r in cumulative],marker="o")
    axes[0].set(xlabel="Training-ordered frozen rules",ylabel="Unique frauds covered")
    for name in ("classifier","IF"):
        groups=[g for g in result["mechanisms"] if g["frauds"]]
        axes[1].plot([g["scenario"] for g in groups],[g["models"][name]["recall"] for g in groups],marker="o",label=name)
    axes[1].set(xlabel="Final simulator scenario code",ylabel="Recall under global 1% list",ylim=(0,1));axes[1].legend()
    fig.suptitle(f"Handbook {result['phase']} — E15 frozen rules and post-hoc mechanisms")
    Path(path).parent.mkdir(parents=True,exist_ok=True);fig.savefig(path,dpi=160)


def prepare_final(args, *, destination=None):
    """Fit/verify only development rows. Write all G2 choices before test scoring."""
    from main import run_team_validation
    output=Path(args.output).resolve();results=Path(destination or args.results).resolve()
    if output.exists():raise ValueError("Use a new preparation output directory")
    if (results/"tables/group_freeze.json").exists():raise ValueError("Freeze already exists; reproduce into another results directory")
    output.mkdir(parents=True)
    print("Preparing ULB fixed models and all E01–E08 validation conditions",flush=True)
    run_team_validation(args.data,output/"ulb")
    train,val,test=split_data(load_data(args.data));parts={n:partition_description(f) for n,f in zip(("train","validation","test"),(train,val,test))};del test
    ulb=joblib.load(output/"ulb/models.joblib")
    c=read(ROOT/"results/tables/c_improved.json")
    alternatives={}
    for spec in c["selection_record"]["candidate_specifications"]:
        name=spec["id"];print("ULB frozen ablation",name,flush=True)
        alternatives[name]=fit_configuration(get_features(train),spec,42)
        scores,_=score_configuration(alternatives[name],get_features(val))
        for method,values in scores.items():
            metric=compact_metrics(make_prediction_frame(val,values,method));old=c["validation_metrics"][method]
            if not np.isclose(metric['AP'],old['AP'],rtol=0,atol=1e-12) or metric['budgets']!=old['budgets']:
                raise ValueError("Frozen C validation mismatch: "+method)
    classifiers={}
    for spec in read(ROOT/"results/tables/b_validation.json")["experiments"]:
        classifiers[spec['method']]=train_model(get_features(train),train.Class,spec['model_name'],spec['strategy'],42)
    classifier_validation = {n:evaluate_amounts(make_prediction_frame(val,score_model(m,get_features(val)),n),val,fit_amount_bins(train)) for n,m in classifiers.items()}
    label_summary=read(output/"ulb/aggregate/tables/b_extensions.json")
    label_models={}
    for row in label_summary['E03']['conditions']:
        key=row['mask_sha256']
        if key not in label_models:
            label_models[key]=joblib.load(output/f"ulb/b_extensions/ulb_labels_{row['fraction']:g}_{row['mask_seed']}.joblib")
    ulb_record=read(output/"ulb/aggregate/tables/team_validation.json")
    package={"ulb":{"bundle":ulb,"classifiers":classifiers,"ablations":alternatives,
                    "label_models":label_models,"label_summary":label_summary,
                    "if256":joblib.load(output/"ulb/b_extensions/ulb_if256.joblib"),
                    "extension_config":read(ROOT/"results/tables/c_extensions.json")["config"],
                    "rules":read(ROOT/"results/tables/d_freeze.json"),
                    "rule_training":rule_training_metrics(ROOT/"report/d_validation/train_validation_rules.csv"),
                    "thresholds":{n:{k:v for k,v in t.items() if k not in ['overall','windows']} for n,t in ulb_record['E08'].items()}}}
    package['ulb']['extension_config']['partitions']['test'] = parts['test']
    print("Preparing Handbook static/history/relationship and delayed-label snapshots",flush=True)
    development=load_handbook_development(args.handbook_dir)
    ht,hv=development['train'],development['validation']
    hresult,hbundle=run_handbook_c(ht,hv,development['train_views'],development['validation_views'],development['context'],
                                 dataset_manifest=development['manifest'],output=output/"handbook/private",results=output/"handbook/aggregate",feature_builder=handbook_history_builder)
    bresult=run_handbook_b(args.handbook_dir,output/"handbook_b",results=output/"handbook/aggregate")
    bmodels={v:joblib.load(output/f"handbook_b/handbook_lr_{v}.joblib") for v in bresult['conditions']}
    rolling=run_handbook_rolling(development,output/"handbook_rolling")
    relation=build_relation_features(development['context'])
    hp,_=score_handbook_c(hbundle,hv,development['validation_views'],relation)
    dtrain=rule_frame(ht,development['train_views']['both_history']);dval=rule_frame(hv,development['validation_views']['both_history'])
    hrules=fit_handbook_rules(dtrain,dval)
    hdiagnostics,cases=evaluate_handbook_rules(dval,{"classifier":hp['lr_both_history'],"IF":hp['if_both_history']},hrules,phase='validation')
    hdiagnostics['frozen_rules']=hrules;write(output/"handbook_rule_cases_validation.json",cases)
    write(results/"tables/handbook_d.json",stripped(hdiagnostics));save_rule_figure(hdiagnostics,results/"figures/handbook_d_validation.png")
    package['handbook']={"C":hbundle,"B":bmodels,"rolling":{n:joblib.load(output/f"handbook_rolling/{n}.joblib") for n in rolling['conditions']},
                         "rolling_summary":rolling,"rules":hrules,"entity_train":ht[['CUSTOMER_ID','TERMINAL_ID']].drop_duplicates(),
                         "manifest":development['manifest'],"thresholds":thresholds(hp),
                         "B_thresholds":{v:select_review_threshold(make_prediction_frame(hv,score_model(m,development['validation_views'][v]),v),.01) for v,m in bmodels.items()}}
    print("Preparing pinned PaySim audit and fixed reference snapshots",flush=True)
    ps=run_paysim_validation(args.paysim_archive,output/"paysim")
    if ps['status']!='validation_complete':raise ValueError("PaySim preparation not complete")
    pparts,pprovenance=load_paysim_sample(args.paysim_archive)
    pbundle=joblib.load(output/"paysim/models.joblib")
    pp={n:make_prediction_frame(pparts['validation'],v,n) for n,v in paysim_scores(pparts['validation'],pbundle).items()}
    package['paysim']={"bundle":pbundle,"manifest":pprovenance,"thresholds":thresholds(pp)}
    path=output/"models.joblib";joblib.dump(package,path)
    freeze={"schema_version":1,"phase":"G2","group_G2_complete":True,"test_scored":False,
            "authorization":"Project owner requested completion of all non-report/poster work on 2026-11-17; this is a computational freeze, not a claim of individual member signoff.",
            "provenance":provenance(),"models":{"path":str(path.relative_to(ROOT)),"sha256":data_sha256(path)},
            "data":{"ulb":{"sha256":data_sha256(args.data),"partitions":parts},
                    "handbook":development['manifest'],"paysim":pprovenance},
            "protocols":{"budgets":list(DEFAULT_BUDGETS),"primary":.01,"ranking":"score descending,row_id ascending,K=ceil(N*q)",
                         "ULB":"Original train only, six B strategies, all 20 declared C scores, fixed quotas and rules; E01–E08 retained",
                         "Handbook":"E10/E12/E13/E14/E15 static initial-train models; E11 separately saved last legal development snapshots; no test-time fit",
                         "E03":{k:v for k,v in label_summary['E03'].items() if k not in ['conditions','IF_reference']},
                         "E05":{k:v for k,v in label_summary['E05'].items() if k not in ['importances','elapsed_seconds','private_cases','case_counts']},
                         "E14":{"rates":[0,.1,.3],"seeds":[42,43,44]},
                         "E15":{"selected_scheme":hrules['selected_scheme'],"rule_ids":[r['rule_id'] for r in hrules['frozen_rules']]},
                         "E11":{n:v['snapshot'] for n,v in rolling['conditions'].items()},
                         "ULB_extensions":package['ulb']['extension_config'],
                         "ULB_classifier_parameters":{n:m.named_steps['classifier'].get_params() for n,m in classifiers.items()},
                         "ULB_ablation_specifications":c['selection_record']['candidate_specifications'],
                         "ULB_rules":package['ulb']['rules'],
                         "Handbook_static_config":hbundle['config'],
                         "thresholds":{"ulb":package['ulb']['thresholds'],"handbook":package['handbook']['thresholds'],
                                       "handbook_B":package['handbook']['B_thresholds'],"paysim":package['paysim']['thresholds']}},
            "validation":{"ULB":ulb_record,"ULB_classification_baselines":classifier_validation,'Handbook_C':hresult,'Handbook_B':bresult,'Handbook_D':hdiagnostics,'PaySim':ps},
            "model_fit_count_note":"13 label-subset LR snapshots reused across 15 mask conditions; E11 snapshots retain each declared label-delay policy."}
    # Only aggregate validation data; no row IDs, labels, models or PR arrays published.
    write(results/"tables/group_freeze.json",stripped(freeze))
    print("G2 written; no held-out predictions scored",flush=True)
    return freeze


def _assessment(data, predictions, bins, fixed_thresholds, output, *, phase='test'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    metrics={n:evaluate_amounts(p,data,bins) for n,p in predictions.items()}
    for name,pred in predictions.items():pred.to_csv(output/(name+'.csv'),index=False,float_format='%.17g')
    policies={n:{**selection,**evaluate_review_threshold(predictions[n],selection,window_seconds=3600)}
              for n,selection in fixed_thresholds.items() if n in predictions}
    return {"phase":phase,"partition":partition_description(data),"metrics":metrics,
            "cost_sensitivity":cost_sensitivity(metrics),"threshold_policies":policies}


def _b_test(data, frozen, output):
    from sklearn.inspection import permutation_importance
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    X=get_features(data);conditions=[]
    for row in frozen['label_summary']['E03']['conditions']:
        pred=make_prediction_frame(data,score_model(frozen['label_models'][row['mask_sha256']],X),'LR')
        conditions.append({k:copy.deepcopy(v) for k,v in row.items() if k not in ['metrics','score_seconds']})
        conditions[-1]['metrics']=compact_metrics(pred)
    full=next(r for r in conditions if r['fraction']==1)
    model=frozen['label_models'][full['mask_sha256']]
    importance=permutation_importance(model,X,data.Class,scoring='average_precision',n_repeats=5,random_state=42,n_jobs=1)
    pred=make_prediction_frame(data,score_model(model,X),'LR')
    cases=fixed_error_cases(pred,int(np.ceil(len(data)*.01)));cases.to_csv(output/'fixed_error_cases.csv',index=False)
    reference=make_prediction_frame(data,-frozen['if256'].score_samples(X),'IF256')
    result={'dataset_id':'ulb','phase':'test','E03':{**{k:copy.deepcopy(v) for k,v in frozen['label_summary']['E03'].items() if k not in ['conditions','IF_reference']},
             'conditions':conditions,'IF_reference':{'metrics':compact_metrics(reference)}},
            'E05':{'scoring':'average_precision','seed':42,'repeats':5,'fitting':'no fit, frozen complete-label LR',
                   'importances':[{'feature':n,'mean':float(importance.importances_mean[i]),'std':float(importance.importances_std[i]),'repeats':importance.importances[i].tolist()} for i,n in enumerate(X.columns)],
                   'case_rule':frozen['label_summary']['E05']['case_rule'],'case_counts':{str(k):int(v) for k,v in cases.category.value_counts().items()}}}
    return result


def cold_start_metrics(data, views, predictions, training_entities):
    """Use one global budget list per method; do not sum independent group quotas."""
    groups=cold_start_groups(training_entities,data)
    results={n:evaluate_groups(pred,groups,DEFAULT_BUDGETS,
                               group_values=['both_seen','new_customer_only','new_terminal_only','both_new'])
             for n,pred in predictions.items()}
    for result in results.values():
        for group in result['groups']:
            group['within_group']={k:v for k,v in group['within_group'].items() if k in ['status','n_samples','n_positive','AP','prevalence']}
            mask=groups==group['cluster_id']
            group['history_availability']={e:float((views['both_history'].loc[mask,f'{e}_no_history_7d']==0).mean()) if mask.any() else None for e in ['customer','terminal']}
    return results


def run_final_test(args, *, freeze_path=None, destination=None):
    """Read a completed freeze and score fixed snapshots. This function never fits."""
    freeze_path=Path(freeze_path or args.freeze);freeze=read(freeze_path)
    if not freeze.get('group_G2_complete') or freeze.get('test_scored'):raise ValueError('Need an unscored G2 manifest')
    if freeze.get('evaluation_source_sha256',freeze['provenance']['source_sha256'])!=source_hashes():raise ValueError('Source differs from frozen implementation')
    model_path=ROOT/freeze['models']['path']
    if data_sha256(model_path)!=freeze['models']['sha256']:raise ValueError('Frozen model bundle hash mismatch')
    output=Path(args.output).resolve();results=Path(destination or args.results).resolve()
    if output.exists() or (results/'tables/project_test.json').exists():raise ValueError('Use new test output/results paths')
    if data_sha256(args.data)!=freeze['data']['ulb']['sha256']:raise ValueError('ULB data changed')
    output.mkdir(parents=True);package=joblib.load(model_path)
    proof={**provenance(),'g2_reference':str(freeze_path),'g2_sha256':data_sha256(freeze_path)}
    print('Scoring frozen ULB snapshots',flush=True)
    _,_,data=split_data(load_data(args.data));u=package['ulb']
    if partition_description(data)!=freeze['data']['ulb']['partitions']['test']:raise ValueError('ULB test identities changed')
    predictions={n:make_prediction_frame(data,score_model(m,get_features(data)),n) for n,m in u['classifiers'].items()}
    for fitted in u['ablations'].values():
        values,_=score_configuration(fitted,get_features(data))
        predictions.update({n:make_prediction_frame(data,v,n) for n,v in values.items()})
    ulb=_assessment(data,predictions,u['extension_config']['amount_bins'],u['thresholds'],output/'ulb')
    core=evaluate_improved_frozen(u['bundle'],data,output/'ulb_core',phase='test')
    publish_frozen_c_results(output/'ulb_core',results,provenance={**proof,'command':'python code/main.py --run-final-test','dataset_sha256':freeze['data']['ulb']['sha256']})
    cp={n:predictions[n] for n in [u['extension_config']['classifier'],*u['extension_config']['selected_candidates'].values()]}
    ext=evaluate_c_extensions(cp,data,u['extension_config'],phase='test')
    ext['provenance']=proof;ext['group_G2_complete']=True
    write(results/'tables/c_extensions_test.json',stripped(ext));save_c_extension_figures(ext,results/'figures')
    diagnostic,model_metrics=evaluate_rule_extensions(data,{'classifier':predictions[u['extension_config']['classifier']], 'IF':predictions[u['extension_config']['anomaly']]},u['rules'],u['rule_training'])
    write(output/'ulb_rule_cases.json',diagnostic.pop('cases'));ulb['E04']=diagnostic
    btest=_b_test(data,u,output/'ulb_B');ulb['E03']=btest['E03'];ulb['E05']=btest['E05']
    write(results/'tables/b_extensions_test.json',stripped(btest));save_b_extension_figure(btest,results/'figures/b_extensions_test.png',phase='test')
    save_review_policy_figure(ulb['threshold_policies'],results/'figures/c_review_policy_test.png',phase='test')
    save_figures({'methods':{n:evaluate_scores(p.Class,p.score,row_ids=p.row_id) for n,p in predictions.items() if n in u['classifiers']},'label':'ULB test — six frozen classification strategies'},output/'ulb_plot')
    import shutil
    for figure in (output/'ulb_plot').glob('*.png'):shutil.copyfile(figure,results/'figures'/('b_test_'+figure.name))
    write(results/'tables/ulb_test.json',stripped(ulb))
    print('Scoring frozen Handbook views, nine history conditions and rule mechanisms',flush=True)
    hb=package['handbook'];csv=Path(args.handbook_dir)/'transactions.csv'
    if data_sha256(csv)!=freeze['data']['handbook']['data_sha256']:raise ValueError('Handbook source changed')
    all_rows=pd.read_csv(csv,float_precision='round_trip');context=all_rows[CONTEXT_COLUMNS]
    data=all_rows.loc[all_rows.Time.ge(45*86400)]
    if partition_description(data)!=freeze['data']['handbook']['partitions']['test']:raise ValueError('Handbook test identities changed')
    views_all=handbook_history_builder(context);views={n:x.loc[data.row_id] for n,x in views_all.items()}
    hp,_=score_handbook_c(hb['C'],data,views,build_relation_features(context))
    for view,m in hb['B'].items():hp['lr_'+view]=make_prediction_frame(data,score_model(m,views[view]),'lr_'+view)
    fixed={**hb['thresholds'],**{'lr_'+v:t for v,t in hb['B_thresholds'].items()}}
    for n,m in hb['rolling'].items():
        key='rolling_'+n;hp[key]=make_prediction_frame(data,score_model(m,views['both_history']),key)
        fixed[key]=hb['rolling_summary']['conditions'][n]['snapshot']['threshold']
    handbook=_assessment(data,hp,hb['C']['config']['amount_bins'],fixed,output/'handbook')
    handbook['dataset_id']='handbook';handbook['evidence_kind']='handbook_test'
    handbook['history_missing']=evaluate_history_missing(hb['C'],context,data,handbook_history_builder,views)
    handbook['cold_start']=cold_start_metrics(data,views,{n:hp[n] for n in ['lr_'+v for v in hb['B']]},hb['entity_train'])
    dresult,cases=evaluate_handbook_rules(rule_frame(data,views['both_history']),{'classifier':hp['lr_both_history'],'IF':hp['if_both_history']},hb['rules'],phase='test')
    write(output/'handbook_rule_cases.json',cases);handbook['E15']=dresult
    write(results/'tables/handbook_d_test.json',stripped(dresult));save_rule_figure(dresult,results/'figures/handbook_d_test.png')
    write(results/'tables/handbook_test.json',stripped(handbook));save_handbook_c_figures(handbook,results/'figures')
    print('Scoring frozen PaySim snapshots',flush=True)
    pparts,pm=load_paysim_sample(args.paysim_archive)
    if pm!=freeze['data']['paysim']:raise ValueError('PaySim source/sampling/splits changed')
    data=pparts['test'];pb=package['paysim']
    pp={n:make_prediction_frame(data,v,n) for n,v in paysim_scores(data,pb['bundle']).items()}
    paysim=_assessment(data,pp,pb['bundle']['amount_bins'],pb['thresholds'],output/'paysim')
    write(results/'tables/paysim_test.json',stripped(paysim))
    summary={'phase':'test','test_scored':True,'group_G2_complete':True,'configuration_reselected':False,
             'provenance':proof,'datasets':{n:{'partition':r['partition'],'methods':list(r['metrics'])} for n,r in [('ulb',ulb),('handbook',handbook),('paysim',paysim)]},
             'checks':{'models_fitted_during_test':0,'all_frozen_partitions_verified':True,'all_protocol_conditions_retained':True}}
    write(results/'tables/project_test.json',summary)
    return summary


def _compare_numeric(expected, actual, path='root', *, platform_differences=None):
    """Cross-machine aggregate reproduction: counts exact; float diagnostics 1e-8."""
    if isinstance(expected,dict):
        if set(expected)!=set(actual):raise ValueError('Reproduction keys differ at '+path)
        for k in expected:_compare_numeric(expected[k],actual[k],path+'.'+k,platform_differences=platform_differences)
    elif isinstance(expected,list):
        if len(expected)!=len(actual):raise ValueError('Reproduction length differs at '+path)
        for i,(x,y) in enumerate(zip(expected,actual)):_compare_numeric(x,y,path+f'[{i}]',platform_differences=platform_differences)
    elif platform_differences is not None and path.startswith('ulb_test.json.metrics.random_forest_original.') and (path.endswith('.AP') or ('.amount_groups[' in path and path.endswith('.reviewed_from_global_list'))):
        if not np.isclose(expected,actual,rtol=1e-8,atol=1e-8):
            platform_differences.append({'path':path,'reference':expected,'rebuilt':actual,'delta':actual-expected})
    elif isinstance(expected,float):
        if not np.isclose(expected,actual,rtol=1e-8,atol=1e-8):raise ValueError('Reproduction numeric difference at '+path)
    elif expected!=actual:raise ValueError('Reproduction value differs at '+path)


def reproduce_final(args):
    """Independent new snapshots, same recorded protocol, compare all test aggregates."""
    from argparse import Namespace
    base=Path(args.output).resolve()
    if base.exists():raise ValueError('Use a new reproduction directory')
    original=read(args.freeze)
    if original.get('evaluation_source_sha256',original['provenance']['source_sha256'])!=source_hashes():raise ValueError('Reproduce the frozen source revision')
    reference_results=Path(args.freeze).resolve().parent.parent
    new_args=Namespace(**vars(args));new_args.output=base/'prepared';new_args.results=base/'development'
    replay_freeze=prepare_final(new_args)
    _compare_numeric(original['data'],replay_freeze['data'])
    for key in ['ranking','E14','E15','ULB_ablation_specifications','ULB_classifier_parameters']:
        _compare_numeric(original['protocols'][key],replay_freeze['protocols'][key],key)
    new_args.freeze=new_args.results/'tables/group_freeze.json';new_args.output=base/'test';new_args.results=base/'results'
    run_final_test(new_args)
    comparisons={};platform_differences=[]
    for filename,keys in {'ulb_test.json':['partition','metrics','E03','E05','E04'],
                          'handbook_test.json':['partition','metrics','history_missing','cold_start','E15'],
                          'paysim_test.json':['partition','metrics'],
                          'c_extensions_test.json':['partition','metrics','cost_sensitivity','uncertainty']}.items():
        old,new=read(reference_results/'tables'/filename),read(new_args.results/'tables'/filename)
        for key in keys:
            # E03 reports stored original fit durations as provenance, not performance.
            def without_runtime(v):
                if isinstance(v,dict):return {k:without_runtime(x) for k,x in v.items() if k not in ['train_seconds','score_seconds','private_cases']}
                if isinstance(v,list):return [without_runtime(x) for x in v]
                return v
            _compare_numeric(without_runtime(old[key]),without_runtime(new[key]),filename+'.'+key,platform_differences=platform_differences)
        comparisons[filename]=True
    result={'status':'core_checks_passed_with_platform_differences' if platform_differences else 'passed','phase':'independent_reproduction','reference_freeze_sha256':data_sha256(args.freeze),
            'checks':comparisons,'platform_differences':platform_differences,'provenance':provenance(),'model_rebuilt':True,
            'physical_host':platform.node(),'platform':platform.platform(),
            'note':'All designated conditions retained; no test-informed selection. Different serialized bundle bytes are allowed across platforms; Primary methods and all other fields must match; the supplementary RF AP and allocation by amount group are separately reported when rebuilding on another platform, never asserted identical.'}
    write(base/'reproduction.json',result)
    print(json.dumps(result),flush=True)
    return result


def package_technical(output):
    """Reproducible technical handoff; final Report.pdf is intentionally absent."""
    from zipfile import ZipFile, ZIP_DEFLATED
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    paths=[]
    for folder in ['code','tests','results','report/d_validation']:
        paths += [p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [ROOT/p for p in ['README.md','requirements.txt','data/README.md','report/C_IMPROVEMENTS.md']]
    with ZipFile(output,'w',ZIP_DEFLATED) as archive:
        for path in sorted(paths):archive.write(path,path.relative_to(ROOT).as_posix())
        archive.writestr('SOURCE_COMMIT',source_commit(ROOT)+'\n')
        archive.writestr('PACKAGE_STATUS.txt','Technical payload only. Add the final reviewed Report.pdf to form Group_20_Project.zip. Poster.pdf is submitted separately.\n')
    return {'file':str(output),'sha256':data_sha256(output),'files':len(paths)+2,'contains_report_or_poster':False}
