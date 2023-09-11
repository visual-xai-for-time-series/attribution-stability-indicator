import tqdm as tq

import scipy as sp
import numpy as np

import torch

from captum.metrics import infidelity

import exp_perturbation_analysis as exp_pa



def probability_distances(a, b, measure='js'):
    if measure == 'js':
        return min(1, max(0, 1 - sp.spatial.distance.jensenshannon(a, b, 2)))
    elif measure == 'h':
        return min(1, max(0, 1 - exp_pa.hellinger_native(a, b)))
    return 1



def correlation_distances(a, b, measure='pearson'):
    if measure == 'pearson':
        return (exp_pa.pearson_correlation(a, b) + 1) / 2
    return 0



def attribution_stability_indicator(model, sample_data, sample_label, func, func_kwargs=None, debug=False, weights=[1, 1, 1, 1], perturbation_fnc=exp_pa.delete_to_zero, seq_len=5):
    device = next(model.parameters()).device
    
    sample_ts = torch.from_numpy(sample_data).float().reshape(1, 1, -1).to(device)
    sample_l = torch.from_numpy(sample_label).reshape(1, -1)

    pred_org = model(sample_ts)

    sample_att = func(model).attribute(sample_ts, target=torch.argmax(sample_l, axis=1), **func_kwargs)

    sample_ts_np = sample_ts.detach().cpu().numpy().reshape(-1)
    sample_att_np = sample_att.detach().cpu().numpy().reshape(-1)

    value = perturbation_fnc(ts=sample_data)
    ts_pert_del, deleted = exp_pa.deletion_sequence(sample_ts_np, sample_att_np, np.percentile(sample_att_np, 90), value, seq_len)
    ts_pert_del_least, deleted = exp_pa.deletion_sequence(sample_ts_np, sample_att_np, np.percentile(sample_att_np, 10), value, seq_len, larger=False)
    
    sample_ts_pert = torch.from_numpy(ts_pert_del).float().reshape(1, 1, -1).to(device)
    pred_pert_del = model(sample_ts_pert)
    
    sample_att_pert_del = func(model).attribute(sample_ts_pert, target=torch.argmax(sample_l, axis=1), **func_kwargs)
    sample_att_pert_del_np = sample_att_pert_del.detach().cpu().numpy().reshape(-1)
    
    sample_ts_pert = torch.from_numpy(ts_pert_del_least).float().reshape(1, 1, -1).to(device)
    pred_pert_del_least = model(sample_ts_pert)
    
    sample_att_pert_del_least = func(model).attribute(sample_ts_pert, target=torch.argmax(sample_l, axis=1), **func_kwargs)
    sample_att_pert_del_np_least = sample_att_pert_del.detach().cpu().numpy().reshape(-1)
    
    # Predictions
    pred_org_np = pred_org.detach().cpu().numpy().reshape(-1).round(6)
    pred_pert_del_np = pred_pert_del.detach().cpu().numpy().reshape(-1).round(6)
    pred_pert_del_np_least = pred_pert_del_least.detach().cpu().numpy().reshape(-1).round(6)
    
    # Argmax
    argmax_factor_del = int(np.argmax(pred_org_np) == np.argmax(pred_pert_del_np))
    argmax_factor = argmax_factor_del
    
    # Jensen-Shannon
    js_factor_del = probability_distances(pred_pert_del_np, pred_org_np)
    js_factor = js_factor_del
    
    # Time Series Correlation
    ts_correl_del = 1 - correlation_distances(sample_ts_np, ts_pert_del)
    ts_factor = ts_correl_del
    
    # Attributions
    att_correl_del = correlation_distances(sample_att_np, sample_att_pert_del_np)
    att_factor = att_correl_del
    
    attribution_stability_indicator = (argmax_factor * weights[0] + js_factor * weights[1] + ts_factor * weights[2] + att_factor * weights[3]) / sum(weights)
    
    lp = np.argmax(sample_label)
    lp_org = np.argmax(pred_org_np)
    lp_del = np.argmax(pred_pert_del_np)
    
    if debug:
        print(f'XAI: {func.__name__}')
        print(f'Original Pred:  {pred_org_np}')
        print(f'Perturbed Pred: {pred_pert_del_np}')
        print()
        print(f'TS Correl: {ts_correl_del}')
        print(f'Att Correl: {att_correl_del}')
        print()
        print('Labels')
        print('TL  P1  Pd')
        print(f'{lp:2}  {lp_org:2}  {lp_del:2}')
        print()
        print('ASI')
        print(f'{attribution_stability_indicator:.5}')
        print()
    
    return attribution_stability_indicator, sample_att_np, (lp, lp_org, lp_del), sample_att_pert_del_np



def get_asi_for_dataset(loader, model, methods, weights, perturbation_fnc=exp_pa.delete_to_zero, seq_len=5, no_bar=False, full_dict=True):
    ret_dict = {m[0]: [] for m in methods}

    
    def adapted_perturbation_func(**kwargs):
        raw_data = np.array([x[0] for x in loader.dataset])
        return perturbation_fnc(ts=kwargs['ts'], gts=raw_data)


    for i in tq.tqdm(range(len(loader.dataset)), disable=no_bar):
        sample_ts_np, sample_l = loader.dataset[i]
        
        for m in methods:
            name, method, kwargs = m
            t = attribution_stability_indicator(model, sample_ts_np, sample_l, method, func_kwargs=kwargs, weights=weights, perturbation_fnc=adapted_perturbation_func, seq_len=seq_len)
            if not full_dict:
                t = (t[0], 0, t[2])
            ret_dict[name].append(t)
    
    return ret_dict



def batch_infidelity(model, sample_data, sample_attributions, perturbation_fnc):
    device = next(model.parameters()).device
    
    def perturb_fn(data):
        noise = torch.tensor(np.random.normal(0, 0.003, data.shape)).float().to(device)
        return noise, data - noise
    
    infid = infidelity(model, perturb_fn, sample_data, sample_attributions, max_examples_per_batch=1)
    return infid



def get_infidelity_for_dataset(loader, model, methods, no_bar=False):
    device = next(model.parameters()).device
    
    ret_dict = {m[0]: [] for m in methods}
    
    for m in methods:
        name, method, kwargs = m
        
        for data in tq.tqdm(loader, disable=no_bar):
            raw_data, labels = data
            raw_data_ = raw_data.reshape(raw_data.shape[0], 1, -1)
            raw_data_ = raw_data_.float().to(device)
            labels_ = labels.float().to(device)
            labels = np.argmax(labels.cpu().numpy(), axis=1)
            
            predictions = model(raw_data_)
            predictions = np.argmax(predictions.detach().cpu().numpy(), axis=1)
            
            raw_data_att = method(model).attribute(raw_data_, target=torch.argmax(labels_, axis=1), **kwargs)
            
            batch_infid = batch_infidelity(model, raw_data_, raw_data_att, None).cpu().numpy().tolist()
            
            l = [(batch_infid[x], 0, (labels[x], predictions[x], predictions[x])) for x in range(len(batch_infid))]
            ret_dict[name].extend(l)
            
    return ret_dict


