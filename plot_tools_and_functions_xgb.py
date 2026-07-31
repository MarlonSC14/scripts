import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import pprint
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, precision_score, auc, precision_recall_curve 
from matplotlib.patches import Patch, Circle
from matplotlib.lines import Line2D
import ks_test

# Defining a function to plot feature importance
# The function needs the importances, the names of the features and the model type

def plot_feature_importance(importance,names,model_type):

  #Creating arrays from feature importance and feature names
  feature_importance = np.array(importance)
  feature_names = np.array(names)

  #Creating a DataFrame using a Dictionary
  data={'feature_names':feature_names,'feature_importance':feature_importance}
  fi_df = pd.DataFrame(data)

  #Sort the DataFrame in order decreasing feature importance
  fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

  pprint.pprint(dict(zip(fi_df['feature_names'],fi_df['feature_importance'])))

  #Define size of bar plot
  plt.figure(figsize=(10,8))
  #Plot Searborn bar chart
  sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
  #Add chart labels
  plt.title(model_type + ' FEATURE IMPORTANCE')
  plt.xlabel('FEATURE IMPORTANCE')
  plt.ylabel('FEATURE NAMES')


# Defining a function to plot a correlation heatmap for the features
# The function only needs the training pandas DataFrame

def correlation_heatmap(train):

  # Obtaining the correlation between colmns of the DataFrame
  correlations = train.corr()

  # Plotting the correlation heatmap using the heatmap function of the seaborn library
  fig, ax = plt.subplots(figsize=(10,10))
  sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', cmap="YlGnBu",
              square=True, linewidths=.5, cbar_kws={"shrink": .70}, annot=True, annot_kws={"size": 35 / np.sqrt(len(correlations))}
              )
  plt.show();


# This function plots a precision-recall curve and displays the aucpr values for train and test datasets in classification

def plot_precision_recall_curve(model, train, test, cols):
  pred_valueste = model.predict_proba(test[cols])[:,1]
  pred_valuestr = model.predict_proba(train[cols])[:,1]
  precisionte, recallte, thresholdste = precision_recall_curve(test['label'], pred_valueste)
  precisiontr, recalltr, thresholdstr = precision_recall_curve(train['label'], pred_valuestr)
  auc_precision_recallte = auc(recallte, precisionte)
  auc_precision_recalltr = auc(recalltr, precisiontr)

  fig, ax = plt.subplots()
  ax.plot(precisionte, recallte, label='Test')
  ax.plot(precisiontr, recalltr, label='Train')
  plt.legend(loc="best")
  ax.set(xlim=(0, 1.1), xticks=np.arange(0, 1.1, 0.1),
          ylim=(0, 1.1), yticks=np.arange(0, 1.1, 0.1))
  ax.set_xlabel('precision')
  ax.set_ylabel('recall')
  plt.text(0,0.6,
          "aucpr test = {}".format(f'{auc_precision_recallte:.4f}'))
  plt.text(0,0.5,
          "aucpr train = {}".format(f'{auc_precision_recalltr:.4f}'))
  plt.grid()

  return fig, ax


def params_to_string(model):
  model.get_params()
  string = ''
  lens = [len(k) for k in model.get_params()]
  max_len = max(lens)
  for k,v in model.get_params().items():
    #string += f'{k+" "*(max_len-len(k))} =  {v}\n'
    string += f'{k} = {v}\n'

  return string


def plot_roc_curve(model, test, train):
  test_xgb_per = model.predict_proba(test[cols])
  train_xgb_per = model.predict_proba(train[cols])

  fpr, tpr, thr_ = roc_curve(test.label, test_xgb_per[:,1])
  auc = roc_auc_score(test.label, test_xgb_per[:,1])
  plt.plot(1-fpr, tpr, label=f'Test = {round(auc,4)}', linewidth=2)

  fpr, tpr, thr_ = roc_curve(train.label, train_xgb_per[:,1])
  auc = roc_auc_score(train.label, train_xgb_per[:,1])
  plt.plot(1-fpr, tpr, label=f'Train = {round(auc,4)}', linewidth=2)

  plt.plot(np.linspace(0,1), 1-np.linspace(0,1), color='red', label='Random choice', ls='--', linewidth=0.5)
  plt.xlabel('Background rejection')
  plt.ylabel('Signal efficiency')
  plt.legend(frameon=True, title='auc')


def plot_classifier_distributions(model, test, train, cols, print_params=False, bins=25, figsize=(10, 7)):

    test_background = model.predict_proba(test.query('label==0')[cols])[:,1]
    test_signal     = model.predict_proba(test.query('label==1')[cols])[:,1]
    train_background= model.predict_proba(train.query('label==0')[cols])[:,1]
    train_signal    = model.predict_proba(train.query('label==1')[cols])[:,1]

    test_pred = model.predict_proba(test[cols])[:,1]
    train_pred= model.predict_proba(train[cols])[:,1]

    density = True

    fig, ax = plt.subplots(figsize=figsize)

    background_color = 'red'

    opts = dict(
        range=[0,1],
        bins = bins,
        density = density
    )
    histtype1 = dict(
        histtype='stepfilled',
        linewidth=3,
        alpha=0.45,
    )

    ax.hist(train_background, **opts, **histtype1,
             facecolor=background_color,
             edgecolor=background_color,
             zorder=0)
    ax.hist(train_signal, **opts, **histtype1,
             facecolor='blue',
             edgecolor='blue',
             zorder=1000)

    hist_test_0 = np.histogram(test_background, **opts)
    hist_test_1 = np.histogram(test_signal, **opts)
    bins_mean = (hist_test_0[1][1:]+hist_test_0[1][:-1])/2
    bin_width = bins_mean[1]-bins_mean[0]
    area0 = bin_width*np.sum(test.label==0)
    area1 = bin_width*np.sum(test.label==1)

    opts2 = dict(
          capsize=3,
          ls='none',
          marker='o'
    )

    ax.errorbar(bins_mean, hist_test_0[0],  yerr = np.sqrt(hist_test_0[0]/area0), xerr=bin_width/2,
                 color=background_color, **opts2, zorder=100)
    ax.errorbar(bins_mean, hist_test_1[0],  yerr = np.sqrt(hist_test_1[0]/area1), xerr=bin_width/2,
                 color='blue', **opts2, zorder=10000)

    _ks_back = ks_test.ks_2samp_sci(train_background, test_background)[1]
    _ks_sign = ks_test.ks_2samp_sci(train_signal, test_signal)[1]

    print('Own ks test\n',
          ks_test.ks_2samp_weighted(train_background, test_background)[1],
          ks_test.ks_2samp_weighted(train_signal, test_signal)[1], sep='\n\t')

    auc_test  = roc_auc_score(test.label,test_pred )
    auc_train = roc_auc_score(train.label,train_pred)
    legend_elements = [Patch(facecolor='black', edgecolor='black', alpha=0.4,
                             label=f'Train (auc) : {round(auc_train,8)}'),
                      Line2D([0], [0], marker='|', color='black',
                             label=f'Test (auc) : {round(auc_test,8)}',
                              markersize=25, linewidth=1),
                       Circle((0.5, 0.5), radius=2, color='red',
                              label=f'Background (ks-pval) : {round(_ks_back,8)}',),
                       Circle((0.5, 0.5), 0.01, color='blue',
                              label=f'Signal (ks-pval) : {round(_ks_sign,8)}',),
                       ]

    ax.legend(
              #title='KS test',
              handles=legend_elements,
              #bbox_to_anchor=(0., 1.02, 1., .102),
              loc='upper center',
              ncol=2,
              #mode="expand",
              #borderaxespad=0.,
              frameon=True,
              fontsize=15)

    if print_params:
      ax.text(1.02, 1.02, params_to_string(model),
        transform=ax.transAxes,
      fontsize=13, ha='left', va='top')

    ax.set_yscale('log')
    ax.set_xlabel('XGB output')
    #ax.set_ylim(0.005, 100)

    #plt.savefig(os.path.join(dir_, 'LR_overtrain.pdf'), bbox_inches='tight')

    # del test_background, test_signal
    # del train_background, train_signal
    # del test_pred, train_pred
    return fig, ax


# Defining the recall metric
# This function is built to work with XGB models and fittings

def recall_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = recall_score(y_true, np.round(y_pred), average='weighted', zero_division=0)
    return 'recall_err', err


# Defining the precision metric
# This function is built to work with XGB models and fittings

def precision_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = precision_score(y_true, np.round(y_pred), average='weighted', zero_division=0)
    return 'precision_err', err