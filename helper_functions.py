import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import roc_curve, roc_auc_score

import scipy.stats as stat

import matplotlib.pyplot as plt
import seaborn as sns

import pickle

import config


# Setting default style of the graphs
sns.set()

# preprocessing
def preprocessing_continous_variables(loan_data):
    # preprocessing object variables
    ## dealing with emp_length
    loan_data['emp_length_int'] = loan_data['emp_length'].str.replace('\+ years','',regex=True)
    loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('< 1 year',str(0))
    loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('n/a',str(0))
    loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' years','')
    loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' year','')
    loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])

    ## dealing with earliest_cr_line    
    loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'], format = '%b-%y')
    ### Assume we are now in December 2017
    loan_data['mths_since_earliest_cr_line'] = (pd.to_datetime('2017-12-01').year - loan_data['earliest_cr_line_date'].dt.year) * 12 + (pd.to_datetime('2017-12-01').month - loan_data['earliest_cr_line_date'].dt.month)
    loan_data['mths_since_earliest_cr_line'][loan_data['mths_since_earliest_cr_line'] < 0] = loan_data['mths_since_earliest_cr_line'].max()

    ## dealing with term
    loan_data['term_int'] = pd.to_numeric(loan_data['term'].str.replace(' months', ''))

    ## dealing with issue_d
    loan_data['issue_d_date'] = pd.to_datetime(loan_data['issue_d'], format = '%b-%y')
    loan_data['mths_since_issue_d'] = (pd.to_datetime('2017-12-01').year - loan_data['issue_d_date'].dt.year)*12 + (pd.to_datetime('2017-12-01').month - loan_data['issue_d_date'].dt.month)

    return loan_data

def preprocessing_discrete_variables(loan_data):
    # creating dummies for categorical variables
    ## We create dummy variables from all 8 original independent variables, and save them into a list.

    variables = ['grade','sub_grade','home_ownership','verification_status',
                 'loan_status','purpose','addr_state','initial_list_status']
    
    loan_data_dummies = []
    for variable in variables:
        loan_data_dummies.append(pd.get_dummies(loan_data[variable], prefix = variable, prefix_sep = ':'))
    
    ## creating dataframe of list of dummies
    loan_data_dummies = pd.concat(loan_data_dummies, axis = 1)

    ## concating dummy variable columns with loan_data
    loan_data = pd.concat([loan_data, loan_data_dummies], axis = 1)
    
    return loan_data

# input data transformation
def customize_data_pd(input_variable_dict):
    variables = input_variable_dict.keys()

    # for PD Model
    ## here all the used features will be converted into dummy variables.
    ## list of vars to be used for PD: []
    pd_variables = [ 'grade',
                    'home_ownership',
                    'addr_state',
                    'verification_status',
                    'purpose',
                    'initial_list_status',
                    'term',
                    'emp_length',
                    'issue_d',
                    'int_rate',
                    'earliest_cr_line',
                    'inq_last_6mths',
                    'acc_now_delinq',
                    'annual_inc',
                    'dti',
                    'mths_since_last_delinq',
                    'mths_since_last_record']
    pd_features = []
    for feature in config.limited_input_features:
        for var in pd_variables:
            if var in feature:
                pd_features.append(feature)
    custom_data_input_dict = {key: 0 for key in pd_features}              

    # dealing with variables
    for var in variables:
        if var=='grade':
            if input_variable_dict['grade']=='A':
                custom_data_input_dict['grade:A'] = 1
            elif input_variable_dict['grade']=='B':
                custom_data_input_dict['grade:B'] = 1
            elif input_variable_dict['grade']=='C':
                custom_data_input_dict['grade:C'] = 1
            elif input_variable_dict['grade']=='D':
                custom_data_input_dict['grade:D'] = 1
            elif input_variable_dict['grade']=='E':
                custom_data_input_dict['grade:E'] = 1
            elif input_variable_dict['grade']=='F':
                custom_data_input_dict['grade:F'] = 1
            elif input_variable_dict['grade']=='G':
                custom_data_input_dict['grade:G'] = 1

        elif var=='home_ownership':
            if input_variable_dict['home_ownership'] in ['RENT','OTHER','NONE','ANY']:
                custom_data_input_dict['home_ownership:RENT_OTHER_NONE_ANY'] = 1
            elif input_variable_dict['home_ownership'] == 'OWN':
                custom_data_input_dict['home_ownership:OWN'] = 1
            elif input_variable_dict['home_ownership'] == 'MORTGAGE':
                custom_data_input_dict['home_ownership:MORTGAGE'] = 1

        elif var=='addr_state':
            if input_variable_dict['addr_state'] in ['ND','NE','IA','NV','FL','HI','AL']:
                custom_data_input_dict['addr_state:ND_NE_IA_NV_FL_HI_AL'] = 1
            elif input_variable_dict['addr_state'] in ['NY']:
                custom_data_input_dict['addr_state:NY'] = 1
            elif input_variable_dict['addr_state'] in ['OK','TN','MO','LA','MD','NC']:
                custom_data_input_dict['addr_state:OK_TN_MO_LA_MD_NC'] = 1
            elif input_variable_dict['addr_state'] in ['CA']:
                custom_data_input_dict['addr_state:CA'] = 1
            elif input_variable_dict['addr_state'] in ['UT','KY','AZ','NJ']:
                custom_data_input_dict['addr_state:UT_KY_AZ_NJ'] = 1
            elif input_variable_dict['addr_state'] in ['AR','MI','PA','OH','MN']:
                custom_data_input_dict['addr_state:AR_MI_PA_OH_MN'] = 1
            elif input_variable_dict['addr_state'] in ['RI','MA','DE','SD','IN']:
                custom_data_input_dict['addr_state:RI_MA_DE_SD_IN'] = 1
            elif input_variable_dict['addr_state'] in ['GA','WA','OR']:
                custom_data_input_dict['addr_state:GA_WA_OR'] = 1
            elif input_variable_dict['addr_state'] in ['WI','MT']:
                custom_data_input_dict['addr_state:WI_MT'] = 1
            elif input_variable_dict['addr_state'] in ['TX']:
                custom_data_input_dict['addr_state:TX'] = 1
            elif input_variable_dict['addr_state'] in ['IL','CT']:
                custom_data_input_dict['addr_state:IL_CT'] = 1
            elif input_variable_dict['addr_state'] in ['KS','SC','CO','VT','AK','MS']:
                custom_data_input_dict['addr_state:KS_SC_CO_VT_AK_MS'] = 1
            elif input_variable_dict['addr_state'] in ['WV','NH','WY','DC','ME','ID']:
                custom_data_input_dict['addr_state:WV_NH_WY_DC_ME_ID'] = 1

        elif var=='verification_status':
            if input_variable_dict['verification_status']=='Not Verified':
                custom_data_input_dict['verification_status:Not Verified'] = 1
            elif input_variable_dict['verification_status']=='Source Verified':
                custom_data_input_dict['verification_status:Source Verified'] = 1
            elif input_variable_dict['verification_status']=='Verified':
                custom_data_input_dict['verification_status:Verified'] = 1

        elif var=='purpose':
            if input_variable_dict['purpose'] in ['educational','small_business','wedding','renewable_energy','moving','house']:
                custom_data_input_dict['purpose:educ__sm_b__wedd__ren_en__mov__house'] = 1
            elif input_variable_dict['purpose'] in ['other','medical','vacation']:
                custom_data_input_dict['purpose:oth__med__vacation'] = 1
            elif input_variable_dict['purpose'] in ['major_purchase','car','home_improvement']:
                custom_data_input_dict['purpose:major_purch__car__home_impr'] = 1

        elif var=='initial_list_status':
            if input_variable_dict['initial_list_status'] == 'f':
                custom_data_input_dict['initial_list_status:f'] = 1
            elif input_variable_dict['initial_list_status'] == 'w':
                custom_data_input_dict['initial_list_status:w'] = 1

        elif var=='term':
            if input_variable_dict['term'] == '36':
                custom_data_input_dict['term:f'] = 1
            elif input_variable_dict['term'] == '60':
                custom_data_input_dict['term:w'] = 1

        elif var=='emp_length':
            if input_variable_dict['emp_length'] == 0:
                custom_data_input_dict['emp_length:0'] = 1
            elif input_variable_dict['emp_length'] == 1:
                custom_data_input_dict['emp_length:1'] = 1
            elif input_variable_dict['emp_length'] in [2,5]:
                custom_data_input_dict['emp_length:2-4'] = 1
            elif input_variable_dict['emp_length'] in [5,7]:
                custom_data_input_dict['emp_length:5-6'] = 1
            elif input_variable_dict['emp_length'] in [7,10]:
                custom_data_input_dict['emp_length:7-9'] = 1
            elif input_variable_dict['emp_length'] == 10:
                custom_data_input_dict['emp_length:10'] = 1

        if var=='issue_d':
            issue_date = datetime.strptime(input_variable_dict[var], "%Y-%m-%d")
            end_date = datetime.strptime('2017-12-01', "%Y-%m-%d")
            mths_since_issue_d = end_date.year * 12 + end_date.month - (issue_date.year * 12 + issue_date.month)
            if mths_since_issue_d < 38:
                custom_data_input_dict['mths_since_issue_d:<38'] = 1
            elif mths_since_issue_d >= 38 and mths_since_issue_d < 40:
                custom_data_input_dict['mths_since_issue_d:38-39'] = 1
            elif mths_since_issue_d >= 40 and mths_since_issue_d < 42:
                custom_data_input_dict['mths_since_issue_d:40-41'] = 1
            elif mths_since_issue_d >= 42 and mths_since_issue_d < 49:
                custom_data_input_dict['mths_since_issue_d:42-48'] = 1    
            elif mths_since_issue_d >= 49 and mths_since_issue_d < 53:
                custom_data_input_dict['mths_since_issue_d:49-52'] = 1    
            elif mths_since_issue_d >= 53 and mths_since_issue_d < 65:
                custom_data_input_dict['mths_since_issue_d:53-64'] = 1    
            elif mths_since_issue_d >= 65 and mths_since_issue_d < 85:
                custom_data_input_dict['mths_since_issue_d:65-84'] = 1    
            elif mths_since_issue_d >= 84:
                custom_data_input_dict['mths_since_issue_d:>84'] = 1

        if var=='int_rate':
            int_rate = input_variable_dict[var]
            if int_rate <= 9.548:
                custom_data_input_dict['int_rate:<9.548'] = 1
            elif int_rate > 9.548 and int_rate <= 12.025:
                custom_data_input_dict['int_rate:9.548-12.025'] = 1
            elif int_rate > 12.025 and int_rate <= 15.74:
                custom_data_input_dict['int_rate:12.025-15.74'] = 1
            elif int_rate > 15.74 and int_rate <= 20.281:
                custom_data_input_dict['int_rate:15.74-20.281'] = 1
            elif int_rate >= 20.281:
                custom_data_input_dict['int_rate:>20.281'] = 1

        if var=='earliest_cr_line':
            issue_date = datetime.strptime(input_variable_dict[var], "%Y-%m-%d")
            end_date = datetime.strptime('2017-12-01', "%Y-%m-%d")
            mths_since_issue_d = end_date.year * 12 + end_date.month - (issue_date.year * 12 + issue_date.month)
            if mths_since_issue_d < 140:
                custom_data_input_dict['mths_since_earliest_cr_line:<140'] = 1
            elif mths_since_issue_d >= 140 and mths_since_issue_d < 165:
                custom_data_input_dict['mths_since_earliest_cr_line:141-164'] = 1
            elif mths_since_issue_d >= 165 and mths_since_issue_d < 248:
                custom_data_input_dict['mths_since_earliest_cr_line:165-247'] = 1
            elif mths_since_issue_d >= 248 and mths_since_issue_d < 271:
                custom_data_input_dict['mths_since_earliest_cr_line:248-270'] = 1    
            elif mths_since_issue_d >= 271 and mths_since_issue_d < 353:
                custom_data_input_dict['mths_since_earliest_cr_line:271-352'] = 1    
            elif mths_since_issue_d >= 353:
                custom_data_input_dict['mths_since_earliest_cr_line:>352'] = 1

        if var=='inq_last_6mths':
            if input_variable_dict[var]==0:
                custom_data_input_dict['inq_last_6mths:0'] = 1
            elif input_variable_dict[var] in [1,3]:
                custom_data_input_dict['inq_last_6mths:1-2'] = 1
            elif input_variable_dict[var] in [3,7]:
                custom_data_input_dict['inq_last_6mths:3-6'] = 1
            elif input_variable_dict[var]>6:
                custom_data_input_dict['inq_last_6mths:>6'] = 1

        if var=='acc_now_delinq':
            if input_variable_dict[var]==0:
                custom_data_input_dict['acc_now_delinq:0'] = 1
            elif input_variable_dict[var]>=1:
                custom_data_input_dict['acc_now_delinq:>=1'] = 1

        if var=='annual_inc':
            annual_inc = input_variable_dict[var]
            if annual_inc <= 20000:
                custom_data_input_dict['annual_inc:<20K'] = 1
            elif annual_inc > 20000 and annual_inc <= 30000:
                custom_data_input_dict['annual_inc:20K-30K'] = 1
            elif annual_inc > 30000 and annual_inc <= 40000:
                custom_data_input_dict['annual_inc:30K-40K'] = 1
            elif annual_inc > 40000 and annual_inc <= 50000:
                custom_data_input_dict['annual_inc:40K-50K'] = 1
            elif annual_inc > 50000 and annual_inc <= 60000:
                custom_data_input_dict['annual_inc:50K-60K'] = 1
            elif annual_inc > 60000 and annual_inc <= 70000:
                custom_data_input_dict['annual_inc:60K-70K'] = 1
            elif annual_inc > 70000 and annual_inc <= 80000:
                custom_data_input_dict['annual_inc:70K-80K'] = 1
            elif annual_inc > 80000 and annual_inc <= 90000:
                custom_data_input_dict['annual_inc:80K-90K'] = 1
            elif annual_inc > 90000 and annual_inc <= 100000:
                custom_data_input_dict['annual_inc:90K-100K'] = 1
            elif annual_inc > 100000 and annual_inc <= 120000:
                custom_data_input_dict['annual_inc:100K-120K'] = 1
            elif annual_inc > 120000 and annual_inc <= 140000:
                custom_data_input_dict['annual_inc:120K-140K'] = 1
            elif annual_inc > 140000:
                custom_data_input_dict['annual_inc:>140K'] = 1

        if var=='dti':
            if input_variable_dict[var]<=1.4:
                custom_data_input_dict['dti:<=1.4'] = 1
            elif input_variable_dict[var]>1.4 and input_variable_dict[var]<=3.5:
                custom_data_input_dict['dti:1.4-3.5'] = 1
            elif input_variable_dict[var]>3.5 and input_variable_dict[var]<=7.7:
                custom_data_input_dict['dti:3.5-7.7'] = 1
            elif input_variable_dict[var]>7.7 and input_variable_dict[var]<=10.5:
                custom_data_input_dict['dti:7.7-10.5'] = 1
            elif input_variable_dict[var]>10.5 and input_variable_dict[var]<=16.1:
                custom_data_input_dict['dti:10.5-16.1'] = 1
            elif input_variable_dict[var]>16.1 and input_variable_dict[var]<=20.3:
                custom_data_input_dict['dti:16.1-20.3'] = 1
            elif input_variable_dict[var]>20.3 and input_variable_dict[var]<=21.7:
                custom_data_input_dict['dti:20.3-21.7'] = 1
            elif input_variable_dict[var]>21.7 and input_variable_dict[var]<=22.4:
                custom_data_input_dict['dti:21.7-22.4'] = 1
            elif input_variable_dict[var]>22.4 and input_variable_dict[var]<=35:
                custom_data_input_dict['dti:22.4-35'] = 1
            elif input_variable_dict[var]>35:
                custom_data_input_dict['dti:>35'] = 1

        if var=='mths_since_last_delinq':
            if input_variable_dict[var] is None:
                custom_data_input_dict['mths_since_last_delinq:Missing'] = 1
            elif input_variable_dict[var]>=0 and input_variable_dict[var]<=3:
                custom_data_input_dict['mths_since_last_delinq:0-3'] = 1
            elif input_variable_dict[var]>=4 and input_variable_dict[var]<=30:
                custom_data_input_dict['mths_since_last_delinq:4-30'] = 1
            elif input_variable_dict[var]>=31 and input_variable_dict[var]<=56:
                custom_data_input_dict['mths_since_last_delinq:31-56'] = 1
            elif input_variable_dict[var]>=57:
                custom_data_input_dict['mths_since_last_delinq:>=57'] = 1

        if var=='mths_since_last_record':
            if input_variable_dict[var] is None:
                custom_data_input_dict['mths_since_last_record:Missing'] = 1
            elif input_variable_dict[var]>=0 and input_variable_dict[var]<=2:
                custom_data_input_dict['mths_since_last_record:0-2'] = 1
            elif input_variable_dict[var]>=3 and input_variable_dict[var]<=20:
                custom_data_input_dict['mths_since_last_record:3-20'] = 1
            elif input_variable_dict[var]>=21 and input_variable_dict[var]<=31:
                custom_data_input_dict['mths_since_last_record:21-31'] = 1
            elif input_variable_dict[var]>=32 and input_variable_dict[var]<=80:
                custom_data_input_dict['mths_since_last_record:32-80'] = 1
            elif input_variable_dict[var]>=81 and input_variable_dict[var]<=86:
                custom_data_input_dict['mths_since_last_record:81-86'] = 1
            elif input_variable_dict[var]>86:
                custom_data_input_dict['mths_since_last_record:>86'] = 1

    # remove referece variables
    remove_keys = []
    for key in custom_data_input_dict.keys():
        if key in config.ref_categories:
            remove_keys.append(key)
    for key in remove_keys:
        if key in custom_data_input_dict:
            del custom_data_input_dict[key]

    return pd.DataFrame(custom_data_input_dict,index=[0]).copy()

def customize_data_lgd(input_variable_dict):
    variables = input_variable_dict.keys()

    # variables for lgd, ead n el model
    lgd_variables =['grade',
                    'home_ownership',
                    'verification_status',
                    'purpose',     
                    'initial_list_status',
                    'term_int',  # check
                    'emp_length_int',  # check
                    'issue_d',
                    'earliest_cr_line',
                    'funded_amnt',
                    'int_rate',
                    'installment',
                    'annual_inc',
                    'dti',
                    'delinq_2yrs',
                    'inq_last_6mths',
                    'mths_since_last_delinq',
                    'mths_since_last_record',
                    'open_acc',
                    'pub_rec',
                    'total_acc',
                    'acc_now_delinq',
                    'total_rev_hi_lim']
    lgd_features = config.features_all
    custom_data_input_dict = {key: 0 for key in lgd_features}

    # dealing with variables
    for var in variables:
        if var=='grade':
            if input_variable_dict['grade']=='A':
                custom_data_input_dict['grade:A'] = 1
            elif input_variable_dict['grade']=='B':
                custom_data_input_dict['grade:B'] = 1
            elif input_variable_dict['grade']=='C':
                custom_data_input_dict['grade:C'] = 1
            elif input_variable_dict['grade']=='D':
                custom_data_input_dict['grade:D'] = 1
            elif input_variable_dict['grade']=='E':
                custom_data_input_dict['grade:E'] = 1
            elif input_variable_dict['grade']=='F':
                custom_data_input_dict['grade:F'] = 1
            elif input_variable_dict['grade']=='G':
                custom_data_input_dict['grade:G'] = 1

        elif var=='home_ownership':
            if input_variable_dict['home_ownership'] == 'OWN':
                custom_data_input_dict['home_ownership:OWN'] = 1
            elif input_variable_dict['home_ownership'] == 'MORTGAGE':
                custom_data_input_dict['home_ownership:MORTGAGE'] = 1
            elif input_variable_dict['home_ownership'] == 'RENT':
                custom_data_input_dict['home_ownership:RENT'] = 1
            elif input_variable_dict['home_ownership'] == 'OTHER':
                custom_data_input_dict['home_ownership:OTHER'] = 1
            elif input_variable_dict['home_ownership'] == 'NONE':
                custom_data_input_dict['home_ownership:NONE'] = 1

        elif var=='verification_status':
            if input_variable_dict['verification_status']=='Not Verified':
                custom_data_input_dict['verification_status:Not Verified'] = 1
            elif input_variable_dict['verification_status']=='Source Verified':
                custom_data_input_dict['verification_status:Source Verified'] = 1
            elif input_variable_dict['verification_status']=='Verified':
                custom_data_input_dict['verification_status:Verified'] = 1

        elif var=='purpose':
            if input_variable_dict['purpose']=='car':
                custom_data_input_dict['purpose:car'] = 1
            elif input_variable_dict['purpose']=='credit_card':
                custom_data_input_dict['purpose:credit_card'] = 1
            elif input_variable_dict['purpose']=='debt_consolidation':
                custom_data_input_dict['purpose:debt_consolidation'] = 1
            elif input_variable_dict['purpose']=='educational':
                custom_data_input_dict['purpose:educational'] = 1
            elif input_variable_dict['purpose']=='home_improvement':
                custom_data_input_dict['purpose:home_improvement'] = 1
            elif input_variable_dict['purpose']=='house':
                custom_data_input_dict['purpose:house'] = 1
            elif input_variable_dict['purpose']=='major_purchase':
                custom_data_input_dict['purpose:major_purchase'] = 1
            elif input_variable_dict['purpose']=='medical':
                custom_data_input_dict['purpose:medical'] = 1
            elif input_variable_dict['purpose']=='moving':
                custom_data_input_dict['purpose:moving'] = 1
            elif input_variable_dict['purpose']=='other':
                custom_data_input_dict['purpose:other'] = 1
            elif input_variable_dict['purpose']=='renewable_energy':
                custom_data_input_dict['purpose:renewable_energy'] = 1
            elif input_variable_dict['purpose']=='small_business':
                custom_data_input_dict['purpose:small_business'] = 1
            elif input_variable_dict['purpose']=='vacation':
                custom_data_input_dict['purpose:vacation'] = 1
            elif input_variable_dict['purpose']=='wedding':
                custom_data_input_dict['purpose:wedding'] = 1

        elif var=='initial_list_status':
            if input_variable_dict['initial_list_status'] == 'f':
                custom_data_input_dict['initial_list_status:f'] = 1
            elif input_variable_dict['initial_list_status'] == 'w':
                custom_data_input_dict['initial_list_status:w'] = 1

        elif var=='term_int':
            custom_data_input_dict['term_int'] = input_variable_dict[var]

        elif var=='emp_length_int':
            custom_data_input_dict['emp_length_int'] = input_variable_dict[var]

        elif var=='mths_since_issue_d':
            custom_data_input_dict['mths_since_issue_d'] = input_variable_dict[var]

        elif var=='mths_since_earliest_cr_line':
            custom_data_input_dict['mths_since_earliest_cr_line'] = input_variable_dict[var]

        elif var=='funded_amnt':
            custom_data_input_dict['funded_amnt'] = input_variable_dict[var]

        elif var=='int_rate':
            custom_data_input_dict['int_rate'] = input_variable_dict[var]

        elif var=='installment':
            custom_data_input_dict['installment'] = input_variable_dict[var]

        elif var=='annual_inc':
            custom_data_input_dict['annual_inc'] = input_variable_dict[var]

        elif var=='dti':
            custom_data_input_dict['dti'] = input_variable_dict[var]

        elif var=='delinq_2yrs':
            custom_data_input_dict['delinq_2yrs'] = input_variable_dict[var]

        elif var=='inq_last_6mths':
            custom_data_input_dict['inq_last_6mths'] = input_variable_dict[var]

        elif var=='mths_since_last_delinq':
            custom_data_input_dict['mths_since_last_delinq'] = input_variable_dict[var]

        elif var=='mths_since_last_record':
            custom_data_input_dict['mths_since_last_record'] = input_variable_dict[var]

        elif var=='open_acc':
            custom_data_input_dict['open_acc'] = input_variable_dict[var]

        elif var=='pub_rec':
            custom_data_input_dict['pub_rec'] = input_variable_dict[var]

        elif var=='total_acc':
            custom_data_input_dict['total_acc'] = input_variable_dict[var]

        elif var=='acc_now_delinq':
            custom_data_input_dict['acc_now_delinq'] = input_variable_dict[var]

        elif var=='total_rev_hi_lim':
            custom_data_input_dict['total_rev_hi_lim'] = input_variable_dict[var]

    # remove reference variables
    remove_keys = []
    for key in custom_data_input_dict.keys():
        if key in config.features_reference_cat:
            remove_keys.append(key)
    for key in remove_keys:
        if key in custom_data_input_dict:
            del custom_data_input_dict[key]

    return pd.DataFrame(custom_data_input_dict,index=[0]).copy()

# input data transformation
def customdata(grade,issue_d,int_rate,annual_inc):

    ## getting all dummy featuers of filtered variables
    iv_variables = ['grade','mths_since_issue_d','int_rate','annual_inc']
    iv_features = []
    for feature in config.limited_input_features:
        for var in iv_variables:
            if var in feature:
                iv_features.append(feature)

    custom_data_input_dict = {key: 0 for key in iv_features}

    # dealing with grade
    if grade=='A':
        custom_data_input_dict['grade:A'] = 1
    elif grade=='B':
        custom_data_input_dict['grade:B'] = 1
    elif grade=='C':
        custom_data_input_dict['grade:C'] = 1
    elif grade=='D':
        custom_data_input_dict['grade:D'] = 1
    elif grade=='E':
        custom_data_input_dict['grade:E'] = 1
    elif grade=='F':
        custom_data_input_dict['grade:F'] = 1
    elif grade=='G':
        custom_data_input_dict['grade:G'] = 1

    # dealing with issue date
    # print(issue_d)
    issue_date = datetime.strptime(issue_d, "%Y-%m-%d")
    end_date = datetime.strptime('2017-12-01', "%Y-%m-%d")
    mths_since_issue_d = end_date.year * 12 + end_date.month - (issue_date.year * 12 + issue_date.month)
    if mths_since_issue_d < 38:
        custom_data_input_dict['mths_since_issue_d:<38'] = 1
    elif mths_since_issue_d >= 38 and mths_since_issue_d < 40:
        custom_data_input_dict['mths_since_issue_d:38-39'] = 1
    elif mths_since_issue_d >= 40 and mths_since_issue_d < 42:
        custom_data_input_dict['mths_since_issue_d:40-41'] = 1
    elif mths_since_issue_d >= 42 and mths_since_issue_d < 49:
        custom_data_input_dict['mths_since_issue_d:42-48'] = 1    
    elif mths_since_issue_d >= 49 and mths_since_issue_d < 53:
        custom_data_input_dict['mths_since_issue_d:49-52'] = 1    
    elif mths_since_issue_d >= 53 and mths_since_issue_d < 65:
        custom_data_input_dict['mths_since_issue_d:53-64'] = 1    
    elif mths_since_issue_d >= 65 and mths_since_issue_d < 85:
        custom_data_input_dict['mths_since_issue_d:65-84'] = 1    
    elif mths_since_issue_d >= 84:
        custom_data_input_dict['mths_since_issue_d:>84'] = 1

    # dealing with int_rate
    if int_rate <= 9.548:
        custom_data_input_dict['int_rate:<9.548'] = 1
    elif int_rate > 9.548 and int_rate <= 12.025:
        custom_data_input_dict['int_rate:9.548-12.025'] = 1
    elif int_rate > 12.025 and int_rate <= 15.74:
        custom_data_input_dict['int_rate:12.025-15.74'] = 1
    elif int_rate > 15.74 and int_rate <= 20.281:
        custom_data_input_dict['int_rate:15.74-20.281'] = 1
    elif int_rate >= 20.281:
        custom_data_input_dict['int_rate:>20.281'] = 1

    # dealing with annual_inc
    if annual_inc <= 20000:
        custom_data_input_dict['annual_inc:<20K'] = 1
    elif annual_inc > 20000 and annual_inc <= 30000:
        custom_data_input_dict['annual_inc:20K-30K'] = 1
    elif annual_inc > 30000 and annual_inc <= 40000:
        custom_data_input_dict['annual_inc:30K-40K'] = 1
    elif annual_inc > 40000 and annual_inc <= 50000:
        custom_data_input_dict['annual_inc:40K-50K'] = 1
    elif annual_inc > 50000 and annual_inc <= 60000:
        custom_data_input_dict['annual_inc:50K-60K'] = 1
    elif annual_inc > 60000 and annual_inc <= 70000:
        custom_data_input_dict['annual_inc:60K-70K'] = 1
    elif annual_inc > 70000 and annual_inc <= 80000:
        custom_data_input_dict['annual_inc:70K-80K'] = 1
    elif annual_inc > 80000 and annual_inc <= 90000:
        custom_data_input_dict['annual_inc:80K-90K'] = 1
    elif annual_inc > 90000 and annual_inc <= 100000:
        custom_data_input_dict['annual_inc:90K-100K'] = 1
    elif annual_inc > 100000 and annual_inc <= 120000:
        custom_data_input_dict['annual_inc:100K-120K'] = 1
    elif annual_inc > 120000 and annual_inc <= 140000:
        custom_data_input_dict['annual_inc:120K-140K'] = 1
    elif annual_inc > 140000:
        custom_data_input_dict['annual_inc:>140K'] = 1

    # remove referece variables
    remove_keys = []
    for key in custom_data_input_dict.keys():
        if key in config.ref_categories:
            remove_keys.append(key)
    for key in remove_keys:
        if key in custom_data_input_dict:
            del custom_data_input_dict[key]

    return pd.DataFrame(custom_data_input_dict,index=[0]).copy()


def deal_with_missing_values(loan_data):
    # dealing with missing values
    loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'], inplace=True)
    loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean(), inplace=True)
    loan_data['mths_since_earliest_cr_line'].fillna(0, inplace=True)
    loan_data['acc_now_delinq'].fillna(0, inplace=True)
    loan_data['total_acc'].fillna(0, inplace=True)
    loan_data['pub_rec'].fillna(0, inplace=True)
    loan_data['open_acc'].fillna(0, inplace=True)
    loan_data['inq_last_6mths'].fillna(0, inplace=True)
    loan_data['delinq_2yrs'].fillna(0, inplace=True)
    loan_data['emp_length_int'].fillna(0, inplace=True)
    loan_data['mths_since_last_delinq'].fillna(0, inplace = True)
    loan_data['mths_since_last_record'].fillna(0, inplace=True)

    return loan_data

# creating target variable using good/bad definition
def target_var(loan_data):
    # Good/ Bad Definition
    loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(['Charged Off', 'Default',
                                                                    'Late (31-120 days)',
                                                                    'Does not meet the credit policy. Status:Charged Off']), 0, 1)
    # 0 means default and 1 means non-default

    return loan_data

# splitting data
def splitting_data(loan_data):
    loan_data_inputs_train, loan_data_inputs_test, \
    loan_data_targets_train, loan_data_targets_test = train_test_split(loan_data.drop('good_bad', axis = 1), 
                                                                       loan_data['good_bad'], 
                                                                       test_size = 0.2, 
                                                                       random_state = 42)

    return loan_data_inputs_train, loan_data_inputs_test, loan_data_targets_train, loan_data_targets_test

# WoE function for discrete unordered variables
def woe_discrete(df,discrete_variable_name,good_bad_variable_df):
    '''
    inputs:
        df: complete train datafrane.
        discrete_variable_name: name of the discrete variable for which WoE and IV will be calculated.
        good_bad_variable_df: dataframe with target column.
    outputs:
        returns dataframe with some calculated columns for the discrete variable.
        calculated columns: WoE, IV.
    '''
    # 
    df = pd.concat([df[discrete_variable_name], good_bad_variable_df], axis = 1)
    # calculating count and mean for every category of the variable.
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)   

    # Selects only columns with specific indexes.
    df = df.iloc[:, [0, 1, 3]]

    # Changes the names of the columns of a dataframe.
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']

    # Inserting the column at the beginning in the DataFrame
    df.insert(loc = 0, column = 'variable',value = discrete_variable_name)

    # proportion of number of observation
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()

    # calculating number of good and number of bad
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']

    # calculating proportion of good and bads
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()

    # calculating WoE and sorting by it
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop = True)

    # diff() calculates difference between two subsequent values of a column
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()

    # calculating IV
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()

    df.rename(columns = {discrete_variable_name:'var_sub_cat'}, inplace = True)

    return df

# for a variable woe visualization function
def plot_by_woe(df_WoE, variable_name,n=0, rotation_of_x_axis_labels = 90):
    # Turns the values of the column with index 0 to strings, makes an array from these strings, and passes it to variable x.
    x = np.array(df_WoE.iloc[:, 1].apply(str))
    y = df_WoE['WoE']

    # Sets the graph size to width 18 x height 6.
    plt.figure(figsize=(18, 6))

    # Sets the marker for each datapoint to a circle, the style line between the points to dashed, and the color to black.
    plt.plot(x, y, marker = 'o', linestyle = '--', color = 'k')

    plt.xlabel(variable_name)    
    plt.ylabel('Weight of Evidence')
    plt.title(str('Weight of Evidence by ' + variable_name))
    plt.xticks(rotation = rotation_of_x_axis_labels)

    # Saving the figure.
    plt.savefig(r"C:\Shubham\credit_scorecard\output_data\graphs\\"+variable_name+str(n)+".jpg")
    plt.close()

# preprocessing discrete variables
def binning_discrete_variables(df_inputs_prepr,df_targets_prepr):

    df_with_woe_n_iv_values = pd.DataFrame()

    # calculating WoE and IV
    discrete_variables = ['grade',
                          'home_ownership',
                          'addr_state',
                          'verification_status',
                          'purpose',
                          'initial_list_status']
    
    # doing preprocessing one-by-one
    for variable in discrete_variables:

        if variable=='addr_state':
            # creating column for those customers which have zero customers in our dataset
            if ['addr_state:ND'] in df_inputs_prepr.columns.values:
                pass
            else:
                df_inputs_prepr['addr_state:ND'] = 0
            if ['addr_state:ID'] in df_inputs_prepr.columns.values:
                pass
            else:
                df_inputs_prepr['addr_state:ID'] = 0
            if ['addr_state:IA'] in df_inputs_prepr.columns.values:
                pass
            else:
                df_inputs_prepr['addr_state:IA'] = 0

        if variable=='home_ownership':
            try:
                df_inputs_prepr['home_ownership:RENT_OTHER_NONE_ANY'] = sum([df_inputs_prepr['home_ownership:RENT'],
                                                                            df_inputs_prepr['home_ownership:OTHER'],
                                                                            df_inputs_prepr['home_ownership:NONE'],
                                                                            df_inputs_prepr['home_ownership:ANY']])
            except:
                df_inputs_prepr['home_ownership:RENT_OTHER_NONE_ANY'] = sum([df_inputs_prepr['home_ownership:RENT'],
                                                                            df_inputs_prepr['home_ownership:ANY']])
            # df_inputs_prepr.drop(['home_ownership:RENT','home_ownership:ANY'],inplace=True)

        df_temp = woe_discrete(df_inputs_prepr,variable,df_targets_prepr)
        plot_by_woe(df_temp,variable)
        df_with_woe_n_iv_values = pd.concat([df_with_woe_n_iv_values,df_temp],axis=0)
        
        if variable=='addr_state':
            plot_by_woe(df_temp.iloc[2: -2, : ],variable,1)
            plot_by_woe(df_temp.iloc[6: -6, : ],variable,2)

            # grouping
            ## include those categories about which we do not have any information into WORST category.
            ## include those categories which have no bad customers into BEST category

            df_inputs_prepr['addr_state:ND_NE_IA_NV_FL_HI_AL'] = sum([df_inputs_prepr['addr_state:ND'],
                                                                      df_inputs_prepr['addr_state:NE'],
                                                                      df_inputs_prepr['addr_state:IA'],
                                                                      df_inputs_prepr['addr_state:NV'],
                                                                      df_inputs_prepr['addr_state:FL'],
                                                                      df_inputs_prepr['addr_state:HI'],
                                                                      df_inputs_prepr['addr_state:AL']])

            df_inputs_prepr['addr_state:NM_VA'] = sum([df_inputs_prepr['addr_state:NM'],
                                                       df_inputs_prepr['addr_state:VA']])

            df_inputs_prepr['addr_state:OK_TN_MO_LA_MD_NC'] = sum([df_inputs_prepr['addr_state:OK'],
                                                                   df_inputs_prepr['addr_state:TN'],
                                                                   df_inputs_prepr['addr_state:MO'],
                                                                   df_inputs_prepr['addr_state:LA'],
                                                                   df_inputs_prepr['addr_state:MD'],
                                                                   df_inputs_prepr['addr_state:NC']])

            df_inputs_prepr['addr_state:UT_KY_AZ_NJ'] = sum([df_inputs_prepr['addr_state:UT'],
                                                             df_inputs_prepr['addr_state:KY'],
                                                             df_inputs_prepr['addr_state:AZ'],
                                                             df_inputs_prepr['addr_state:NJ']])

            df_inputs_prepr['addr_state:AR_MI_PA_OH_MN'] = sum([df_inputs_prepr['addr_state:AR'],
                                                                df_inputs_prepr['addr_state:MI'],
                                                                df_inputs_prepr['addr_state:PA'],
                                                                df_inputs_prepr['addr_state:OH'],
                                                                df_inputs_prepr['addr_state:MN']])

            df_inputs_prepr['addr_state:RI_MA_DE_SD_IN'] = sum([df_inputs_prepr['addr_state:RI'],
                                                                df_inputs_prepr['addr_state:MA'],
                                                                df_inputs_prepr['addr_state:DE'],
                                                                df_inputs_prepr['addr_state:SD'],
                                                                df_inputs_prepr['addr_state:IN']])

            df_inputs_prepr['addr_state:GA_WA_OR'] = sum([df_inputs_prepr['addr_state:GA'],
                                                          df_inputs_prepr['addr_state:WA'],
                                                          df_inputs_prepr['addr_state:OR']])

            df_inputs_prepr['addr_state:WI_MT'] = sum([df_inputs_prepr['addr_state:WI'],
                                                       df_inputs_prepr['addr_state:MT']])

            df_inputs_prepr['addr_state:IL_CT'] = sum([df_inputs_prepr['addr_state:IL'],
                                                       df_inputs_prepr['addr_state:CT']])

            df_inputs_prepr['addr_state:KS_SC_CO_VT_AK_MS'] = sum([df_inputs_prepr['addr_state:KS'],
                                                                   df_inputs_prepr['addr_state:SC'],
                                                                   df_inputs_prepr['addr_state:CO'],
                                                                   df_inputs_prepr['addr_state:VT'],
                                                                   df_inputs_prepr['addr_state:AK'],
                                                                   df_inputs_prepr['addr_state:MS']])

            df_inputs_prepr['addr_state:WV_NH_WY_DC_ME_ID'] = sum([df_inputs_prepr['addr_state:WV'],
                                                                   df_inputs_prepr['addr_state:NH'],
                                                                   df_inputs_prepr['addr_state:WY'],
                                                                   df_inputs_prepr['addr_state:DC'],
                                                                   df_inputs_prepr['addr_state:ME'],
                                                                   df_inputs_prepr['addr_state:ID']])

        if variable=='purpose':
            df_inputs_prepr['purpose:educ__sm_b__wedd__ren_en__mov__house'] = sum([df_inputs_prepr['purpose:educational'],
                                                                                   df_inputs_prepr['purpose:small_business'],
                                                                                   df_inputs_prepr['purpose:wedding'],
                                                                                   df_inputs_prepr['purpose:renewable_energy'],
                                                                                   df_inputs_prepr['purpose:moving'],
                                                                                   df_inputs_prepr['purpose:house']])
            
            df_inputs_prepr['purpose:oth__med__vacation'] = sum([df_inputs_prepr['purpose:other'],
                                                                 df_inputs_prepr['purpose:medical'],
                                                                 df_inputs_prepr['purpose:vacation']])
            
            df_inputs_prepr['purpose:major_purch__car__home_impr'] = sum([df_inputs_prepr['purpose:major_purchase'],
                                                                          df_inputs_prepr['purpose:car'],
                                                                          df_inputs_prepr['purpose:home_improvement']])
            
        # Reference Categories
        ## 'grade' : 'G' will be the reference category.
        ## 'home_ownership' : 'RENT_OTHER_NONE_ANY' will be the reference category.
        ## 'addr_state' : 'ID_IA_SD_NV_HI_AK' will be the reference category.
        ## 'verification_status' : 'Verified' will be the reference category.
        ## 'purpose' : 'sm_b__ren_en__mov__house' will be the reference category.
        ## 'initial_list_status' : 'f' will be the reference category.

    return df_inputs_prepr, df_with_woe_n_iv_values

# WoE function for ordered discrete and continuous variables
def woe_continuous(df, discrete_variable_name, good_bad_variable_df):
    df = pd.concat([df[discrete_variable_name], good_bad_variable_df], axis = 1)
    df = pd.concat([df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].count(),
                    df.groupby(df.columns.values[0], as_index = False)[df.columns.values[1]].mean()], axis = 1)
    df = df.iloc[:, [0, 1, 3]]

    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    #df = df.sort_values(['WoE'])
    #df = df.reset_index(drop = True)
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_WoE'] = df['WoE'].diff().abs()
    df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    
    # Inserting the column at the beginning in the DataFrame
    df.insert(loc = 0, column = 'variable',value = discrete_variable_name)
    df.rename(columns = {discrete_variable_name:'var_sub_cat'}, inplace = True)

    return df

# preprocessing continous variables
def binning_continous_variables(df_inputs_prepr,df_targets_prepr):

    df_with_woe_n_iv_values = pd.DataFrame()

    # calculating WoE and IV
    continous_variables = ['term_int',
                           'emp_length_int',
                           'mths_since_issue_d',
                           'int_rate',
                           'mths_since_earliest_cr_line',
                           'delinq_2yrs',
                           'inq_last_6mths',
                           'open_acc',
                           'pub_rec',
                           'total_acc',
                           'acc_now_delinq',
                           'total_rev_hi_lim',
                           'annual_inc',
                           'dti',
                           'mths_since_last_delinq',
                           'mths_since_last_record',
                           'funded_amnt',
                           'installment']
    
    # doing preprocessing one-by-one
    for variable in continous_variables:

        temp = 0

        if variable=='mths_since_issue_d':
            df_inputs_prepr['mths_since_issue_d_factor'] = pd.cut(df_inputs_prepr['mths_since_issue_d'], 50)
            variable = 'mths_since_issue_d_factor'

        if variable=='int_rate':
            df_inputs_prepr['int_rate_factor'] = pd.cut(df_inputs_prepr['int_rate'], 50)
            variable = 'int_rate_factor'

        if variable=='mths_since_earliest_cr_line':
            df_inputs_prepr['mths_since_earliest_cr_line_factor'] = pd.cut(df_inputs_prepr['mths_since_earliest_cr_line'], 50)
            variable = 'mths_since_earliest_cr_line_factor'

        if variable=='total_acc':
            df_inputs_prepr['total_acc_factor'] = pd.cut(df_inputs_prepr['total_acc'], 50)
            variable = 'total_acc_factor'

        if variable=='total_rev_hi_lim':
            df_inputs_prepr['total_rev_hi_lim_factor'] = pd.cut(df_inputs_prepr['total_rev_hi_lim'], 2000)
            variable = 'total_rev_hi_lim_factor'

        if variable=='annual_inc':
            # Initial examination shows that there are too few individuals with large income and too many with small income.
            # Hence, we are going to have one category for more than 150K, and we are going to apply our approach to determine
            # the categories of everyone with 140k or less.
            temp = 1
            df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['annual_inc'] <= 140000, : ]
            df_inputs_prepr_temp["annual_inc_factor"] = pd.cut(df_inputs_prepr_temp['annual_inc'], 50)
            variable = 'annual_inc_factor'

        if variable=='dti':
            # Similarly to income, initial examination shows that most values are lower than 200.
            # Hence, we are going to have one category for more than 35, and we are going to apply our approach to determine
            # the categories of everyone with 150k or less.
            temp = 1
            df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['dti'] <= 35, : ]
            df_inputs_prepr_temp['dti_factor'] = pd.cut(df_inputs_prepr_temp['dti'], 50)
            variable = 'dti_factor'

        if variable=='mths_since_last_delinq':
            # We have to create one category for missing values and do fine and coarse classing for the rest.
            temp = 1
            df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(df_inputs_prepr['mths_since_last_delinq'])]
            df_inputs_prepr_temp['mths_since_last_delinq_factor'] = pd.cut(df_inputs_prepr_temp['mths_since_last_delinq'], 50)
            variable = 'mths_since_last_delinq_factor'

        if variable=='mths_since_last_record':
            # We have to create one category for missing values and do fine and coarse classing for the rest.
            temp = 1
            df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(df_inputs_prepr['mths_since_last_record'])]
            df_inputs_prepr_temp['mths_since_last_record_factor'] = pd.cut(df_inputs_prepr_temp['mths_since_last_record'], 50)
            variable = 'mths_since_last_record_factor'

        if variable=='funded_amnt':
            df_inputs_prepr['funded_amnt_factor'] = pd.cut(df_inputs_prepr['funded_amnt'], 50)
            variable = 'funded_amnt_factor'

        if variable=='installment':
            df_inputs_prepr['installment_factor'] = pd.cut(df_inputs_prepr['installment'], 50)
            variable = 'installment_factor'

        # calculating woe and iv values
        if temp==0:
            df_temp = woe_continuous(df_inputs_prepr,variable,df_targets_prepr)
        if temp==1:
            df_temp = woe_continuous(df_inputs_prepr_temp,variable,df_targets_prepr[df_inputs_prepr_temp.index])
        plot_by_woe(df_temp,variable)
        df_with_woe_n_iv_values = pd.concat([df_with_woe_n_iv_values,df_temp],axis=0)

        # category creation within the variable
        if variable=='term_int':
            df_inputs_prepr['term:36'] = np.where((df_inputs_prepr['term_int'] == 36), 1, 0)
            df_inputs_prepr['term:60'] = np.where((df_inputs_prepr['term_int'] == 60), 1, 0)

        if variable=='emp_length_int':
            df_inputs_prepr['emp_length:0'] = np.where(df_inputs_prepr['emp_length_int'].isin([0]), 1, 0)
            df_inputs_prepr['emp_length:1'] = np.where(df_inputs_prepr['emp_length_int'].isin([1]), 1, 0)
            df_inputs_prepr['emp_length:2-4'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(2, 5)), 1, 0)
            df_inputs_prepr['emp_length:5-6'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(5, 7)), 1, 0)
            df_inputs_prepr['emp_length:7-9'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(7, 10)), 1, 0)
            df_inputs_prepr['emp_length:10'] = np.where(df_inputs_prepr['emp_length_int'].isin([10]), 1, 0)

        if variable=='mths_since_issue_d_factor':
            df_inputs_prepr['mths_since_issue_d:<38'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(38)), 1, 0)
            df_inputs_prepr['mths_since_issue_d:38-39'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(38, 40)), 1, 0)
            df_inputs_prepr['mths_since_issue_d:40-41'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(40, 42)), 1, 0)
            df_inputs_prepr['mths_since_issue_d:42-48'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(42, 49)), 1, 0)
            df_inputs_prepr['mths_since_issue_d:49-52'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(49, 53)), 1, 0)
            df_inputs_prepr['mths_since_issue_d:53-64'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(53, 65)), 1, 0)
            df_inputs_prepr['mths_since_issue_d:65-84'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(65, 85)), 1, 0)
            df_inputs_prepr['mths_since_issue_d:>84'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(85, int(df_inputs_prepr['mths_since_issue_d'].max()))), 1, 0)

        if variable=='int_rate_factor':
            df_inputs_prepr['int_rate:<9.548'] = np.where((df_inputs_prepr['int_rate'] <= 9.548), 1, 0)
            df_inputs_prepr['int_rate:9.548-12.025'] = np.where((df_inputs_prepr['int_rate'] > 9.548) & (df_inputs_prepr['int_rate'] <= 12.025), 1, 0)
            df_inputs_prepr['int_rate:12.025-15.74'] = np.where((df_inputs_prepr['int_rate'] > 12.025) & (df_inputs_prepr['int_rate'] <= 15.74), 1, 0)
            df_inputs_prepr['int_rate:15.74-20.281'] = np.where((df_inputs_prepr['int_rate'] > 15.74) & (df_inputs_prepr['int_rate'] <= 20.281), 1, 0)
            df_inputs_prepr['int_rate:>20.281'] = np.where((df_inputs_prepr['int_rate'] > 20.281), 1, 0)

        if variable=='mths_since_earliest_cr_line_factor':
            df_inputs_prepr['mths_since_earliest_cr_line:<140'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140)), 1, 0)
            df_inputs_prepr['mths_since_earliest_cr_line:141-164'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140, 165)), 1, 0)
            df_inputs_prepr['mths_since_earliest_cr_line:165-247'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(165, 248)), 1, 0)
            df_inputs_prepr['mths_since_earliest_cr_line:248-270'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(248, 271)), 1, 0)
            df_inputs_prepr['mths_since_earliest_cr_line:271-352'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(271, 353)), 1, 0)
            df_inputs_prepr['mths_since_earliest_cr_line:>352'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(353, int(df_inputs_prepr['mths_since_earliest_cr_line'].max()))), 1, 0)

        if variable=='delinq_2yrs':
            df_inputs_prepr['delinq_2yrs:0'] = np.where((df_inputs_prepr['delinq_2yrs'] == 0), 1, 0)
            df_inputs_prepr['delinq_2yrs:1-3'] = np.where((df_inputs_prepr['delinq_2yrs'] >= 1) & (df_inputs_prepr['delinq_2yrs'] <= 3), 1, 0)
            df_inputs_prepr['delinq_2yrs:>=4'] = np.where((df_inputs_prepr['delinq_2yrs'] >= 4), 1, 0)

        if variable=='inq_last_6mths':
            df_inputs_prepr['inq_last_6mths:0'] = np.where((df_inputs_prepr['inq_last_6mths'] == 0), 1, 0)
            df_inputs_prepr['inq_last_6mths:1-2'] = np.where((df_inputs_prepr['inq_last_6mths'] >= 1) & (df_inputs_prepr['inq_last_6mths'] <= 2), 1, 0)
            df_inputs_prepr['inq_last_6mths:3-6'] = np.where((df_inputs_prepr['inq_last_6mths'] >= 3) & (df_inputs_prepr['inq_last_6mths'] <= 6), 1, 0)
            df_inputs_prepr['inq_last_6mths:>6'] = np.where((df_inputs_prepr['inq_last_6mths'] > 6), 1, 0)

        if variable=='open_acc':
            df_inputs_prepr['open_acc:0'] = np.where((df_inputs_prepr['open_acc'] == 0), 1, 0)
            df_inputs_prepr['open_acc:1-3'] = np.where((df_inputs_prepr['open_acc'] >= 1) & (df_inputs_prepr['open_acc'] <= 3), 1, 0)
            df_inputs_prepr['open_acc:4-12'] = np.where((df_inputs_prepr['open_acc'] >= 4) & (df_inputs_prepr['open_acc'] <= 12), 1, 0)
            df_inputs_prepr['open_acc:13-17'] = np.where((df_inputs_prepr['open_acc'] >= 13) & (df_inputs_prepr['open_acc'] <= 17), 1, 0)
            df_inputs_prepr['open_acc:18-22'] = np.where((df_inputs_prepr['open_acc'] >= 18) & (df_inputs_prepr['open_acc'] <= 22), 1, 0)
            df_inputs_prepr['open_acc:23-25'] = np.where((df_inputs_prepr['open_acc'] >= 23) & (df_inputs_prepr['open_acc'] <= 25), 1, 0)
            df_inputs_prepr['open_acc:26-30'] = np.where((df_inputs_prepr['open_acc'] >= 26) & (df_inputs_prepr['open_acc'] <= 30), 1, 0)
            df_inputs_prepr['open_acc:>=31'] = np.where((df_inputs_prepr['open_acc'] >= 31), 1, 0)
            
        if variable=='pub_rec':
            df_inputs_prepr['pub_rec:0-2'] = np.where((df_inputs_prepr['pub_rec'] >= 0) & (df_inputs_prepr['pub_rec'] <= 2), 1, 0)
            df_inputs_prepr['pub_rec:3-4'] = np.where((df_inputs_prepr['pub_rec'] >= 3) & (df_inputs_prepr['pub_rec'] <= 4), 1, 0)
            df_inputs_prepr['pub_rec:>=5'] = np.where((df_inputs_prepr['pub_rec'] >= 5), 1, 0)

        if variable=='total_acc_factor':
            df_inputs_prepr['total_acc:<=27'] = np.where((df_inputs_prepr['total_acc'] <= 27), 1, 0)
            df_inputs_prepr['total_acc:28-51'] = np.where((df_inputs_prepr['total_acc'] >= 28) & (df_inputs_prepr['total_acc'] <= 51), 1, 0)
            df_inputs_prepr['total_acc:>=52'] = np.where((df_inputs_prepr['total_acc'] >= 52), 1, 0)
        
        if variable=='acc_now_delinq':
            df_inputs_prepr['acc_now_delinq:0'] = np.where((df_inputs_prepr['acc_now_delinq'] == 0), 1, 0)
            df_inputs_prepr['acc_now_delinq:>=1'] = np.where((df_inputs_prepr['acc_now_delinq'] >= 1), 1, 0)

        if variable=='total_rev_hi_lim_factor':
            df_inputs_prepr['total_rev_hi_lim:<=5K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] <= 5000), 1, 0)
            df_inputs_prepr['total_rev_hi_lim:5K-10K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 5000) & (df_inputs_prepr['total_rev_hi_lim'] <= 10000), 1, 0)
            df_inputs_prepr['total_rev_hi_lim:10K-20K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 10000) & (df_inputs_prepr['total_rev_hi_lim'] <= 20000), 1, 0)
            df_inputs_prepr['total_rev_hi_lim:20K-30K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 20000) & (df_inputs_prepr['total_rev_hi_lim'] <= 30000), 1, 0)
            df_inputs_prepr['total_rev_hi_lim:30K-40K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 30000) & (df_inputs_prepr['total_rev_hi_lim'] <= 40000), 1, 0)
            df_inputs_prepr['total_rev_hi_lim:40K-55K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 40000) & (df_inputs_prepr['total_rev_hi_lim'] <= 55000), 1, 0)
            df_inputs_prepr['total_rev_hi_lim:55K-95K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 55000) & (df_inputs_prepr['total_rev_hi_lim'] <= 95000), 1, 0)
            df_inputs_prepr['total_rev_hi_lim:>95K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 95000), 1, 0)

        if variable=='annual_inc_factor':
            # WoE is monotonically decreasing with income, so we split income in 10 equal categories, each with width of 15k.
            df_inputs_prepr['annual_inc:<20K'] = np.where((df_inputs_prepr['annual_inc'] <= 20000), 1, 0)
            df_inputs_prepr['annual_inc:20K-30K'] = np.where((df_inputs_prepr['annual_inc'] > 20000) & (df_inputs_prepr['annual_inc'] <= 30000), 1, 0)
            df_inputs_prepr['annual_inc:30K-40K'] = np.where((df_inputs_prepr['annual_inc'] > 30000) & (df_inputs_prepr['annual_inc'] <= 40000), 1, 0)
            df_inputs_prepr['annual_inc:40K-50K'] = np.where((df_inputs_prepr['annual_inc'] > 40000) & (df_inputs_prepr['annual_inc'] <= 50000), 1, 0)
            df_inputs_prepr['annual_inc:50K-60K'] = np.where((df_inputs_prepr['annual_inc'] > 50000) & (df_inputs_prepr['annual_inc'] <= 60000), 1, 0)
            df_inputs_prepr['annual_inc:60K-70K'] = np.where((df_inputs_prepr['annual_inc'] > 60000) & (df_inputs_prepr['annual_inc'] <= 70000), 1, 0)
            df_inputs_prepr['annual_inc:70K-80K'] = np.where((df_inputs_prepr['annual_inc'] > 70000) & (df_inputs_prepr['annual_inc'] <= 80000), 1, 0)
            df_inputs_prepr['annual_inc:80K-90K'] = np.where((df_inputs_prepr['annual_inc'] > 80000) & (df_inputs_prepr['annual_inc'] <= 90000), 1, 0)
            df_inputs_prepr['annual_inc:90K-100K'] = np.where((df_inputs_prepr['annual_inc'] > 90000) & (df_inputs_prepr['annual_inc'] <= 100000), 1, 0)
            df_inputs_prepr['annual_inc:100K-120K'] = np.where((df_inputs_prepr['annual_inc'] > 100000) & (df_inputs_prepr['annual_inc'] <= 120000), 1, 0)
            df_inputs_prepr['annual_inc:120K-140K'] = np.where((df_inputs_prepr['annual_inc'] > 120000) & (df_inputs_prepr['annual_inc'] <= 140000), 1, 0)
            df_inputs_prepr['annual_inc:>140K'] = np.where((df_inputs_prepr['annual_inc'] > 140000), 1, 0)

        if variable=='dti_factor':
            df_inputs_prepr['dti:<=1.4'] = np.where((df_inputs_prepr['dti'] <= 1.4), 1, 0)
            df_inputs_prepr['dti:1.4-3.5'] = np.where((df_inputs_prepr['dti'] > 1.4) & (df_inputs_prepr['dti'] <= 3.5), 1, 0)
            df_inputs_prepr['dti:3.5-7.7'] = np.where((df_inputs_prepr['dti'] > 3.5) & (df_inputs_prepr['dti'] <= 7.7), 1, 0)
            df_inputs_prepr['dti:7.7-10.5'] = np.where((df_inputs_prepr['dti'] > 7.7) & (df_inputs_prepr['dti'] <= 10.5), 1, 0)
            df_inputs_prepr['dti:10.5-16.1'] = np.where((df_inputs_prepr['dti'] > 10.5) & (df_inputs_prepr['dti'] <= 16.1), 1, 0)
            df_inputs_prepr['dti:16.1-20.3'] = np.where((df_inputs_prepr['dti'] > 16.1) & (df_inputs_prepr['dti'] <= 20.3), 1, 0)
            df_inputs_prepr['dti:20.3-21.7'] = np.where((df_inputs_prepr['dti'] > 20.3) & (df_inputs_prepr['dti'] <= 21.7), 1, 0)
            df_inputs_prepr['dti:21.7-22.4'] = np.where((df_inputs_prepr['dti'] > 21.7) & (df_inputs_prepr['dti'] <= 22.4), 1, 0)
            df_inputs_prepr['dti:22.4-35'] = np.where((df_inputs_prepr['dti'] > 22.4) & (df_inputs_prepr['dti'] <= 35), 1, 0)
            df_inputs_prepr['dti:>35'] = np.where((df_inputs_prepr['dti'] > 35), 1, 0)

        if variable=='mths_since_last_delinq_factor':
            df_inputs_prepr['mths_since_last_delinq:Missing'] = np.where((df_inputs_prepr['mths_since_last_delinq'].isnull()), 1, 0)
            df_inputs_prepr['mths_since_last_delinq:0-3'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 0) & (df_inputs_prepr['mths_since_last_delinq'] <= 3), 1, 0)
            df_inputs_prepr['mths_since_last_delinq:4-30'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 4) & (df_inputs_prepr['mths_since_last_delinq'] <= 30), 1, 0)
            df_inputs_prepr['mths_since_last_delinq:31-56'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 31) & (df_inputs_prepr['mths_since_last_delinq'] <= 56), 1, 0)
            df_inputs_prepr['mths_since_last_delinq:>=57'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 57), 1, 0)

        if variable=='mths_since_last_record_factor':
            df_inputs_prepr['mths_since_last_record:Missing'] = np.where((df_inputs_prepr['mths_since_last_record'].isnull()), 1, 0)
            df_inputs_prepr['mths_since_last_record:0-2'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 0) & (df_inputs_prepr['mths_since_last_record'] <= 2), 1, 0)
            df_inputs_prepr['mths_since_last_record:3-20'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 3) & (df_inputs_prepr['mths_since_last_record'] <= 20), 1, 0)
            df_inputs_prepr['mths_since_last_record:21-31'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 21) & (df_inputs_prepr['mths_since_last_record'] <= 31), 1, 0)
            df_inputs_prepr['mths_since_last_record:32-80'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 32) & (df_inputs_prepr['mths_since_last_record'] <= 80), 1, 0)
            df_inputs_prepr['mths_since_last_record:81-86'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 81) & (df_inputs_prepr['mths_since_last_record'] <= 86), 1, 0)
            df_inputs_prepr['mths_since_last_record:>86'] = np.where((df_inputs_prepr['mths_since_last_record'] > 86), 1, 0)

        # if variable=='funded_amnt_factor':
        #     df_inputs_prepr['funded_amnt:<10520'] = np.where((df_inputs_prepr['funded_amnt'] < 10520), 1, 0)
        #     df_inputs_prepr['funded_amnt:10520-15280'] = np.where((df_inputs_prepr['funded_amnt'] >= 10520) & (df_inputs_prepr['funded_amnt'] < 15280), 1, 0)
        #     df_inputs_prepr['funded_amnt:15280-19360'] = np.where((df_inputs_prepr['funded_amnt'] >= 15280) & (df_inputs_prepr['funded_amnt'] < 19360), 1, 0)
        #     df_inputs_prepr['funded_amnt:19360-22080'] = np.where((df_inputs_prepr['funded_amnt'] >= 19360) & (df_inputs_prepr['funded_amnt'] < 22080), 1, 0)
        #     df_inputs_prepr['funded_amnt:22080-26840'] = np.where((df_inputs_prepr['funded_amnt'] >= 22080) & (df_inputs_prepr['funded_amnt'] < 26840), 1, 0)
        #     df_inputs_prepr['funded_amnt:26840-30240'] = np.where((df_inputs_prepr['funded_amnt'] >= 26840) & (df_inputs_prepr['funded_amnt'] < 30240), 1, 0)
        #     df_inputs_prepr['funded_amnt:>=30240'] = np.where((df_inputs_prepr['funded_amnt'] >= 30240), 1, 0)

        # if variable=='installment_factor':
        #     df_inputs_prepr['installment:<266'] = np.where((df_inputs_prepr['installment'] < 266), 1, 0)
        #     df_inputs_prepr['installment:266-502'] = np.where((df_inputs_prepr['installment'] >= 266) & (df_inputs_prepr['installment'] < 502), 1, 0)
        #     df_inputs_prepr['installment:502-737.8'] = np.where((df_inputs_prepr['installment'] >= 502) & (df_inputs_prepr['installment'] < 737.8), 1, 0)
        #     df_inputs_prepr['installment:737.8-1020.8'] = np.where((df_inputs_prepr['installment'] >= 737.8) & (df_inputs_prepr['installment'] < 1020.8), 1, 0)
        #     df_inputs_prepr['installment:>=1020.8'] = np.where((df_inputs_prepr['installment'] >= 1020.8), 1, 0)

    # Reference Categories
    ## pick the riskiest group for reference category.
    ## 'term_int' : '60' will be the reference category.
    ## 'emp_length_int': '0' will be the reference category.

    return df_inputs_prepr,df_with_woe_n_iv_values

# removing features based on p-values
def feature_selection_using_pvalue(variables):
    # removing features based on p-values
    features = []
    ## here identifying the features which are left after p-value filtering
    temp = 0
    for feature in config.limited_input_features:
        for var in variables:
            if var in feature:
                temp = 1
        if temp==0:
            features.append(feature)
        temp=0

    variables = []
    for feature in features:
        variable = (feature).split(':',1)[0]
        if variable not in variables:
            variables.append(variable)
    
    return features,variables

# create dataframe back to original variable columns from dummy variable columns
def original_var_col(inputs_train_with_ref_cat,variables):

    columns = list(inputs_train_with_ref_cat.columns)
    final_df = pd.DataFrame()
    for variable in variables:
        req_cols = []
        for col in columns:
            if variable in col:
                req_cols.append(col)
        df = inputs_train_with_ref_cat[req_cols].idxmax(axis=1).to_frame()
        df.rename(columns={df.columns[0]:variable},inplace=True)
        if final_df.empty:
            final_df = df
        else:
            final_df = pd.merge(final_df,df,left_index=True,right_index=True)

    return final_df

# shortlisting varibles using IV
def feature_selection_using_iv(final_df,loan_data_targets_train,variables):
    # Calculating WoE and IV wrt to final_df
    df = final_df.copy()
    good_bad_variable_df = loan_data_targets_train.copy()
    ## calculating WoE values
    df_woe = pd.DataFrame()
    for variable in variables:
        discrete_variable_name = variable
        df1 = woe_continuous(df, discrete_variable_name, good_bad_variable_df)
        df_woe = pd.concat([df_woe,df1],axis=0)
    ## calculating IV values
    df_iv = df_woe[['variable','IV']].groupby(['variable']).mean()
    df_iv = df_iv.reset_index()
    ## shortlisting variables using IV
    iv_variables = list(df_iv[df_iv['IV']>=0.05]['variable'])

    ## getting all dummy featuers of filtered variables
    iv_features = []
    for feature in config.limited_input_features:
        for var in iv_variables:
            if var in feature:
                iv_features.append(feature)
    
    return df_woe,df_iv,iv_variables,iv_features

# start model estimation
def get_input_train_data(loan_data_inputs_train):

    inputs_train_with_ref_cat = loan_data_inputs_train.loc[: , config.limited_input_features]
    inputs_train = inputs_train_with_ref_cat.drop(config.ref_categories, axis = 1)

    inputs_train_with_ref_cat.to_csv(r'C:\Shubham\credit_scorecard\output_data\inputs_train_with_ref_cat.csv',index=False)

    return inputs_train


# logistic regression
def logistic_regression(inputs_train,loan_data_targets_train):

    reg = LogisticRegression()
    reg.fit(inputs_train, loan_data_targets_train)

    # creating a summary table
    feature_name = inputs_train.columns.values
    summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
    summary_table['Coefficients'] = np.transpose(reg.coef_)
    summary_table.index = summary_table.index + 1
    summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
    summary_table = summary_table.sort_index()

    return reg,summary_table

class LogisticRegression_with_p_values:
    
    def __init__(self,*args,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)

    def fit(self,X,y):
        self.model.fit(X,y)

        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X) ## Fisher Information Matrix
        F_ij = F_ij.astype(np.float64) ## Inverse Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores] ### two tailed test for p-values

        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values

def logistic_regression_with_p_values(inputs_train,loan_data_targets_train):

    reg = LogisticRegression_with_p_values()
    reg.fit(inputs_train, loan_data_targets_train)

    feature_name = inputs_train.columns.values
    summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
    summary_table['Coefficients'] = np.transpose(reg.coef_)
    summary_table.index = summary_table.index + 1
    summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
    summary_table = summary_table.sort_index()

    p_values = reg.p_values
    p_values = np.append(np.nan, np.array(p_values))
    summary_table['p_values'] = p_values

    return reg,summary_table

def get_input_test_data(loan_data_inputs_test):

    inputs_test_with_ref_cat = loan_data_inputs_test.loc[: , config.limited_input_features]
    inputs_test = inputs_test_with_ref_cat.drop(config.ref_categories, axis = 1)

    inputs_test_with_ref_cat.to_csv(r'C:\Shubham\credit_scorecard\output_data\inputs_test_with_ref_cat.csv',index=False)

    return inputs_test

# predicting default for a customer
def default_prediction(input_df):

    with open(r'C:\Shubham\credit_scorecard\src\pd_model.pkl',"rb") as file_obj:
        model = pickle.load(file_obj)
    # print("Hello")
    pred_dict = {
        "default_prediction" : model.model.predict(input_df),
        "default_prob" : model.model.predict_proba(input_df)
    }

    return pred_dict

# model validation
def out_of_sample_validation(inputs_test,loan_data_targets_test,reg):

    # Calculates the predicted values for the dependent variable (targets)
    # based on the values of the independent variables (inputs) supplied as an argument.
    # y_hat_test = reg.model.predict(inputs_test)

    # Calculates the predicted probability values for the dependent variable (targets)
    # based on the values of the independent variables (inputs) supplied as an argument.
    y_hat_test_proba = reg.model.predict_proba(inputs_test)

    # by doing this, we take only the probabilities for being 1.
    y_hat_test_proba = y_hat_test_proba[: ][: , 1]

    loan_data_targets_test_temp = loan_data_targets_test
    loan_data_targets_test_temp.reset_index(drop = True, inplace = True)

    df_actual_predicted_probs = pd.concat([loan_data_targets_test_temp, pd.DataFrame(y_hat_test_proba)], axis = 1)
    df_actual_predicted_probs.columns = ['loan_data_targets_test', 'y_hat_test_proba']
    df_actual_predicted_probs.index = inputs_test.index

    return df_actual_predicted_probs

# accuracy and area under the curve
def accuracy_n_auc(df_actual_predicted_probs):

    tr = 0.6
    df_actual_predicted_probs['y_hat_test'] = np.where(df_actual_predicted_probs['y_hat_test_proba'] > tr, 1, 0)

    # calculating accuracy
    confusion_matrix = pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'],
                                   df_actual_predicted_probs['y_hat_test'], 
                                   rownames = ['Actual'], 
                                   colnames = ['Predicted'])
    print(confusion_matrix)
    accuracy = (confusion_matrix.iloc[0, 0] + confusion_matrix.iloc[1, 1])/df_actual_predicted_probs.shape[0]

    # ROC Curve
    ## As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.
    fpr, tpr, thresholds = roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])

    # plotting roc curve
    plt.plot(fpr, tpr)
    # We plot the false positive rate along the x-axis and the true positive rate along the y-axis,
    # thus plotting the ROC curve.
    plt.plot(fpr, fpr, linestyle = '--', color = 'k')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')

    # calculating area under roc (AUROC)
    AUROC = roc_auc_score(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])

    return df_actual_predicted_probs, confusion_matrix, accuracy, fpr, tpr, thresholds, AUROC

# calculating GINI and Kolmogorov-Smirnov
def gini_n_ks(df_actual_predicted_probs,AUROC):

    df_actual_predicted_probs = df_actual_predicted_probs.sort_values('y_hat_test_proba')
    df_actual_predicted_probs = df_actual_predicted_probs.reset_index()
    # We calculate the cumulative number of all observations.
    df_actual_predicted_probs['Cumulative N Population'] = df_actual_predicted_probs.index + 1
    df_actual_predicted_probs['Cumulative N Good'] = df_actual_predicted_probs['loan_data_targets_test'].cumsum()
    df_actual_predicted_probs['Cumulative N Bad'] = df_actual_predicted_probs['Cumulative N Population'] - df_actual_predicted_probs['loan_data_targets_test'].cumsum()
    df_actual_predicted_probs['Cumulative Perc Population'] = df_actual_predicted_probs['Cumulative N Population'] / (df_actual_predicted_probs.shape[0])
    df_actual_predicted_probs['Cumulative Perc Good'] = df_actual_predicted_probs['Cumulative N Good'] / df_actual_predicted_probs['loan_data_targets_test'].sum()
    df_actual_predicted_probs['Cumulative Perc Bad'] = df_actual_predicted_probs['Cumulative N Bad'] / (df_actual_predicted_probs.shape[0] - df_actual_predicted_probs['loan_data_targets_test'].sum())

    # plotting GINI
    plt.plot(df_actual_predicted_probs['Cumulative Perc Population'], df_actual_predicted_probs['Cumulative Perc Bad'])
    plt.plot(df_actual_predicted_probs['Cumulative Perc Population'], df_actual_predicted_probs['Cumulative Perc Population'], linestyle = '--', color = 'k')
    plt.xlabel('Cumulative % Population')
    plt.ylabel('Cumulative % Bad')
    plt.title('Gini')

    # Here we calculate Gini from AUROC.
    Gini = AUROC * 2 - 1

    # plotting KS
    plt.plot(df_actual_predicted_probs['y_hat_test_proba'], df_actual_predicted_probs['Cumulative Perc Bad'], color = 'r')
    plt.plot(df_actual_predicted_probs['y_hat_test_proba'], df_actual_predicted_probs['Cumulative Perc Good'], color = 'b')
    plt.xlabel('Estimated Probability for being Good')
    plt.ylabel('Cumulative %')
    plt.title('Kolmogorov-Smirnov')

    # calculating KS
    KS = max(df_actual_predicted_probs['Cumulative Perc Bad'] - df_actual_predicted_probs['Cumulative Perc Good'])

    return df_actual_predicted_probs, Gini, KS

# crating a scorecard
def socrecard(ref_categories,summary_table):
    # creating reference category dataframe 
    df_ref_categories = pd.DataFrame(ref_categories, columns = ['Feature name'])
    df_ref_categories['Coefficients'] = 0
    df_ref_categories['p_values'] = np.nan

    # setting min and max score
    min_score = 300
    max_score = 850

    # scorecard dataframe creation
    df_scorecard = pd.concat([summary_table, df_ref_categories])
    df_scorecard = df_scorecard.reset_index()
    df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(':').str[0]

    min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
    max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()

    df_scorecard['Score - Calculation'] = df_scorecard['Coefficients'] * (max_score - min_score) / (max_sum_coef - min_sum_coef)
    df_scorecard['Score - Calculation'][0] = ((df_scorecard['Coefficients'][0] - min_sum_coef) / (max_sum_coef - min_sum_coef)) * (max_score - min_score) + min_score
    df_scorecard['Score - Preliminary'] = df_scorecard['Score - Calculation'].round()

    min_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].min().sum()
    max_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].max().sum()

    df_scorecard['Difference'] = df_scorecard['Score - Preliminary'] - df_scorecard['Score - Calculation']
    df_scorecard['Score - Final'] = df_scorecard['Score - Preliminary']
    # df_scorecard['Score - Final'][77] = 16  # manual

    min_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Final'].min().sum()
    max_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Final'].max().sum()

    df_scorecard.to_csv(r'C:\Shubham\credit_scorecard\output_data\df_scorecard.csv',index=False)

    return df_scorecard,min_sum_score_prel,max_sum_score_prel,min_sum_coef,max_sum_coef

# calculating credit score
def calc_credit_score(inputs_test_with_ref_cat,df_scorecard):

    inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat.copy()
    inputs_test_with_ref_cat_w_intercept.insert(0, 'Intercept', 1)
    # picking only those columns which are present in feature name column of df_scorecard
    inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat_w_intercept[df_scorecard['Feature name'].values]

    scorecard_scores = (df_scorecard['Score - Final']).to_list()

    y_scores = inputs_test_with_ref_cat_w_intercept.dot(scorecard_scores)

    return y_scores

# create PD from credit score
def credit_score_to_pd(y_scores,min_sum_coef,max_sum_coef,min_score=300,max_score=850):

    sum_coef_from_score = ((y_scores - min_score) / (max_score - min_score)) * (max_sum_coef - min_sum_coef) + min_sum_coef
    y_hat_proba_from_score = np.exp(sum_coef_from_score) / (np.exp(sum_coef_from_score) + 1)

    return y_hat_proba_from_score

# setting cut-off
def set_cut_off(df_actual_predicted_probs,fpr,tpr,thresholds,min_sum_coef,max_sum_coef,min_score=300,max_score=850):

    df_cutoffs = pd.concat([pd.DataFrame(thresholds), pd.DataFrame(fpr), pd.DataFrame(tpr)], axis = 1)
    df_cutoffs.columns = ['thresholds', 'fpr', 'tpr']
    df_cutoffs['thresholds'][0] = 1 - 1 / np.power(10, 16)

    # calculating score
    df_cutoffs['Score'] = ((np.log(df_cutoffs['thresholds']/(1 - df_cutoffs['thresholds'])) - min_sum_coef) *
                            (max_score - min_score)/(max_sum_coef - min_sum_coef) + 
                            min_score).round()

    df_cutoffs['Score'][0] = max_score

    def n_approved(p):
        return np.where(df_actual_predicted_probs['y_hat_test_proba'] >= p, 1, 0).sum()

    df_cutoffs['N Approved'] = df_cutoffs['thresholds'].apply(n_approved)

    df_cutoffs['N Rejected'] = df_actual_predicted_probs['y_hat_test_proba'].shape[0] - df_cutoffs['N Approved']

    df_cutoffs['Approval Rate'] = df_cutoffs['N Approved'] / df_actual_predicted_probs['y_hat_test_proba'].shape[0]
    df_cutoffs['Rejection Rate'] = 1 - df_cutoffs['Approval Rate']

    return df_cutoffs

def test_data_preprocessing(loan_data_inputs_2015):

    # contains all the binned variables including reference categories.
    inputs_train_with_ref_cat = pd.read_csv(r"C:\Shubham\credit_scorecard\output_data\inputs_train_with_ref_cat.csv")

    inputs_2015_with_ref_cat = loan_data_inputs_2015.loc[: , config.limited_input_features]
    
    df_scorecard = pd.read_csv(r'C:\Shubham\credit_scorecard\output_data\df_scorecard.csv', index_col = 0)

    inputs_train_with_ref_cat_w_intercept = inputs_train_with_ref_cat
    inputs_train_with_ref_cat_w_intercept.insert(0, 'Intercept', 1)
    inputs_train_with_ref_cat_w_intercept = inputs_train_with_ref_cat_w_intercept[df_scorecard['Feature name'].values]
    # "df_scorecard['Feature name'].values" will contain only iv_features.

    inputs_2015_with_ref_cat_w_intercept = inputs_2015_with_ref_cat
    inputs_2015_with_ref_cat_w_intercept.insert(0, 'Intercept', 1)
    inputs_2015_with_ref_cat_w_intercept = inputs_2015_with_ref_cat_w_intercept[df_scorecard['Feature name'].values]

    scorecard_scores = (df_scorecard['Score - Final']).to_list()

    y_scores_train = inputs_train_with_ref_cat_w_intercept.dot(scorecard_scores)
    y_scores_2015 = inputs_2015_with_ref_cat_w_intercept.dot(scorecard_scores)

    inputs_train_with_ref_cat_w_intercept = pd.concat([inputs_train_with_ref_cat_w_intercept, y_scores_train], axis = 1)
    inputs_2015_with_ref_cat_w_intercept = pd.concat([inputs_2015_with_ref_cat_w_intercept, y_scores_2015], axis = 1)

    inputs_train_with_ref_cat_w_intercept.columns.values[inputs_train_with_ref_cat_w_intercept.shape[1] - 1] = 'Score'
    inputs_2015_with_ref_cat_w_intercept.columns.values[inputs_2015_with_ref_cat_w_intercept.shape[1] - 1] = 'Score'
    # Here we rename the columns containing scores to "Score" in both dataframes.

    inputs_train_with_ref_cat_w_intercept['Score:300-350'] = np.where((inputs_train_with_ref_cat_w_intercept['Score'] >= 300) & (inputs_train_with_ref_cat_w_intercept['Score'] < 350), 1, 0)
    inputs_train_with_ref_cat_w_intercept['Score:350-400'] = np.where((inputs_train_with_ref_cat_w_intercept['Score'] >= 350) & (inputs_train_with_ref_cat_w_intercept['Score'] < 400), 1, 0)
    inputs_train_with_ref_cat_w_intercept['Score:400-450'] = np.where((inputs_train_with_ref_cat_w_intercept['Score'] >= 400) & (inputs_train_with_ref_cat_w_intercept['Score'] < 450), 1, 0)
    inputs_train_with_ref_cat_w_intercept['Score:450-500'] = np.where((inputs_train_with_ref_cat_w_intercept['Score'] >= 450) & (inputs_train_with_ref_cat_w_intercept['Score'] < 500), 1, 0)
    inputs_train_with_ref_cat_w_intercept['Score:500-550'] = np.where((inputs_train_with_ref_cat_w_intercept['Score'] >= 500) & (inputs_train_with_ref_cat_w_intercept['Score'] < 550), 1, 0)
    inputs_train_with_ref_cat_w_intercept['Score:550-600'] = np.where((inputs_train_with_ref_cat_w_intercept['Score'] >= 550) & (inputs_train_with_ref_cat_w_intercept['Score'] < 600), 1, 0)
    inputs_train_with_ref_cat_w_intercept['Score:600-650'] = np.where((inputs_train_with_ref_cat_w_intercept['Score'] >= 600) & (inputs_train_with_ref_cat_w_intercept['Score'] < 650), 1, 0)
    inputs_train_with_ref_cat_w_intercept['Score:650-700'] = np.where((inputs_train_with_ref_cat_w_intercept['Score'] >= 650) & (inputs_train_with_ref_cat_w_intercept['Score'] < 700), 1, 0)
    inputs_train_with_ref_cat_w_intercept['Score:700-750'] = np.where((inputs_train_with_ref_cat_w_intercept['Score'] >= 700) & (inputs_train_with_ref_cat_w_intercept['Score'] < 750), 1, 0)
    inputs_train_with_ref_cat_w_intercept['Score:750-800'] = np.where((inputs_train_with_ref_cat_w_intercept['Score'] >= 750) & (inputs_train_with_ref_cat_w_intercept['Score'] < 800), 1, 0)
    inputs_train_with_ref_cat_w_intercept['Score:800-850'] = np.where((inputs_train_with_ref_cat_w_intercept['Score'] >= 800) & (inputs_train_with_ref_cat_w_intercept['Score'] <= 850), 1, 0)

    inputs_2015_with_ref_cat_w_intercept['Score:300-350'] = np.where((inputs_2015_with_ref_cat_w_intercept['Score'] >= 300) & (inputs_2015_with_ref_cat_w_intercept['Score'] < 350), 1, 0)
    inputs_2015_with_ref_cat_w_intercept['Score:350-400'] = np.where((inputs_2015_with_ref_cat_w_intercept['Score'] >= 350) & (inputs_2015_with_ref_cat_w_intercept['Score'] < 400), 1, 0)
    inputs_2015_with_ref_cat_w_intercept['Score:400-450'] = np.where((inputs_2015_with_ref_cat_w_intercept['Score'] >= 400) & (inputs_2015_with_ref_cat_w_intercept['Score'] < 450), 1, 0)
    inputs_2015_with_ref_cat_w_intercept['Score:450-500'] = np.where((inputs_2015_with_ref_cat_w_intercept['Score'] >= 450) & (inputs_2015_with_ref_cat_w_intercept['Score'] < 500), 1, 0)
    inputs_2015_with_ref_cat_w_intercept['Score:500-550'] = np.where((inputs_2015_with_ref_cat_w_intercept['Score'] >= 500) & (inputs_2015_with_ref_cat_w_intercept['Score'] < 550), 1, 0)
    inputs_2015_with_ref_cat_w_intercept['Score:550-600'] = np.where((inputs_2015_with_ref_cat_w_intercept['Score'] >= 550) & (inputs_2015_with_ref_cat_w_intercept['Score'] < 600), 1, 0)
    inputs_2015_with_ref_cat_w_intercept['Score:600-650'] = np.where((inputs_2015_with_ref_cat_w_intercept['Score'] >= 600) & (inputs_2015_with_ref_cat_w_intercept['Score'] < 650), 1, 0)
    inputs_2015_with_ref_cat_w_intercept['Score:650-700'] = np.where((inputs_2015_with_ref_cat_w_intercept['Score'] >= 650) & (inputs_2015_with_ref_cat_w_intercept['Score'] < 700), 1, 0)
    inputs_2015_with_ref_cat_w_intercept['Score:700-750'] = np.where((inputs_2015_with_ref_cat_w_intercept['Score'] >= 700) & (inputs_2015_with_ref_cat_w_intercept['Score'] < 750), 1, 0)
    inputs_2015_with_ref_cat_w_intercept['Score:750-800'] = np.where((inputs_2015_with_ref_cat_w_intercept['Score'] >= 750) & (inputs_2015_with_ref_cat_w_intercept['Score'] < 800), 1, 0)
    inputs_2015_with_ref_cat_w_intercept['Score:800-850'] = np.where((inputs_2015_with_ref_cat_w_intercept['Score'] >= 800) & (inputs_2015_with_ref_cat_w_intercept['Score'] <= 850), 1, 0)

    return inputs_train_with_ref_cat_w_intercept, inputs_2015_with_ref_cat_w_intercept

# population stability index
def pop_stability_index(inputs_train_with_ref_cat_w_intercept,inputs_2015_with_ref_cat_w_intercept):

    PSI_calc_train = inputs_train_with_ref_cat_w_intercept.sum() / inputs_train_with_ref_cat_w_intercept.shape[0]
    # We create a dataframe with proportions of observations for each dummy variable for the old ("expected") data.

    PSI_calc_2015 = inputs_2015_with_ref_cat_w_intercept.sum() / inputs_2015_with_ref_cat_w_intercept.shape[0]
    # We create a dataframe with proportions of observations for each dummy variable for the new ("actual") data.

    PSI_calc = pd.concat([PSI_calc_train, PSI_calc_2015], axis = 1)
    # We concatenate the two dataframes along the columns.

    PSI_calc = PSI_calc.reset_index()
    # We reset the index of the dataframe. The index becomes from 0 to the total number of rows less one.
    # The old index, which is the dummy variable name, becomes a column, named 'index'.
    PSI_calc['Original feature name'] = PSI_calc['index'].str.split(':').str[0]
    # We create a new column, called 'Original feature name', which contains the value of the 'Feature name' column,
    # up to the column symbol.
    PSI_calc.columns = ['index', 'Proportions_Train', 'Proportions_New', 'Original feature name']
    # We change the names of the columns of the dataframe.

    PSI_calc = PSI_calc[np.array(['index', 'Original feature name', 'Proportions_Train', 'Proportions_New'])]

    PSI_calc = PSI_calc[(PSI_calc['index'] != 'Intercept') & (PSI_calc['index'] != 'Score')]
    # We remove the rows with values in the 'index' column 'Intercept' and 'Score'.

    PSI_calc['Contribution'] = np.where((PSI_calc['Proportions_Train'] == 0) | (PSI_calc['Proportions_New'] == 0), 0, (PSI_calc['Proportions_New'] - PSI_calc['Proportions_Train']) * np.log(PSI_calc['Proportions_New'] / PSI_calc['Proportions_Train']))
    # We calculate the contribution of each dummy variable to the PSI of each original variable it comes from.
    # If either the proportion of old data or the proportion of new data are 0, the contribution is 0.
    # Otherwise, we apply the PSI formula for each contribution.

    PSI_calc.groupby('Original feature name')['Contribution'].sum()
    # Finally, we sum all contributions for each original independent variable and the 'Score' variable.

    return PSI_calc

# Functions for LGD and EAD Models
## creating dependent variables
def dependent_var_creation(loan_data_defaults):
    loan_data_defaults['recovery_rate'] = loan_data_defaults['recoveries'] / loan_data_defaults['funded_amnt']
    loan_data_defaults['recovery_rate'] = np.where(loan_data_defaults['recovery_rate'] > 1, 1, loan_data_defaults['recovery_rate'])
    loan_data_defaults['recovery_rate'] = np.where(loan_data_defaults['recovery_rate'] < 0, 0, loan_data_defaults['recovery_rate'])

    loan_data_defaults['CCF'] = (loan_data_defaults['funded_amnt'] - loan_data_defaults['total_rec_prncp']) / loan_data_defaults['funded_amnt']

    loan_data_defaults['recovery_rate_0_1'] = np.where(loan_data_defaults['recovery_rate'] == 0, 0, 1)

    loan_data_defaults.to_csv(r'C:\Shubham\credit_scorecard\output_data_lgd_n_ead\loan_data_defaults.csv')

    return loan_data_defaults

## Exploring Dependent Variables
def dependent_var_visualization(loan_data_defaults):

    plt.hist(loan_data_defaults['recovery_rate'], bins = 50)
    plt.show()

    plt.hist(loan_data_defaults['CCF'], bins = 100)
    plt.show()

## filtering the train dataset using required features
def filter_lgd_train_data(lgd_inputs_stage_1):

    lgd_inputs_stage_1 = lgd_inputs_stage_1[config.features_all]
    lgd_inputs_stage_1 = lgd_inputs_stage_1.drop(config.features_reference_cat, axis = 1)

    return lgd_inputs_stage_1

## Linear Regression
def linear_regression(inputs_train,loan_data_targets_train):

    reg = LinearRegression()
    reg.fit(inputs_train, loan_data_targets_train)

    # creating a summary table
    feature_name = inputs_train.columns.values
    summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
    summary_table['Coefficients'] = np.transpose(reg.coef_)
    summary_table.index = summary_table.index + 1
    summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
    summary_table = summary_table.sort_index()

    return reg,summary_table


'''
By typing the code below we will ovewrite a part of the class with one that includes p-values
Here's the full source code of the ORIGINAL class: https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/linear_model/base.py#L362
'''
class LinearRegression(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """
    
    # nothing changes in __init__
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=1):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs

    
    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)
        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])
        # compute the t-statistic for each feature
        self.t = self.coef_ / se
        # find the p-value for each feature
        self.p = np.squeeze(2 * (1 - stat.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1])))
        return self


def linear_regression_with_p_values(lgd_inputs_stage_2_train, lgd_targets_stage_2_train):

    reg_lgd_st_2 = LinearRegression()
    reg_lgd_st_2.fit(lgd_inputs_stage_2_train, lgd_targets_stage_2_train)

    feature_name = lgd_inputs_stage_2_train.columns.values
    summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)
    summary_table['Coefficients'] = np.transpose(reg_lgd_st_2.coef_)
    summary_table.index = summary_table.index + 1
    summary_table.loc[0] = ['Intercept', reg_lgd_st_2.intercept_]
    summary_table = summary_table.sort_index()

    p_values = reg_lgd_st_2.p
    p_values = np.append(np.nan,np.array(p_values))
    summary_table['p_values'] = p_values.round(3)

    return reg_lgd_st_2,summary_table

### validating linear regression model
def validate_linear_regression(reg_lgd_st_2,lgd_inputs_stage_2_test,lgd_targets_stage_2_test):

    y_hat_test_lgd_stage_2 = reg_lgd_st_2.predict(lgd_inputs_stage_2_test)

    lgd_targets_stage_2_test_temp = lgd_targets_stage_2_test
    lgd_targets_stage_2_test_temp = lgd_targets_stage_2_test_temp.reset_index(drop = True)

    df_corr = pd.concat([lgd_targets_stage_2_test_temp, pd.DataFrame(y_hat_test_lgd_stage_2)], axis = 1).corr()

    sns.distplot(lgd_targets_stage_2_test - y_hat_test_lgd_stage_2)

    return df_corr

## Combining Logistic and Linear Regression Models
def combined_model(reg_lgd_st_2,reg_lgd_st_1,lgd_inputs_stage_1_test):

    y_hat_test_lgd_stage_1 = reg_lgd_st_1.model.predict(lgd_inputs_stage_1_test)
    y_hat_test_lgd_stage_2_all = reg_lgd_st_2.predict(lgd_inputs_stage_1_test)
    # Here we combine the predictions of the models from the two stages.
    y_hat_test_lgd = y_hat_test_lgd_stage_1 * y_hat_test_lgd_stage_2_all

    y_hat_test_lgd = np.where(y_hat_test_lgd < 0, 0, y_hat_test_lgd)
    y_hat_test_lgd = np.where(y_hat_test_lgd > 1, 1, y_hat_test_lgd)

    return y_hat_test_lgd

## EAD Model Validation
def validate_ead_model(reg_ead,ead_inputs_test,ead_targets_test):

    y_hat_test_ead = reg_ead.predict(ead_inputs_test)

    ead_targets_test_temp = ead_targets_test
    ead_targets_test_temp = ead_targets_test_temp.reset_index(drop = True)

    df_corr = pd.concat([ead_targets_test_temp, pd.DataFrame(y_hat_test_ead)], axis = 1).corr()

    sns.distplot(ead_targets_test - y_hat_test_ead)

    y_hat_test_ead = np.where(y_hat_test_ead < 0, 0, y_hat_test_ead)
    y_hat_test_ead = np.where(y_hat_test_ead > 1, 1, y_hat_test_ead)

    return df_corr, y_hat_test_ead

# Expected Loss
def expected_loss_estimation(reg_lgd_st_1,reg_lgd_st_2,reg_ead,loan_data_preprocessed,loan_data_preprocessed_lgd_ead):

    loan_data_preprocessed['recovery_rate_st_1'] = reg_lgd_st_1.model.predict(loan_data_preprocessed_lgd_ead)
    loan_data_preprocessed['recovery_rate_st_2'] = reg_lgd_st_2.predict(loan_data_preprocessed_lgd_ead)

    loan_data_preprocessed['recovery_rate'] = loan_data_preprocessed['recovery_rate_st_1'] * loan_data_preprocessed['recovery_rate_st_2']
    loan_data_preprocessed['recovery_rate'] = np.where(loan_data_preprocessed['recovery_rate'] < 0, 0, loan_data_preprocessed['recovery_rate'])
    loan_data_preprocessed['recovery_rate'] = np.where(loan_data_preprocessed['recovery_rate'] > 1, 1, loan_data_preprocessed['recovery_rate'])

    loan_data_preprocessed['LGD'] = 1 - loan_data_preprocessed['recovery_rate']

    loan_data_preprocessed['CCF'] = reg_ead.predict(loan_data_preprocessed_lgd_ead)

    loan_data_preprocessed['CCF'] = np.where(loan_data_preprocessed['CCF'] < 0, 0, loan_data_preprocessed['CCF'])
    loan_data_preprocessed['CCF'] = np.where(loan_data_preprocessed['CCF'] > 1, 1, loan_data_preprocessed['CCF'])

    loan_data_preprocessed['EAD'] = loan_data_preprocessed['CCF'] * loan_data_preprocessed_lgd_ead['funded_amnt']

    return loan_data_preprocessed

# predictions for a customer inputs
def final_predictions(var_data_dict):

    pd_features = [ 'grade',
                    'home_ownership',
                    'addr_state',
                    'verification_status',
                    'purpose',
                    'initial_list_status',
                    'term',
                    'emp_length',
                    'issue_d',
                    'int_rate',
                    'earliest_cr_line',
                    'inq_last_6mths',
                    'acc_now_delinq',
                    'annual_inc',
                    'dti',
                    'mths_since_last_delinq',
                    'mths_since_last_record']

    lgd_features =         [    'grade',
                                'home_ownership',
                                'verification_status',
                                'purpose',     
                                'initial_list_status',
                                'term',
                                'emp_length',
                                'issue_d',
                                'earliest_cr_line',
                                'funded_amnt',
                                'int_rate',
                                'installment',
                                'annual_inc',
                                'dti',
                                'delinq_2yrs',
                                'inq_last_6mths',
                                'mths_since_last_delinq',
                                'mths_since_last_record',
                                'open_acc',
                                'pub_rec',
                                'total_acc',
                                'acc_now_delinq',
                                'total_rev_hi_lim']

    pd_input_var_dict = {key: var_data_dict[key] for key in pd_features if key in var_data_dict}
    lgd_input_var_dict = {key: var_data_dict[key] for key in lgd_features if key in var_data_dict}

    # customizing the input var values
    ## fro PD
    pd_input_df = customize_data_pd(pd_input_var_dict)
    ## for LGD, EAD and EL
    lgd_input_var_dict['term_int'] = lgd_input_var_dict.pop('term')
    lgd_input_var_dict['emp_length_int'] = lgd_input_var_dict.pop('emp_length')
    lgd_input_var_dict['mths_since_issue_d'] = lgd_input_var_dict.pop('issue_d')
    issue_date = datetime.strptime(lgd_input_var_dict['mths_since_issue_d'], "%Y-%m-%d")
    end_date = datetime.strptime('2017-12-01', "%Y-%m-%d")
    lgd_input_var_dict['mths_since_issue_d'] = end_date.year * 12 + end_date.month - (issue_date.year * 12 + issue_date.month)
    lgd_input_var_dict['mths_since_earliest_cr_line'] = lgd_input_var_dict.pop('earliest_cr_line')
    issue_date = datetime.strptime(lgd_input_var_dict['mths_since_earliest_cr_line'], "%Y-%m-%d")
    end_date = datetime.strptime('2017-12-01', "%Y-%m-%d")
    lgd_input_var_dict['mths_since_earliest_cr_line'] = end_date.year * 12 + end_date.month - (issue_date.year * 12 + issue_date.month)
    lgd_input_df = customize_data_lgd(lgd_input_var_dict)

    # loading models
    lgd_model_stage_1 = pickle.load(open(r'C:\Shubham\credit_scorecard\src\lgd_model_stage_1.sav',"rb"))
    lgd_model_stage_2 = pickle.load(open(r'C:\Shubham\credit_scorecard\src\lgd_model_stage_2.sav',"rb"))
    reg_ead = pickle.load(open(r'C:\Shubham\credit_scorecard\src\reg_ead.sav',"rb"))
    reg_pd = pickle.load(open(r'C:\Shubham\credit_scorecard\src\lr_pd_model_with_pvalue_filtered_features.sav', 'rb'))

    # make predictions
    loan_data_preprocessed_lgd_ead = lgd_input_df.copy()
    loan_data_preprocessed = loan_data_preprocessed_lgd_ead.copy()

    loan_data_preprocessed = expected_loss_estimation(lgd_model_stage_1,lgd_model_stage_2,reg_ead,loan_data_preprocessed,loan_data_preprocessed_lgd_ead)

    # predicting default for a customer
    def default_prediction(input_df,model):

        pred_dict = {
            "default_prediction" : model.predict(input_df),
            "default_prob" : model.predict_proba(input_df)
        }

        return pred_dict

    pred_pd = default_prediction(pd_input_df,reg_pd)
    loan_data_preprocessed['PD'] = pred_pd['default_prob'][0][1]
    loan_data_preprocessed['EL'] = loan_data_preprocessed['PD'] * loan_data_preprocessed['LGD'] * loan_data_preprocessed['EAD']
    loan_data_preprocessed['EL/funded_amnt'] = loan_data_preprocessed['EL']/loan_data_preprocessed['funded_amnt']
    
    # creating dictionary of predicted values
    pred_values = {
        "Prob. of Default" : loan_data_preprocessed['PD'][0],
        "Recovery Rate" : loan_data_preprocessed['recovery_rate'][0],
        "Loss Given Default" : loan_data_preprocessed['LGD'][0],
        "Credit Conversion Factor" : loan_data_preprocessed['CCF'][0],
        "Exposure at Default" : loan_data_preprocessed['EAD'][0],
        "Expected Loss" : loan_data_preprocessed['EL'][0],
        "EL/Funded Amount" : loan_data_preprocessed['EL/funded_amnt'][0],
    }

    return pred_values