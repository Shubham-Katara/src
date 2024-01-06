from flask import Flask,request,render_template,jsonify
import helper_functions
import json
import pandas as pd
import numpy as np

pd.options.display.max_columns=None

app=Flask(__name__)

# route for home page
@app.route("/")
def index():
    return render_template("index.html")

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# @app.route("/predictdata",methods=["GET","POST"])
# def predict_datapoint():
#     """
#     Getting data and making prediction.
#     """
#     if request.method=="GET":
#         return render_template("home.html")
#     else:
#         # input_df = helper_functions.customdata(
#         #     grade = data.get("Grade"),
#         #     issue_d = data.get("IssueDate"),
#         #     int_rate = int(data.get("InterestRate")),
#         #     annual_inc = float(data.get("income"))
#         # )
#         data = request.get_json()
#         input_df =  helper_functions.customdata(
#             grade = data.get("grade"),
#             issue_d = data.get("issue_date"),
#             int_rate = int(data.get("interest_rate")),
#             annual_inc = float(data.get("income"))
#         )
#         # making predictions
#         print(input_df)
#         pred_dict = helper_functions.default_prediction(input_df.copy())
#         print(pred_dict)
#         # return render_template("home.html",result=round(pred_dict["default_prob"][0][1],2))
#         pred_dict = json.dumps(pred_dict, cls=NumpyEncoder)
#         return pred_dict, 200
    
@app.route("/finalpredictdata",methods=["GET","POST"])
def final_predict_datapoint():
    """
    Getting data and making prediction.
    """
    if request.method=="GET":
        return render_template("home.html")
    else:
        var_data_dict = request.get_json()
        # var_data_dict = {
        #     "grade" : data.get("grade"),
        #     "home_ownership" : data.get("home_ownership"),
        #     "addr_state" : data.get("addr_state"),
        #     "verification_status" : data.get("verification_status"),
        #     "purpose" : data.get("purpose"),
        #     "initial_list_status" : data.get("initial_list_status"),
        #     "term" : data.get("term"),
        #     "emp_length" : data.get("emp_length"),
        #     "issue_d" : data.get("issue_d"),
        #     "int_rate" : data.get("int_rate"),
        #     "earliest_cr_line" : data.get("earliest_cr_line"),
        #     "delinq_2yrs" : data.get("delinq_2yrs"),
        #     "inq_last_6mths" : data.get("inq_last_6mths"),
        #     "open_acc" : data.get("open_acc"),
        #     "pub_rec" : data.get("pub_rec"),
        #     "total_acc" : data.get("total_acc"),
        #     "acc_now_delinq" : data.get("acc_now_delinq"),
        #     "total_rev_hi_lim" : data.get("total_rev_hi_lim"),
        #     "annual_inc" : data.get("annual_inc"),
        #     "dti" : data.get("dti"),
        #     "mths_since_last_delinq" : data.get("mths_since_last_delinq"),
        #     "mths_since_last_record" : data.get("mths_since_last_record"),
        #     "funded_amnt" : data.get("funded_amnt"),
        #     "installment" : data.get("installment"),
        # }
        pred_values = helper_functions.final_predictions(var_data_dict)
        # return render_template("home.html",result=round(pred_dict["default_prob"][0][1],2))
        pred_dict = json.dumps(pred_values, cls=NumpyEncoder)
        return pred_dict, 200
        
if __name__=="__main__":
    app.run(host="0.0.0.0")