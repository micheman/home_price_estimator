import sys
import pandas as pd

print("The name of this script is: ", sys.argv[0])
print("Number of arguments: ", len(sys.argv))
print("The arguments are: " , str(sys.argv))

# cmd_line_args = {}
# cmd_line_args['Beds'] = sys.argv[1]
# cmd_line_args['#Bath'] = sys.argv[2]
# cmd_line_args['#HlfBath'] = sys.argv[3]
# cmd_line_args['Gar'] = sys.argv[4]
# cmd_line_args['TCP'] = sys.argv[5]
# cmd_line_args['YB'] = sys.argv[6]
# cmd_line_args['Pool'] = sys.argv[7]
# cmd_line_args['SF'] = sys.argv[8]
# cmd_line_args['Acres'] = sys.argv[9]

# instructions:
# need the python source file <rempest_cmd.py> (this file!)
# need the orig data source file <DF_Short_Listings.pkl> to get the column names for reference
# need the best model source file <random_forest_model.joblib>

# read the data source file
cmd = pd.read_pickle("DF_Short_Listings.pkl") # cmd: clean MLS data
data = cmd.drop(["Sold $","MLS","Address","City","Zip","$/SF",\
                 "List $","Sold Date","Sold $","SP%LP","CDOM"], axis=1)
# We only care about the column names
feature_names = data.columns
# target = cmd["Sold $"]
# data.head()
print(feature_names)

# onehouse = pd.DataFrame(sys.argv[1:9], columns=[feature_names])
onehouse = pd.DataFrame(columns=[feature_names])

# load up the command line arg to DF
onehouse.at[0,'Beds'] = sys.argv[1]
onehouse.at[0,'#Bath'] = sys.argv[2]
onehouse.at[0,'#HlfBath'] = sys.argv[3]
onehouse.at[0,'Gar'] = sys.argv[4]
onehouse.at[0,'TCP'] = sys.argv[5]
onehouse.at[0,'YB'] = sys.argv[6]
onehouse.at[0,'Pool'] = sys.argv[7]
onehouse.at[0,'SF'] = sys.argv[8]
onehouse.at[0,'Acres'] = sys.argv[9]


# # read the model
# oneHouse = X_test.head(1)
# oneHouse
# gb.predict(oneHouse)

# rf.to_pickle("./random_forest_model.pkl")
from joblib import dump, load
# dump(clf, 'filename.joblib') 
# dump(rf, 'random_forest_model.joblib') 

rfm = load('random_forest_model.joblib') 

# rfm = pd.read_pickle("random_forest_model.pkl")
oneprice = rfm.predict(onehouse)

print(f'${oneprice[0]}')