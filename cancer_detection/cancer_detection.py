import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report
import pickle 

#all this can also be done using a neural netwrok instead of this model

def createModel(data):
    X = data.drop(["diagnosis"],axis=1)
    y = data["diagnosis"]
    
    #some columns have data with very large values and some very small, so we need to SCALE the data
    #we apply standard scaler which means we make the mean as 0 and std deviation as 1 for a column(not necessarily a normal distribution ,c a  be skewed depending upon the values in the column)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # no need to scale y as it is already 0 and 1
    X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,random_state =42, test_size=0.2)
    
    #creating model
    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    
    y_pred = lr.predict(X_test)

   
    
    #cheking the accuracy
    accuracy = accuracy_score(y_test,y_pred)
    print(f"accuracy : {accuracy: .2f}")
    print(f"classification_report: {classification_report(y_test,y_pred)}")
   
    return lr, scaler

#EXPLORATORY DATA ANALYSIS

def getCleanData():
    df = pd.read_csv("../cancer_detection/cancer.csv") # should be in the same directory as the file
    df = df.drop(["Unnamed: 32","id"],axis =1) # axis = 1 means column, axis = 0 means row
    df["diagnosis"] = df["diagnosis"].map({"M":1,"B":0}) #encoding M and B to 1's and 0's to convert it into numbers
    print(df.head())
    print(df.info()) # this gives us the listof columns with number of non null values, data tyoes of values
    # after these operation we have all the columns with non null values, so we dont need to replace NANs with the mean of the colums
    # all columns have values with datatypes as int or float(desired)
    # we succesfully removed the non required columns such as id and unnamed:32, which dont play any role in predicting if the person has cancer or not

    return df
                                

def main():
    data = getCleanData()

    model, scaler  = createModel(data)

    with open("model.pkl","wb") as f:  # it is a binary file so wb
        pickle.dump(model,f)

    with open("scaler.pkl","wb") as g:
        pickle.dump(scaler,g)    




if __name__ == "__main__":  # we cover the execution inside the main() function and this condition will only be true if we directly run this file and this is not imported from some other file, so that if its imported, then the execution of this file will not happen there
    main()

# we export model and scaler to predict in our application, if we directly run this code in our application, it would be like training the model again and again in our appliaction, makes application slower
# we export the model into a binary file and import that binary file into our application.
# for this we have to install a package called pickle     