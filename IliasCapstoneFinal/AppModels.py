import itertools

import joblib
import numpy as np
import pandas as panda
from flask import render_template, request, Flask
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split

try:

    #Defines the read data as a dataset
    DataSet = panda.read_csv("C:\\Users\\Ilias\\OneDrive\\Desktop\\Capstone Final\\IliasCapstoneFinal\\static\\youtoxic_english_1000.csv")

    ContentData = DataSet["Text"]
    ContentClassToxic = DataSet["IsToxic"]
#Just prints here for Debugging + to test if correct info is produced
    #print (ContentData)
    #print (ContentClassToxic)
    #Defines the Training test
    ContentTrain, ContentTest,ToxicTrain,ToxicTest = train_test_split(ContentData,ContentClassToxic,test_size = 0.2)
    #Defines the vectorizor
    ContentVectorization = TfidfVectorizer()
    #This is the vecotrized content that has been trained
    VectorizedContentTrain = ContentVectorization.fit_transform(ContentTrain)
    #This is the vectorized Content test
    VectorizedContentTest = ContentVectorization.transform(ContentTest)

    #This defines the Machine learning model as a Content Algo
    ContentAlgo = LogisticRegression()
    #this basically fits both training
    ContentAlgo.fit(VectorizedContentTrain,ToxicTrain)

    #This defines the algorithm predictions
    #This will be used later
    AlgoPrediction = ContentAlgo.predict(VectorizedContentTest)

    #Just prints its been trained so I know how
    #The process works.
    print("Machine Learning Model Has been trained")
    print("Accuracy:", accuracy_score(ToxicTest, AlgoPrediction))
    #This is a dump that puts the Algo into the Models
    joblib.dump(ContentAlgo, "C:\\Users\\Ilias\\OneDrive\\Desktop\\Capstone Final\\IliasCapstoneFinal\\static\models\\ToxicOrNotClassifier.pkl")
    #This is a dump that puts the vectorizer into Models.
    joblib.dump(ContentVectorization, "C:\\Users\\Ilias\\OneDrive\\Desktop\\Capstone Final\\IliasCapstoneFinal\\static\models\\modelsContentVectorizer.pkl")

except Exception as e:
     print(f"Error:{e}")

#This is the comment
ContentData = DataSet["Text"]
#This decides if its toxic or not
ContentClassToxic = DataSet["IsToxic"]

ContentTrain, ContentTest,ToxicTrain,ToxicTest = train_test_split(ContentData,ContentClassToxic,test_size = 0.2)
#VectorizedContentTest = ContentVectorization.transform(ContentTest)

#Creating visuals for the Feature Importance
def FeatureImportanceToxicOrNot(ContentVectorization,ContentAlgo):
        #This gets the feature names from the content Vectorizer
        TONAFeatureName = ContentVectorization.get_feature_names_out()
        #This is basically Algorithm that is raveled
        ContentRavel = ContentAlgo.coef_.ravel()
        #This is as it says
        Positive_Coefficient = np.argsort(ContentRavel)[-20:]
        #This too
        Negative_Coefficient = np.argsort(ContentRavel)[:20]
        #Combination of both to be compared
        Top_Coefficient = np.hstack([Positive_Coefficient, Negative_Coefficient])
        #says the side
        plt.figure(figsize=(15, 7))
        colors = ["green" if c < 0 else "red" for c in ContentRavel[Top_Coefficient]]
        bars = plt.bar(np.arange(2 * 20), ContentRavel[Top_Coefficient], color=colors)

        #This displays the toxic words bars
        plt.text(10, max(ContentRavel[Top_Coefficient]) - .5, "Toxic Words", horizontalalignment="center", color="red", fontsize=14)
        #This displays the Non Toxic words bars
        plt.text(30, max(ContentRavel[Top_Coefficient]) - .5, "Not Toxic Words", horizontalalignment="center", color="green", fontsize=14)

        #Makes the Array
        TONAFeatureName = np.array(TONAFeatureName)
        #Defines the ticks
        plt.xticks(np.arange(1, 1+2 * 20), TONAFeatureName[Top_Coefficient], rotation=60, ha="right")
        plt.title("Top Used Features, Which indicate if its toxic or not")
        plt.tight_layout(pad=2.5)
        #Tells you to dump the image into the Images
        plt.savefig("C:\\Users\\Ilias\\OneDrive\\Desktop\\Capstone Final\\IliasCapstoneFinal\\static\\Images\\ToxicOrNoteFeatureImportance.png")
        #This is printed once the visual is created
        #Really useful for debugging
        print("Feature Importance Visual Created")

#This makes the Confusion matrix for the Toxic Or Not App
def ConfustionMatrixTONA(ToxicTest, AlgoPrediction):
        #This defines the confusion matrix with the test and the Machine learning Algorrithm.
        ConfusionMatrix = confusion_matrix(ToxicTest, AlgoPrediction)
        #Just says the size
        plt.figure(figsize=(7,7))
        #Color and more
        plt.imshow(ConfusionMatrix, interpolation="nearest", cmap=plt.cm.Blues)
        #Defines Title
        plt.title("Confusion Matrix")
        plt.colorbar()

  #numbers for each quadrent
        for i, j in itertools.product(range(ConfusionMatrix.shape[0]), range(ConfusionMatrix.shape[1])):
                plt.text(j, i, ConfusionMatrix[i, j],
                horizontalalignment="center",
                color="white" if ConfusionMatrix[i, j] > (ConfusionMatrix.max() / 2.) else "black")

        tick_marks = np.arange(2)
        #Makes Left Right Markings
        plt.xticks(tick_marks, ["Toxic", "Not Toxic"], rotation=45)
        #This is top down markings
        plt.yticks(tick_marks, ["Toxic", "Not Toxic"])
        plt.tight_layout(pad=2.5) #needs more padding, dont remove
        #This label that is true
        plt.ylabel("True Label")
        #The Machine learning models predicted
        plt.xlabel("Predicted Label")
        #This makes sure to save the image into the Images file
        plt.savefig("static/Images/ConfusionMatrixTONA.png")
        #This prints the confusion matrix
        #Is extremely useful for debugging
        print("Confusion Matrix visualization created")
#This is the final 3rd visualization
#Requirenment for the Visuals
def ROCCurveTONA(ToxicTest,Prediction_Probability ):
       #This defines the roc curve
        fpr, tpr, _ = roc_curve(ToxicTest, Prediction_Probability[:, 1])
        roc_auc = auc(fpr, tpr)
        #Makes the figure
        plt.figure()
       #Makes the color
        plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC Curve (Area = %0.2f)" % roc_auc)
        #Defines the other color
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        #The X label positive rate
        plt.xlabel("False Positive Rate")
       #This is true positive rate
        plt.ylabel("True Positive Rate")
        #The ROC curve in long
        plt.title("Receiver Operating Characteristic Curve")
        plt.legend(loc="lower right")
       #This saves the figure/image into the Images folder
        plt.savefig("static/Images/ROCCurveTONA.png")
       #Once its created this is printed
       #used for debugging
        print("ROC Curve visualization created")




#This is needed to define the stuff below and Additionally
#It is used to get parts of program to work
FeatureImportanceToxicOrNot(ContentVectorization, ContentAlgo)
AlgoPrediction = ContentAlgo.predict(VectorizedContentTest)
ConfustionMatrixTONA(ToxicTest, AlgoPrediction)
Prediction_Probability = ContentAlgo.predict_proba(VectorizedContentTest)
ROCCurveTONA(ToxicTest,Prediction_Probability )

print("Welcome to my program")

# NegativityApplication = Flask(__name__)
#NegativityApplication = Flask(__name__, template_folder='Templates')
ContentAlgo = joblib.load("C:\\Users\\Ilias\\OneDrive\\Desktop\\Capstone Final\\IliasCapstoneFinal\\static\models\\ToxicOrNotClassifier.pkl")
ContentVectorization = joblib.load("C:\\Users\\Ilias\\OneDrive\\Desktop\\Capstone Final\\IliasCapstoneFinal\\static\models\\modelsContentVectorizer.pkl")
#This defines the App folder layouts
app = Flask(__name__,template_folder='templates')
@app.route("/", methods=["GET", "POST"])
#This is basically the main part of the app
def ToxicOrNot():
    global TxtNumeric
    #Variable for the Negativity Toxic vs Non Toxic
    Negativity = None
    #Simmilar to the Negativity but with a different purpose
    Negativity_Status = None
    #Confidence for the class requirnment
    confi = None
    #This is used for a future feature
    Positivity = None
    #Basically gets the request from the submission
    if request.method == "POST":  # Form submission
        #This defines the text input
        TxtInput = request.form["comment"]  # Text Input
        #This is input vectorized
        VectorizedInput = ContentVectorization.transform([TxtInput])
        #This is used for predictions
        TxtInputPredict = ContentAlgo.predict(VectorizedInput)
        #For the probability
        TxtInputProb = ContentAlgo.predict_proba(VectorizedInput)
        #This is the confidence rating
        confi = round(TxtInputProb[0][list(ContentAlgo.classes_).index(TxtInputPredict[0])] * 100, 2)



        #TxtNumeric = int(TxtInputPredict)
        print("TxtInputPredict",TxtInputPredict)
        match TxtInputPredict:

        #Basically defines something as toxic or not toxic if TxtInput is
            #True or False
            case 0:
                Negativity_Status = "Not-Toxic"
                Negativity = "Not Toxic"
                print("Negativity:", Negativity)
            case 1:
                Negativity_Status = "Toxic"
                Negativity = "Toxic"

#Basically gathers stuff and initiates
    return render_template("ToxicOrNot.html", Negativity=Negativity, Negativity_Status=Negativity_Status, confi=confi)

    #This allows the HTML Page to be open
@app.route("/FeatureImportanceTONA")
#This allows to go from main menu to the Feature Importance
#section
def FeatureImportanceTONA():
    return render_template("FeatureImportanceTONA.html")

@app.route("/ConfusionMatrixTONA")
#This allows to go from main menu to the confusion matrix
def ConfusionMatrixTONA():
    return render_template("ConfusionMatrixTONA.html")

@app.route("/ROCCurveTONA")
#this allows to go from main menu to ROC Curve
def ROCCurveTONA():
    return render_template("ROCCurveTONA.html")
#This allows main program to be run
#The debug mode is off due to the amount of Errors created
if __name__ == "__main__":
    app.run()
