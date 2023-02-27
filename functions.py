import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import statistics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score


class Functionalities:
    
    def __init__(self, dataset_path):
        
        self.dataset_path = dataset_path
        self.data = pd.read_csv(self.dataset_path, delimiter=',')
        self.model_names = ["LogisticRegression", "KNeighborsClassifier", "DecisionTreeClassifier", "Support Vector Classification", "RandomForestClassifier", "GaussianNB"]
    
    def get_data(self):
        return self.data
    
    def age_preprocess(self):
        droped_na = (((self.data)["Age"]).dropna())
        median1 = droped_na[(self.data)["Pclass"]==1].median()
        median2 = droped_na[(self.data)["Pclass"]==2].median()
        median3 = droped_na[(self.data)["Pclass"]==3].median()

        (self.data)["Age"] = (self.data)["Age"].fillna(0)
        for i in range((self.data).shape[0]):
            if (self.data).Age[i] == 0:
                if (self.data)["Pclass"][i] == 1:
                    (self.data).Age[i] = median1
                elif (self.data)["Pclass"][i] == 2:
                    (self.data).Age[i] = median2
                else:
                    (self.data).Age[i] = median3
        return self.data
    
    
    def cabin_preprocess(self):
        (self.data)["Cabin"] = (self.data)["Cabin"].fillna(0)
        (self.data)["Cabin"] = (self.data).Cabin.apply(lambda x: 1 if x != 0 else x)
        return self.data
    
    
    def remove_nan(self):
        return (self.data).dropna()
    
    
    def sex_preprocess(self):
        (self.data)["Sex"] = (self.data)["Sex"].apply(lambda x: 1 if x == "female" else 0)
        return self.data
    
    
    def embark_preprocess(self):
    
        dummies = pd.get_dummies((self.data)["Embarked"])
        (self.data)["S"] = dummies["S"]
        (self.data)["C"] = dummies["C"]
        (self.data).drop(columns="Embarked", inplace=True)
        return self.data
    
    
    def ticket_preprocess(self):
        (self.data)["Ticket"] = [1 if i is False else 0 for i in list((self.data)['Ticket'].duplicated())]
        return self.data
    
    
    def drop_name(self):
        return (self.data).drop(columns="Name", inplace=True)
    
    
    def normalize_fare(self):
        (self.data)["Fare"] = (preprocessing.normalize([((self.data)["Fare"])]))[0]
        return self.data
     
    def fare_preprocess(self):
        avg_mean = (self.data)["Fare"].mean()
        (self.data)["Fare"] = (self.data)["Fare"].fillna(avg_mean)
        return self.data
    
    def surv_plot(self):
        
        fig, ax = plt.subplots(figsize=(6,6))
        
        survival_count = [(self.data).Survived.value_counts()[1], (self.data).Survived.value_counts()[0]]
        explode = [0.1, 0]
        labels = ["Survived", "Died"]

        ax.pie(survival_count, explode=explode, labels=labels, autopct='%1.2f%%', shadow=True,startangle=45)
        ax.axis('equal')
        ax.set_title("\nSurvival Percentage for Passengers Embarked on Titanic\n", fontsize=16, fontweight="bold")

        return ax
    
    
    def hist_pdf_plot(self, feature):
    
        mean = np.mean((self.data)[feature])
        standard_dev = statistics.stdev((self.data)[feature])
        median = np.median((self.data)[feature])

        plt.style.use('ggplot')

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,6)) #creating plot axes
        (values, bins, _) = ax.hist((self.data)[feature], bins=25, density=True, color='grey', alpha=.7, label="Histogram")

        bin_centers = 0.5*(bins[1:] + bins[:-1])

        pdf = stats.norm.pdf(x = bin_centers, loc=mean, scale=standard_dev) #Compute probability density function
        ax.plot(bin_centers, pdf, label="PDF",color='darkblue') #Plot PDF
        ax.axvline(mean, color='darkgreen', linestyle='dashed', linewidth=2, label='mean')
        ax.axvline(median, color='orange', linestyle='dashed', linewidth=2, label='median')

        ax.legend()#Legend entries
        ax.set_title(f'Normal Distribution (PDF) and Histogram representation of {feature}', fontsize=16)

        plt.tight_layout()

        return ax
    
    
    def avg_score(self, model, x_train_scaled, y_train):
        return cross_val_score(model, x_train_scaled, y_train, cv=5).mean()
    
    def scores(self, models, x_train_scaled, y_train):
        return [(self.avg_score(model, x_train_scaled, y_train)) for model in models]
    
    
    def eval_table(self, avgScores):
        return pd.DataFrame({
            "Model" : self.model_names,
            "Average Score" : avgScores}).sort_values(by="Average Score", ascending=False).reset_index(drop=True)
    
    
    def score_plot(self, avgScores):
    
        models_table = self.eval_table(avgScores)
        
        fig,ax = plt.subplots(figsize=(15,10))

        colors = ["#60d147", "#FFA500", "#d35de3", "#87CEEB", "#0000FF", "#FF0000"]

        plt.bar(models_table["Model"], models_table["Average Score"], color=colors)
        plt.xticks(rotation=30, fontsize=12)

        ax.set_title("\nModel Average Accuracy Score Evaluation\n", fontsize=16)
        ax.set_xlabel("Models", fontsize=14)
        ax.set_ylabel("Average Score", fontsize=14)

        for k, i in enumerate(ax.patches):
                val = models_table["Average Score"][k]

                plt.text((i.get_xy()[0] + .24), (val/2),
                f"{round((val*100),2)} %", fontsize=12, fontweight='bold',
                color ='white', rotation=0)
        return ax
    
    def submission_file(self, passID, pred_result):
        file = pd.DataFrame({"PassengerId" : passID, "Survived" : pred_result}, columns=["PassengerId", "Survived"])
        file.to_csv("Data/survival_submission.csv", index=False)