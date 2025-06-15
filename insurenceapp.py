
import pickle
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler

mi1=pd.read_csv("C:/Users/shanm/Downloads/medical_insurance (2).csv")

# Initialize StandardScaler
with open("C:/Users/shanm/OneDrive/Desktop/Insurence/scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)

# Load the model from file
with open("C:/Users/shanm/OneDrive/Desktop/Insurence/model1.pkl", 'rb') as file:
    model1 = pickle.load(file)


#set home page with titles of other pages
st.sidebar.title('HOME')
page=st.sidebar.radio("Getpage",["Project Info","View Dataset","Exploratory Data Analysis","Final Regressor model ",
                                 "Know Your Insurence Price",
                                    "Who Creates"])
#page 1 is selected
if page=="Project Info":
    st.title("MEDICAL INSURENCE COST PREDICTON")
    #put one image 
    st.image("https://miro.medium.com/v2/resize:fit:1100/format:webp/0*t-YEeLunFUQy12oa.png")
    # give a intro to the website
    st.write(""" The Insurance Cost Prediction App leverages cutting-edge machine learning to help users 
             estimate insurance premiums based on key factors such as age, health, location, and coverage type.
             Insurance costs can be complex, but this app simplifies the process by analyzing past data 
             and predicting costs with high accuracy. By integrating AI-powered algorithms, users receive 
             personalized estimates to make informed financial decisions quickly and efficiently. """)

    st.write("""To ensure accuracy, the app evaluates multiple machine learning models, including Decision Tree, Random Forest,
             LightGBM, and XGBoost. Each model is tested to determine the best-performing one for predicting insurance costs.
             The app utilizes **MLflow** for tracking experiments, logging model parameters, and deploying the optimal model seamlessly.
             This enables continuous model improvement by retraining on new data and optimizing hyperparameters for better precision. """)

    st.write(""" After selecting the best model, the app allows users to input essential details and generates instant 
             insurance cost predictions. Through this smart automation, users gain valuable insights into their expected premium rates,
             helping them compare options and choose the best coverage. By combining machine learning with real-time prediction capabilities, 
             the app ensures transparency, efficiency, and accessibility in financial planning. 
 """)
  
   
elif page=="View Dataset":
    st.header('MEDICAL INSURENCE COST PREDICTION DATASET')
    st.image("https://online.maryville.edu/wp-content/uploads/sites/97/2020/10/MVU-MSDSCI-2020-Q1-Skyscraper-Predictive-Analytics-in-Insurance-Types-Tools-and-the-Future-header-v1.jpg")
    
    st.write(mi1)



elif page=="Exploratory Data Analysis":
    Query=st.selectbox("Univariate Analysis",["select below queries","What is the distribution of medical insurance charges?",
                                         "What is the age distribution of the individuals?",
                                         "How many people are smokers vs non-smokers?",
                                         "What is the average BMI in the dataset?",
                                         "Which regions have the most number of policyholders?"])
    if Query=="What is the distribution of medical insurance charges?":
        plt.figure(figsize=(8, 5))
        sns.histplot(mi1['charges'], bins=30, kde=True)
        plt.xlabel("Medical Insurance Charges")
        plt.ylabel("Frequency")
        plt.title("Distribution of Medical Insurance Charges")
        st.pyplot(plt)
    if Query=="What is the age distribution of the individuals?":
        plt.figure(figsize=(8, 5))
        sns.histplot(mi1['age'], bins=30, kde=True)
        plt.xlabel("age")
        plt.ylabel("Frequency")
        plt.title("Distribution of Individual Age")
        st.pyplot(plt)
    if Query=="How many people are smokers vs non-smokers?":
        plt.figure(figsize=(6, 5))
        sns.histplot(mi1['smoker'], bins=30)
        plt.xlabel("smoker")
        plt.ylabel("Frequency")
        plt.title("Smoking habit")
        st.pyplot(plt)
    if Query=="What is the average BMI in the dataset?":
        average_bmi = mi1['bmi'].mean()
        st.write(average_bmi)
    if Query=="Which regions have the most number of policyholders?":
        plt.figure(figsize=(6, 5))
        sns.histplot(mi1['region'], bins=30)
        plt.xlabel("region")
        plt.ylabel("Frequency")
        plt.title("Redion wise Distribution of Insurence Holers") 
        st.pyplot(plt)    


    Query=st.selectbox(" Bivariate Analysis",["select below queries","How do charges vary with age?",
                                         "Is there a difference in average charges between smokers and non-smokers?",
                                         "Does BMI impact insurance charges?",
                                         "Do men or women pay more on average?",
                                         "Is there a correlation between the number of children and the insurance charges?"])

    if Query=="How do charges vary with age?":
        age_charges = mi1.groupby('age')['charges'].mean().reset_index()

        plt.figure(figsize=(10, 5))
        plt.plot(age_charges['age'], age_charges['charges'], marker='o', linestyle='-', color='red')
        plt.xlabel('Age')
        plt.ylabel('Average Charges')
        plt.title(' Charges Variation with Age')
        plt.grid(True)
        st.pyplot(plt)
    if Query=="Is there a difference in average charges between smokers and non-smokers?":
        plt.figure(figsize=(10, 5))
        plt.scatter(mi1['smoker'], mi1['charges'], alpha=0.6, c='orange')
        plt.xlabel('smoker')
        plt.ylabel('Charges')
        plt.title(' smoking habit impact on Charges')
        plt.grid(True)
        st.pyplot(plt)

        smoker_charges = mi1.groupby('smoker')['charges'].mean().reset_index()
        plt.figure(figsize=(10, 5))
        plt.plot(smoker_charges['smoker'], smoker_charges['charges'], marker='o', linestyle='-', color='green')
        plt.xlabel('smoker')
        plt.ylabel('Average Charges')
        plt.grid(True)
        st.pyplot(plt)
    if Query=="Does BMI impact insurance charges?":
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=mi1['bmi'], y=mi1['charges'], alpha=0.6, color='blue')

        plt.xlabel('BMI')
        plt.ylabel('Charges')
        plt.title('Insurance Charges Distribution Based on BMI')
        plt.grid(True)
        st.pyplot(plt)

    if Query=="Do men or women pay more on average?":
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=mi1['sex'], y=mi1['charges'], alpha=0.6, color='blue')

        plt.xlabel('sex')
        plt.ylabel('Charges')
        plt.title(' Insurance Charges Distribution Based on Gender')
        st.pyplot(plt)

        sexwise_charges = mi1.groupby('sex')['charges'].mean().reset_index()
        plt.figure(figsize=(10, 5))
        plt.plot(sexwise_charges['sex'], sexwise_charges['charges'], marker='o', linestyle=' ', color='green')
        plt.xlabel('sex')
        plt.ylabel('Average Charges')
        plt.title(' Average  Charge Based on Gender')
        
        plt.show()

        st.pyplot(plt)        
    if Query=="Is there a correlation between the number of children and the insurance charges?":
        correlation_data = mi1[['children', 'charges']].corr()

        # Create a heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

        plt.title('Correlation Between Number of Children and Insurance Charges')

        st.pyplot(plt)
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=mi1['children'], y=mi1['charges'], alpha=0.6, color='blue')

        plt.xlabel('children')
        plt.ylabel('Charges')
        plt.title('children vs Insurance Charges')
        plt.grid(True)
        
        st.pyplot(plt)
    Query=st.selectbox(" Multivariate Analysis",["How does smoking status combined with age affect medical charges?",
                                         "What is the impact of gender and region on charges for smokers?",
                                         "How do age, BMI, and smoking status together affect insurance cost?",
                                         "Do obese smokers (BMI > 30) pay significantly higher than non-obese non-smokers?"])
    if Query=="How does smoking status combined with age affect medical charges?":
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=mi1['age'], y=mi1['charges'], hue=mi1['smoker'], alpha=0.6, palette='coolwarm')

        plt.xlabel('Age')
        plt.ylabel('Medical Charges')
        plt.title('Age & Smoking Status vs. Charges')
        plt.grid(True)
        st.pyplot(plt)
        plt.figure(figsize=(12, 6))
        sns.swarmplot(x=mi1['smoker'], y=mi1['charges'], hue=mi1['age'], palette='coolwarm', dodge=True)

        plt.xlabel('Smoking Status')
        plt.ylabel('Medical Charges')
        plt.title('Age & Smoking Status vs. Charges')
        st.pyplot(plt)

    if Query=="What is the impact of gender and region on charges for smokers?":
        smokers_mi1 = mi1[mi1['smoker'] == 'yes']

# Create a box plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=smokers_mi1['region'], y=smokers_mi1['charges'], hue=smokers_mi1['sex'], palette='coolwarm')

        plt.xlabel('Region')
        plt.ylabel('Medical Charges')
        plt.title('Impact of Gender & Region on Charges for Smokers')
        plt.legend(title='sex')
        plt.grid(True)
        st.pyplot(plt)

        smokers_mi1 = mi1[mi1['smoker'] == 'no']
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=smokers_mi1['region'], y=smokers_mi1['charges'], hue=smokers_mi1['sex'], palette='coolwarm')

        plt.xlabel('Region')
        plt.ylabel('Medical Charges')
        plt.title('Impact of Gender & Region on Charges for Non Smokers')
        plt.legend(title='sex')
        plt.grid(True)
        st.pyplot(plt)

    if Query=="How do age, BMI, and smoking status together affect insurance cost?":
        smokers_mi1 = mi1[mi1['smoker'] == 'yes']

        # Correct groupby syntax (charges & age should be a list)
        smokers_grouped = smokers_mi1.groupby(['bmi'])[['charges', 'age']].mean().reset_index()

        # Scatter plot
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x=smokers_grouped['bmi'], y=smokers_grouped['charges'], hue=smokers_grouped['age'], palette='coolwarm', size=smokers_grouped['charges'], sizes=(20, 200))

        plt.xlabel('BMI')
        plt.ylabel('Average Medical Charges')
        plt.title('Impact of BMI & Age on Charges for Smokers')

        # Set x-axis and y-axis range
        plt.xlim(10, 100)  
        plt.ylim(5000, 70000)  

        plt.xticks(rotation=45)  # Rotate x labels for better readability
        st.pyplot(plt)

    if Query=="Do obese smokers (BMI > 30) pay significantly higher than non-obese non-smokers?":
        # mi1['Obese_Smoker'] = (mi1['bmi'] > 30) & (mi1['smoker'] == "yes")
        smokers_mi1 = mi1[mi1['smoker'] == 'yes']
        smokers_mi1['Obesity_Status'] = smokers_mi1['bmi'].apply(lambda x: 'Obese_smoker' if x > 30 else 'Non-Obese_smoker')

        plt.figure(figsize=(6, 3))
        sns.boxplot(x=smokers_mi1['Obesity_Status'], y=smokers_mi1['charges'], palette='coolwarm')
        plt.xlabel('Obesity Status')
        plt.ylabel('Medical Charges')
        plt.title('Impact of Obesity on Charges for Smokers')
        st.pyplot(plt)

        non_smokers_mi1 = mi1[mi1['smoker'] == 'no']
        non_smokers_mi1['Non_Obesity_Status'] = non_smokers_mi1['bmi'].apply(lambda x: 'non_Obese_non_smoker' if x <= 30 else 'Obese_non_smoker')

        plt.figure(figsize=(6, 3))
        sns.boxplot(x=non_smokers_mi1['Non_Obesity_Status'], y=non_smokers_mi1['charges'], palette='coolwarm')
        plt.xlabel('Obesity Status')
        plt.ylabel('Medical Charges')
        plt.title('Impact of Obesity on Charges for Smokers')
        st.pyplot(plt)

    Query=st.selectbox(" Correlation Analysis",["What is the correlation between numeric features like age, BMI, number of children, and charges?",
                                         "Which features have the strongest correlation with the target variable (charges)?"])
    if Query=="What is the correlation between numeric features like age, BMI, number of children, and charges?":
        corr_matrix = mi1[['age', 'bmi','children', 'charges']].corr()
        st.write(corr_matrix)

        # Plot correlation heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Heatmap: Age, BMI, Children & Charges')
        plt.show()
        st.pyplot(plt)

    if Query=="Which features have the strongest correlation with the target variable (charges)?":
        mi1["sex"] = mi1["sex"].map({"male": 1, "female": 0})
        mi1["sex"] = mi1["sex"].astype(int)
        mi1["smoker"] = mi1["smoker"].map({"yes": 1, "no": 0})
        mi1["smoker"] = mi1["smoker"].astype(int)
        from sklearn.preprocessing import LabelEncoder

        df = pd.DataFrame({'region': ['region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']})
        encoder = LabelEncoder()
        mi1['region'] = encoder.fit_transform(mi1['region'])
        corr_matrix = mi1.corr()
        st.write(corr_matrix['charges'].sort_values(ascending=False))


        























elif page=="Final Regressor model ":
    mi1=pd.read_csv("C:/Users/shanm/Downloads/medical_insurance (2).csv")
    st.header("RandomForestRegressor")
    mi1["sex"] = mi1["sex"].map({"male": 1, "female": 0})
    mi1["sex"] = mi1["sex"].astype(int)
    mi1["smoker"] = mi1["smoker"].map({"yes": 1, "no": 0})
    mi1["smoker"] = mi1["smoker"].astype(int)
    from sklearn.preprocessing import LabelEncoder

    df = pd.DataFrame({'region': ['region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']})
    encoder = LabelEncoder()
    mi1['region'] = encoder.fit_transform(mi1['region'])

    mi1 = mi1.drop_duplicates(subset=["age", "sex","bmi","children","smoker", 'region', 'charges'], keep="first")

    
    from sklearn.preprocessing import StandardScaler

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Apply StandardScaler only to the 'Age' column
    mi1['charges'] = scaler.fit_transform(mi1[['charges']])
    y = mi1["charges"]
    X = mi1.drop(columns="charges")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 1)
    # X_train.shape, y_train.shape, X_test.shape, y_test.shape

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.ensemble import RandomForestRegressor
    Randfor = RandomForestRegressor(max_depth= 3,min_samples_leaf= 1, min_samples_split= 2,  n_estimators= 100, random_state=42)
    Randfor.fit(X_train, y_train)

    # Predictions
    # y_pred = Randfor.predict(X_test)
    y_train_pred = Randfor.predict(X_train)
    y_test_pred = Randfor.predict(X_test)
    # Evaluation Metrics
    r2 = r2_score(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)

    # Evaluate model performance
    r2_trainRF = r2_score(y_train, y_train_pred)
    r2_testRF = r2_score(y_test, y_test_pred)
    mae_trainRF = mean_absolute_error(y_train, y_train_pred)
    mae_testRF = mean_absolute_error(y_test, y_test_pred)
    mse_trainRF = mean_squared_error(y_train, y_train_pred)
    mse_testRF = mean_squared_error(y_test, y_test_pred)
    st.subheader("Feature Importance")
    feature_importances = Randfor.feature_importances_

    # Create DataFrame
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

    # Sort by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Iterate through rows correctly


    
 
    # Create a bar plot
    plt.figure(figsize=(8, 5))
    sns.barplot(x=feature_importances, y=X_train.columns, palette='coolwarm')
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.title("Feature Importance - RandomForestRegressor")
    st.pyplot(plt)
    # Iterate through rows correctly
    for i, row in importance_df.iterrows():
        st.write(f" **{row['Feature']}**:  **{row['Importance']}**")
    
    if st.button("r2_train & r2_test"):
        st.write(f"Train R² Score: {r2_trainRF}, Test R² Score: {r2_testRF}")
    if st.button("mae_train & mae_test"):
        st.write(f"Train MAE: {mae_trainRF}, Test MAE: {mae_testRF}")
    if st.button("mse_train & mse_test"):
        st.write(f"Train MSE: {mse_trainRF}, Test MSE: {mse_testRF}")



elif page=="Know Your Insurence Price":
    st.subheader("Give Details And Get Your Insurence Price")
       #age  sex     bmi  children  smoker  region   charges
    # st.title("Customer Data Collection")

    age=st.number_input("Enter your age", min_value=0)

    gender_mapping = {"Male": 1, "Female": 0}
    sex = st.selectbox("Select your gender",["Male","Female"])
    sex_numeric = gender_mapping[sex] 

    bmi=st.number_input("Enter your bmi ",  min_value=0.0, format="%.4f")

    children=st.number_input("Enter number of children", min_value=0)

    smoker_mapping = {"Yes": 1, "No": 0}
    smoker = st.selectbox("Are You Smoker",["Yes","No"])
    smoker_numeric = smoker_mapping[smoker]

    Region_mapping = {"northeast":0,"northwest": 1, "southeast": 2,"southwest":3}
    Region = st.selectbox("Select your Region",["northeast","northwest","southeast","southwest"])
    Region_numeric = Region_mapping[Region] 
    data=(age,sex_numeric,bmi,children,smoker_numeric,Region_numeric)


    if data and st.button('PREDICT'):

        
        input_data = np.asarray(data).reshape(1, -1)
       
        # Make predictions
        prediction_scaled = model1.predict(input_data)
        prediction=scaler.inverse_transform(np.array(prediction_scaled).reshape(1, -1))
        st.write(prediction)


elif page=="Who Creates":
    col1, col2 = st.columns(2)			 																				

    with col1:
        st.image("https://thumbs.dreamstime.com/b/chibi-style-d-render-devops-engineer-playful-young-male-character-laptop-table-isolated-white-background-rendered-361524322.jpg")
    st.write("I am Shanmugasundaram, and this is my first Machine Learning project after " \
        "joining the Data Science course on the Guvi platform. This project marks the beginning" \
        " of my journey into the world of data-driven decision-making and predictive analytics. " \
        "Through this project, I aim to apply the concepts learned in my course and build a model that provides " \
        "meaningful insights.")
    st.write("""Coming from an engineering background, I have always been intrigued by problem-solving,
                  automation, and analytical thinking. Machine Learning fascinates me as it combines mathematics, 
                 programming, and real-world applications to transform raw data into meaningful insights.""")

    st.write("""Coming from an engineering background, I have always been intrigued by problem-solving, 
                 automation, and analytical thinking. Machine Learning fascinates me as it combines mathematics,
                 programming, and real-world applications to transform raw data into meaningful insights.""")

                
st.sidebar.markdown("[Importance of Health Insurence ](https://www.aha.org/guidesreports/report-importance-health-coverage) ")
