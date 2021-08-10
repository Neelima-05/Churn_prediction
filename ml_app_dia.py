import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

pickle_out = open('XGBoost.pkl', 'wb')
classifier = pickle.load(pickle_out)

st.sidebar.header('Churn Prediction')
select = st.sidebar.selectbox('Select Form', ['Form 1'], key='2')
if not st.sidebar.checkbox("Hide", True, key='2'):
    st.title('Customer Churn Prediction')
    EmailsSent = st.text_input("Email Sent:")
    PUPDismiss = st.number_input("PUP dismiss:")
    OverTime = st.number_input("Over time:")
    SmsSent =  st.number_input("SMS Sent:")
    ProfilesForwarded = st.number_input("Profile Forwarded:")
    EventsCreated = st.number_input("Events Created:")
    PastApplicant = st.number_input("Past Applicant:")
    ResumeUploaded = st.number_input("Resume Upload:")
    ViewAttachments = st.number_input("View Attachments:")
    CampaignsRun = st.text_input("Campaigns Run:")
    SavedAI-JobMatchingCriteria = st.number_input("SavedAI-Job Matching Criteria:")
    NotesAdded = st.number_input("Notes added:")
    JobRecommendationList =  st.number_input("Job Recommendation List:")
    AddToJob-CRM = st.number_input("AddToJob-CRM:")
    EventsShared = st.number_input("Events Shared:")
    TagsAdded = st.number_input("Tags Added:")

    submit = st.button('Predict')

    if submit:
        prediction = classifier.predict([[EmailsSent,PUPDismiss,OverTime,SmsSent,ProfilesForwarded,EventsCreated,PastApplicant,ResumeUploaded,ViewAttachments,CampaignsRun,SavedAI-JobMatchingCriteria,NotesAdded,
  JobRecommendationList,AddToJob-CRM,EventsShared,TagsAdded]])
        if prediction == 1:
            st.write('Congratulation the customer wont churn')
        else:
            st.write(name," The customer will churn. But don't lose hope we have some recommendations for preventing the Churn:")
