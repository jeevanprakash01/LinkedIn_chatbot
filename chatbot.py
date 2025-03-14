import streamlit as st
import requests
import pandas as pd
import openai
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Set API credentials
LINKEDIN_ACCESS_TOKEN = "YOUR_LINKEDIN_ACCESS_TOKEN"
OPENAI_API_KEY = "YOUR OPENAI_API_KEY"

HEADERS = {"Authorization": f"Bearer {LINKEDIN_ACCESS_TOKEN}"}

# Initialize OpenAI LLM
openai.api_key = OPENAI_API_KEY
llm = OpenAI(temperature=0.7)

# Prompt template for chatbot
prompt_template = PromptTemplate(
    input_variables=["query"],
    template="You are a chatbot that retrieves LinkedIn company data. Answer the user based on the provided company data: {query}"
)

chain = LLMChain(llm=llm, prompt=prompt_template)

# Function to fetch company data
def get_company_data(company_name):
    url = f"https://api.linkedin.com/v2/organizations/{company_name}"
    response = requests.get(url, headers=HEADERS)
    return response.json() if response.status_code == 200 else None

# Function to fetch employees
def get_company_employees(company_name):
    url = f"https://api.linkedin.com/v2/organizationAcls?q=roleAssignee"
    response = requests.get(url, headers=HEADERS)
    return response.json() if response.status_code == 200 else None

# Save data to CSV
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return filename

# Streamlit UI
st.title("LinkedIn AI Chatbot with OpenAI & LangChain")

# User input
company_name = st.text_input("Enter the Company name :", "")
user_query = st.text_input("Ask a question about the company:", "")

if st.button("Fetch Data & Chat"):
    if company_name:
        # Get company details
        company_data = get_company_data(company_name)
        employees_data = get_company_employees(company_name)

        if company_data:
            company_info = f"Company Name: {company_data.get('localizedName', 'N/A')}\nDescription: {company_data.get('description', 'N/A')}"
            filename = save_to_csv([company_data], "company_profile.csv")
            st.success(f"Company Profile saved to {filename}")
            st.download_button("Download Company Profile", open(filename, "rb"), file_name=filename)
        else:
            company_info = "Company data not found."

        if employees_data:
            filename = save_to_csv(employees_data['elements'], "company_employees.csv")
            st.success(f"Employee Data saved to {filename}")
            st.download_button("Download Employee Data", open(filename, "rb"), file_name=filename)

        # Generate chatbot response
        if user_query:
            chat_response = chain.run(query=f"{user_query} | Data: {company_info}")
            st.write("### Chatbot Response:")
            st.write(chat_response)
    else:
        st.error("Please enter a valid Company name.")
