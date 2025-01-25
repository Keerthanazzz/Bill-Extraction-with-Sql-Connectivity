import os
import streamlit as st
import PyPDF2
import re
import pandas as pd
from datetime import datetime
from PIL import Image
from pytesseract import pytesseract
import google.generativeai as genai
import time
import google.api_core.exceptions
from io import BytesIO
import spacy
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Step 1: Database Setup
DATABASE_URL = 'sqlite:///file_storage.db'  # Using SQLite for the database
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Step 2: Define the FileStorage Model
class FileStorage(Base):
    __tablename__ = 'file_storage'
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_data = Column(LargeBinary, nullable=False)  # Store the entire file as binary
    upload_date = Column(DateTime, default=datetime.utcnow)

# Step 3: Create the table in the database
Base.metadata.create_all(engine)

def store_file_in_database(file):
    """Store the uploaded file in the database."""
    session = SessionLocal()  # Create a new session
    file_name = file.name  # Get the file name
    file_type = file.type  # Get the MIME type (e.g., "image/jpeg", "application/pdf")
    file_data = file.read()  # Read the file content as binary

    # Create a new instance of FileStorage to store in the database
    new_file = FileStorage(
        file_name=file_name,
        file_type=file_type,
        file_data=file_data,
        upload_date=datetime.utcnow()
    )
    # Add the file to the session and commit
    session.add(new_file)
    session.commit()
    session.close()

    st.success(f"File '{file_name}' has been uploaded successfully and stored in the database.")


# Specify the path to the Tesseract executable
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

EXCEL_FILE_PATH = r'D:\database.xlsx'

# Load a small English NLP model
nlp = spacy.load('en_core_web_sm')

# Inject custom CSS for UI styling
def inject_custom_css():
    css = """
    <style>
        .stApp {background-color: #FFFFFF;}
        .stApp h1, .stApp h2 {color: #010080;}
        .stButton > button {
            background-color: #010080;
            color: white;
            font-size: 22px;
            border-radius: 30px;
            cursor: pointer;
        }
        .stButton > button:hover {background-color: white; color: #010080;}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Extract text from images
def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

# Clean value, avoid unwanted characters
def clean_value(value):
    value = re.sub(r'[^\d.]', '', value).strip()
    return float(value) if value else 0.0

# Convert date formats to 'DD/MM/YYYY'
def convert_date_format(date_str):
    for fmt in ("%d-%b-%Y", "%d/%m/%Y", "%d-%b-%y", "%d/%m/%y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    return date_str

# Configure Google Generative AI
genai.configure(api_key="AIzaSyAXB9wD52Q8kRrlXd7m_uK8nhDe6MlNseg")
model = genai.GenerativeModel('gemini-1.5-flash')

# Process text with the generative model
def process_text_with_model(text):
    prompt = (
        "Extract invoice data such as invoice number, GST number, invoice date, product names, "
        "quantity, and prices without any additional text or unwanted data. Format the data clearly."
        "You are given the text extracted from an invoice or bill. Your task is to extract and accurately populate the following details "
        "for each product listed in the bill. Ensure that all values are exactly as mentioned in the bill, with no alterations or accompanying text. Exclude "
        "any installation and labour charges and related data from the dataframe.\n\n"
        
        "The extracted data should be formatted in the following structure:\n\n"
        
        "1. Invoice Number - Extract the invoice number or bill number. It may be labeled as 'Invoice Number', 'Inv No', 'Inv No:', 'Invoice No:', 'Bill No' or similar variations. Ensure the correct number is extracted and placed in the Invoice Number column, without any accompanying text.\n"
        "2. GST Number - Extract the GST Number exactly as mentioned in the bill, without any additional text.\n"
        "3. Invoice Date - Extract and convert the invoice date to 'DD/MM/YYYY' format, without any additional text.\n\n"
        
        "Product Details:\n"
        "4. Product Name - Extract the product name as mentioned in the bill. The name should be exactly as it appears, without any abbreviations or modifications.\n"
        "5. Quantity - Extract the quantity of the product as mentioned in the bill. The quantity should match exactly as shown.\n"
        "6. Actual Total Price - Extract the exact total amount of the product excluding GST. Ensure this value is accurate and matches the bill.\n"
        "7. SGST Rate - Extract the SGST rate (%) mentioned in the bill for that product. If not present, mark as '-'. Ensure that the rate is formatted as 'X%' (e.g., '18%') with no decimals.\n"
        "8. CGST Rate - Extract the CGST rate (%) mentioned in the bill for that product. If not present, mark as '-'. Ensure that the rate is formatted as 'X%' (e.g., '18%') with no decimals.\n"
        "9. IGST Rate - Extract the IGST rate (%) mentioned in the bill for that product. If not present, mark as '-'. Ensure that the rate is formatted as 'X%' (e.g., '18%') with no decimals.\n"
        "10. Total GST Rate - Calculate the total GST rate by summing SGST and CGST or IGST rates. Ensure that the total rate does not exceed 100% and is formatted as 'X%' (e.g., '18%').\n"
        "11. Total GST Amount - Calculate the total GST amount for the product based on the 'Actual Total Price' and 'Total GST Rate'. Ensure this value is formatted to two decimal places.\n"
        "12. Total Amount - Calculate the total amount, which is the sum of 'Actual Total Price' and 'Total GST Amount' for that product. Ensure this value is formatted to two decimal places.\n\n"
        
        "The extracted data should be formatted in the following structure:\n\n"
        "Invoice Number: [Invoice Number]\n"
        "GST Number: [GST Number]\n"
        "Invoice Date: [Invoice Date]\n\n"
        "Product Details:\n"
        "Product Name: [Product Name]\n"
        "Quantity: [Quantity]\n"
        "Actual Total Price: [Actual Total Price]\n"
        "SGST Rate: [SGST Rate]\n"
        "CGST Rate: [CGST Rate]\n"
        "IGST Rate: [IGST Rate]\n"
        "Total GST Rate: [Total GST Rate]\n"
        "Total GST Amount: [Total GST Amount]\n"
        "Total Amount: [Total Amount]\n\n"
        
        "If any data is missing, mark it as '-'. Ensure all values are correctly placed in the respective fields for EACH PRODUCT OF each bill, exactly as they appear in the bill, without any accompanying text.\n\n"
        "Extracted text:\n" + text
    )
    response = model.generate_content(prompt)
    return response.text

def parse_bill_details(text):
    data = []

    # Extract invoice number
    invoice_number_match = re.search(r'(?:Invoice Number|Bill No)[:\s]*(\S+)', text)
    invoice_number = invoice_number_match.group(1).strip() if invoice_number_match else '-'
    
    # Extract GST number
    gst_number_match = re.search(r'GST Number[:\s]*(\S+)', text)
    gst_number = gst_number_match.group(1).strip() if gst_number_match else '-'
    
    # Extract invoice date
    invoice_date_match = re.search(r'Invoice Date[:\s]*([\d/]+)', text)
    invoice_date = convert_date_format(invoice_date_match.group(1).strip()) if invoice_date_match else '-'
    
    # Extract product details, ensuring to capture both single-word and multi-word product names
    product_blocks = re.split(r'\n\s*\n', text)

    for block in product_blocks:
        if 'Product Name' in block and 'INSTALLATION' not in block and 'LABOUR' not in block:
            # Extract product name
            product_name_match = re.search(r'(?:Product Name|Description)[:\s]*([^\n]+)', block, re.IGNORECASE)
            product_name = product_name_match.group(1).strip() if product_name_match else '-'

            # Extract quantity
            quantity_match = re.search(r'Quantity[:\s]*(\d+)', block)
            quantity = int(quantity_match.group(1).strip()) if quantity_match else 0

            # Extract actual total price using various patterns to ensure proper extraction
            actual_total_price_match = re.search(r'(?:Total Price|Amount|Total)[:\s]*([0-9,.]+)', block, re.IGNORECASE)
            actual_total_price = clean_value(actual_total_price_match.group(1).strip()) if actual_total_price_match else 0.0

            # Extract SGST rate and ensure it's formatted as a percentage or '-'
            sgst_rate_match = re.search(r'SGST Rate[:\s]*(\d+)%', block)
            sgst_rate = sgst_rate_match.group(1).strip() if sgst_rate_match else '-'

            # Extract CGST rate and ensure it's formatted as a percentage or '-'
            cgst_rate_match = re.search(r'CGST Rate[:\s]*(\d+)%', block)
            cgst_rate = cgst_rate_match.group(1).strip() if cgst_rate_match else '-'

            # Extract IGST rate and ensure it's formatted as a percentage or '-'
            igst_rate_match = re.search(r'IGST Rate[:\s]*(\d+)%', block)
            igst_rate = igst_rate_match.group(1).strip() if igst_rate_match else '-'

            # Convert rates from percentages to integers, replace '0%' with '-'
            sgst = int(sgst_rate) if sgst_rate != '-' else 0
            cgst = int(cgst_rate) if cgst_rate != '-' else 0
            igst = int(igst_rate) if igst_rate != '-' else 0

            # Replace '0%' with '-'
            sgst_rate_display = '-' if sgst == 0 else f"{sgst}%"
            cgst_rate_display = '-' if cgst == 0 else f"{cgst}%"
            igst_rate_display = '-' if igst == 0 else f"{igst}%"

            # Calculate total GST rate and amounts
            total_gst_rate = sgst + cgst + igst
            total_gst_amount = round((quantity * actual_total_price * total_gst_rate / 100), 2)
            total_amount = round((quantity * actual_total_price) + total_gst_amount, 2)

            # Append data
            data.append([
                invoice_number, gst_number, invoice_date, product_name, quantity, actual_total_price,
                sgst_rate_display, cgst_rate_display, igst_rate_display, 
                f"{total_gst_rate}%", total_gst_amount, total_amount
            ])

    df = pd.DataFrame(data, columns=[
        "Invoice Number", "GST Number", "Invoice Date", "Product Name", "Quantity", "Actual Total Price",
        "SGST Rate", "CGST Rate", "IGST Rate", "Total GST Rate", "Total GST Amount", "Total Amount"
    ])
    
    return df

# Prevent duplicates and append data to Excel
def append_to_excel(df, excel_file_path):
    if os.path.exists(excel_file_path):
        existing_df = pd.read_excel(excel_file_path)
        combined_df = pd.concat([existing_df, df]).drop_duplicates(
            subset=["Invoice Number", "GST Number", "Product Name", "Quantity", "Actual Total Price"], keep='first'
        )
        combined_df.to_excel(excel_file_path, index=False)
        return combined_df
    else:
        df.drop_duplicates(subset=["Invoice Number", "GST Number", "Product Name", "Quantity", "Actual Total Price"], keep='first').to_excel(excel_file_path, index=False)
        return df

# Create monthly summary
def create_monthly_summary(df):
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Invoice Date'])
    df['Month'] = df['Invoice Date'].dt.to_period('M').astype(str)
    summary_df = df.groupby('Month').agg({
        'Actual Total Price': 'sum', 'Total GST Amount': 'sum', 'Total Amount': 'sum'
    }).reset_index()
    summary_df.columns = ['Month', 'Total Actual Price', 'Total GST Amount', 'Total Amount']
    return summary_df

# Convert DataFrame to Excel without timing in the 'Invoice Date'
def convert_df_to_excel(df, summary_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Ensure the 'Invoice Date' column is formatted correctly in Excel
        df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], format='%d/%m/%Y', errors='coerce').dt.strftime('%d/%m/%Y')

        # Write the extracted data to the first sheet
        df.to_excel(writer, index=False, sheet_name='Extracted Data')
        
        # Write the monthly summary to the second sheet
        summary_df.to_excel(writer, index=False, sheet_name='Monthly Summary')
        
    return output.getvalue()

# Advanced query analyzer
def analyze_query(df, query):
    query = query.lower()
    
    if 'price' in query:
        return df[['Product Name', 'Actual Total Price', 'Total Amount']]
    if 'date' in query:
        return df[['Invoice Number', 'Invoice Date']]
    if 'gst' in query:
        return df[['Product Name', 'SGST Rate', 'CGST Rate', 'IGST Rate']]
    
    # Search across all columns for any matching rows
    return df[df.apply(lambda row: query in row.to_string().lower(), axis=1)]

# Prevent duplicates and append data to Excel
def append_to_excel(df, excel_file_path):
    if os.path.exists(excel_file_path):
        existing_df = pd.read_excel(excel_file_path)
        combined_df = pd.concat([existing_df, df]).drop_duplicates(
            subset=["Invoice Number", "GST Number", "Product Name", "Quantity", "Actual Total Price"], keep='first'
        )
        combined_df.to_excel(excel_file_path, index=False)
        return combined_df
    else:
        df.drop_duplicates(subset=["Invoice Number", "GST Number", "Product Name", "Quantity", "Actual Total Price"], keep='first').to_excel(excel_file_path, index=False)
        return df
    
# Analyze the query and return the relevant DataFrame
def analyze_query(df, query):
    query_doc = nlp(query.lower())
    price_keywords = ["price", "amount", "cost", "total"]
    date_keywords = ["date", "invoice date"]
    gst_keywords = ["gst", "tax"]

    product = None
    action = None

    for token in query_doc:
        if token.text not in price_keywords + date_keywords + gst_keywords:
            product = token.text
        elif token.text in price_keywords:
            action = "price"
        elif token.text in date_keywords:
            action = "date"
        elif token.text in gst_keywords:
            action = "gst"

    if product:
        return df[df['Product Name'].str.contains(product, case=False, na=False)]
    elif action == "date":
        return df[['Invoice Number', 'Invoice Date']].drop_duplicates()
    elif action == "gst":
        return df[['Product Name', 'SGST Rate', 'CGST Rate', 'IGST Rate']].drop_duplicates()
    elif "invoice number" in query.lower():
        return df[['Invoice Number']].drop_duplicates()
    elif "gst number" in query.lower():
        return df[['GST Number']].drop_duplicates()
    else:
        return None
# Main Streamlit application
def main():
    # Inject custom CSS
    inject_custom_css()

    st.title("Invoice Processing and Data Aggregation App")

    # Display the upload button first
    uploaded_files = st.file_uploader("Upload PDF or Image files", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)

    all_dfs = []
    combined_df = None
    monthly_summary = None

    if uploaded_files:
        # Determine the number of columns based on the number of uploaded files
        cols = st.columns(min(len(uploaded_files), 3))  # Adjust the number of columns (e.g., 3 columns per row)

        for i, uploaded_file in enumerate(uploaded_files):
            # Process file based on type
            if uploaded_file.type == "application/pdf":
                extracted_text = extract_text_from_pdf(uploaded_file)
            else:
                extracted_text = extract_text_from_image(uploaded_file)
                # Display the image in the grid
                with cols[i % 3]:  # Change the number based on columns (e.g., 3)
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f'Uploaded Image {i+1}', use_column_width=True)
            
            # Process the extracted text
            processed_text = process_text_with_model(extracted_text)
            df_new = parse_bill_details(processed_text)
            all_dfs.append(df_new)

        # Combine all DataFrames if multiple files were processed
        if all_dfs:
            df_combined = pd.concat(all_dfs, ignore_index=True)

            # Append new data to the existing Excel file and clean duplicates
            combined_df = append_to_excel(df_combined, EXCEL_FILE_PATH)

            # Generate monthly summary
            monthly_summary = create_monthly_summary(combined_df)

    # Display the download button after processing files
    if uploaded_files and combined_df is not None:
        st.download_button(
            label='Download Updated Excel',
            data=convert_df_to_excel(combined_df, monthly_summary),
            file_name=EXCEL_FILE_PATH,
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

        # Display the Excel Data Analyzer part only after the download button
        st.subheader("Excel Data Analyzer")
        
        # Automatically load the existing Excel file for analysis
        if os.path.exists(EXCEL_FILE_PATH):
            df = pd.read_excel(EXCEL_FILE_PATH, engine='openpyxl')
            
            # Input field for user query
            query = st.text_input("Ask a question (e.g., 'show the laptop price'):")

            if query:
                # Analyze the user's query and retrieve the relevant DataFrame
                search_results = analyze_query(df, query)

                if search_results is not None and not search_results.empty:
                    # Display the search results
                    st.write("Search Results:")
                    st.dataframe(search_results)

                    # Check if multiple results are found
                    if len(search_results) > 1:
                        st.write("Multiple results found. Please provide additional details to filter the results.")
                        
                        # Additional filter inputs
                        invoice_number = st.text_input("Filter by Invoice Number (optional):")
                        gst_number = st.text_input("Filter by GST Number (optional):")
                        quantity = st.number_input("Filter by Quantity (optional):", min_value=0, step=1, format="%d")
                        invoice_date = st.text_input("Filter by Invoice Date (optional) (format: DD/MM/YYYY):")

                        # Apply additional filters based on user input
                        if invoice_number:
                            search_results = search_results[search_results['Invoice Number'].str.contains(invoice_number, case=False, na=False)]

                        if gst_number:
                            search_results = search_results[search_results['GST Number'].str.contains(gst_number, case=False, na=False)]

                        if quantity:
                            search_results = search_results[search_results['Quantity'] == quantity]

                        if invoice_date:
                            try:
                                invoice_date = pd.to_datetime(invoice_date, format='%d/%m/%Y', errors='coerce')
                                search_results['Invoice Date'] = pd.to_datetime(search_results['Invoice Date'], format='%d/%m/%Y', errors='coerce')
                                search_results = search_results[search_results['Invoice Date'] == invoice_date]
                            except ValueError:
                                st.write("Invalid date format. Please enter a valid date in DD/MM/YYYY format.")    

                        # Display filtered results
                        if not search_results.empty:
                            st.write("Filtered Results:")
                            st.dataframe(search_results)
                        else:
                            st.write("No matching results found after filtering.")
                    else:
                        st.write("No further filtering needed.")
                else:
                    st.write("No matching results found for your query.")
        else:
            st.write("No Excel file found. Please upload or process invoices first.")

if __name__ == "__main__":
    main()
