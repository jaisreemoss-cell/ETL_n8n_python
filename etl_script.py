import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv()

# Initialize the OpenAI client (automatically picks up OPENAI_API_KEY from environment)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file or environment variables.")

client = OpenAI(api_key=api_key)

def run_etl():
    input_file = "student_enrollment_raw.csv"
    output_file = "students_cleaned.csv"
    
    # Step 1 & 2: Load and read the CSV file
    print("Step 1 & 2: Loading and reading the CSV file...")
    try:
        df = pd.read_csv(input_file)
        print(f"-> Successfully loaded {len(df)} rows from {input_file}.")
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Please check the file path.")
        return

    # Convert dataframe to string format to send to OpenAI
    raw_data_string = df.to_string(index=False)
    
    # Step 3: OpenAI clean the data with strict rule against hallucinating missing fields
    print("Step 3: Sending data to OpenAI for cleaning...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert data cleaning and ETL assistant. Clean the provided student enrollment dataset "
                    "following these strict guidelines:\n"
                    "1. Name, Course, City, Fee_Paid, and Enrolled_Date: Fix spelling typos, formatting inconsistencies, and standardizations properly.\n"
                    "2. Phone: Check if each phone number is a valid 10-digit number (accounting for scientific notation errors if they represent valid numbers, or missing/NaN values). DO NOT guess or fabricate missing phone numbers. If a phone number is missing, blank, or invalid/less than 10 digits, set its value explicitly to 'Invalid'.\n"
                    "3. Email: Check if each email has a proper format (e.g., contains '@' and a domain extension like '.com'). DO NOT guess or fabricate missing emails. If an email is missing, blank, or malformed/incomplete, set its value explicitly to 'Invalid'.\n"
                    "4. Do NOT guess missing or invalid phone numbers or emails under any circumstances; mark them as 'Invalid'.\n\n"
                    "Return ONLY valid CSV data including the header row. Do not include markdown code fences or conversational text."
                ),
            },
            {
                "role": "user",
                "content": f"Here is the uncleaned student data:\n{raw_data_string}",
            },
        ],
        temperature=0.1,  # Low temperature for precise, deterministic cleaning
    )

    cleaned_csv_data = response.choices[0].message.content.strip()
    
    # Optional safety: Strip markdown code blocks if the model accidentally includes them
    if cleaned_csv_data.startswith("```"):
        cleaned_csv_data = cleaned_csv_data.split("```")[1]
        if cleaned_csv_data.startswith("csv"):
            cleaned_csv_data = cleaned_csv_data[3:].strip()

    # Step 4: Create the final csv file
    print(f"Step 4: Writing cleaned data to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_csv_data)

    print(f"-> ETL Pipeline complete! Final file saved as {output_file}")

if __name__ == "__main__":
    run_etl()