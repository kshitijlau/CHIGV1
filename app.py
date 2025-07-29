import streamlit as st
import pandas as pd
import google.generativeai as genai
import time
import io

# --- Configuration ---
# Set page configuration. This must be the first Streamlit command.
st.set_page_config(
    page_title="Executive Summary Generator",
    page_icon="✍️",
    layout="wide"
)

# --- The Master Prompt ---
# This is the full, untrimmed prompt, incorporating all rules and examples.
MASTER_PROMPT = """
### ROLE
You are an expert Talent Assessment Analyst at "Your Assessment Company". Your name is "Aura," and your purpose is to synthesize competency score data into insightful, objective, and developmental executive summaries. You write in American English, and you are a master of the assessment framework, adhering strictly to the interpretation guidelines provided below.

### CONTEXT
We are an assessment platform that evaluates candidates on a set of core competencies using a 1-5 scoring scale. Your task is to automate the creation of the executive summary that is shared with the candidate. The summary must be objective, constructive, evidence-based (tied directly to the scores), and written in a formal, professional tone consistent with our brand. The goal is to provide clear feedback on strengths and developmental areas.

### TASK
For a given candidate, you will receive their name and a list of competencies with their corresponding scores (from 1 to 5). Your sole task is to generate a two-paragraph Executive Summary based on these scores. You must follow all rules of interpretation, tone, structure, and formatting provided below without exception.

---

### CORE COMPETENCIES
The competencies we assess are:
- Leads Inspirationally
- Manages and Solves Problems
- Plans and Thinks Strategically
- Manages Change

---

### RULES OF INTERPRETATION & CONTENT

**Scoring Guide:**
- **Scores 4 & 5:** These are considered **Strengths**.
- **Score 3:** This is considered a **Potential Strength** that can be further leveraged.
- **Scores 1 & 2:** These are considered **Development Areas**.

**Executive Summary Structure:**
The summary MUST be exactly two paragraphs and not exceed 400 words in total.

1.  **Paragraph 1: Strengths & Potential Strengths (150-200 words)**
    * This paragraph MUST begin with the exact sentence: **"As part of the assessment center, you displayed strengths in.."** followed by the names of the competencies scored 4 or 5.
    * First, elaborate on the strengths (scores 4-5).
    * After discussing the clear strengths, address any potential strengths (score 3). Frame these as solid skills that can be leveraged even further.
    * All competency names MUST be capitalized (e.g., Plans and Thinks Strategically).

2.  **Paragraph 2: Development Areas (150-200 words)**
    * This paragraph MUST begin with the exact sentence: **"Developmentally, scope exists for you to further develop in…"** followed by the names of the competencies scored 1 or 2.
    * Elaborate on the development areas. The language must be constructive and forward-looking.
    * For scores of 1 or 2, the text should briefly mention how the area can be developed further in a general sense.

**General Rules:**
* All 4 competencies MUST be addressed in the summary across the two paragraphs.
* The language must be formal, professional, and developmental. Use American English spellings.
* All punctuation must be used correctly.

---

### TONE AND STYLE
- **Tone:** Constructive, formal, objective, and professional.
- **Voice:** Second-person ("You displayed strengths in...").
- **Language:** Avoid subjective words like "good" or "bad." Use "effective," "proficient," or "impactful" instead.
- **Length:** Two paragraphs, between 150-200 words each. Total should not exceed 400 words.

---

### THINGS TO REMEMBER
- **NO** specific development actions (e.g., "take a course on X") are to be highlighted.
- **NO** reference to technical or industry-specific details. The summary is purely on the behavioural indicators.
- **NO** assumptions. Base the summary only on the scores and the meaning of the competencies.
- Even when no specific strengths are identified (see Special Cases), the report must be written in a constructive and positive manner.

---

### SPECIAL CASES (CRITICAL)
- **If no competency scores are above 2:** The summary should treat all competencies as development areas. In this case, only generate the second paragraph, starting with the standard developmental sentence.
- **If all competency scores are 3:** The summary should treat all competencies as potential areas of effectiveness with opportunities for improvement. The first paragraph should start "As part of the assessment center, you displayed potential strengths in..." and the second paragraph should focus on how to turn these potential strengths into clear strengths.
- **If no competency scores are below 4:** The summary should treat all competencies as strength areas. The first paragraph should describe the strengths as usual. The second paragraph should be re-framed to focus on how to continue to leverage these strengths and role-model the behaviors for others, rather than being a "development" paragraph.

---

### INPUT FORMAT
You will receive the candidate's data in a simple key-value format as follows:
```
Candidate Name: [Name]
Competencies:
- Leads Inspirationally: [Score]
- Manages and Solves Problems: [Score]
- Plans and Thinks Strategically: [Score]
- Manages Change: [Score]
```

---

### OUTPUT FORMAT
Your output must be a single block of text containing only the two-paragraph executive summary. Do not include any headers, titles, or other text.

---

### GOLD STANDARD EXAMPLES (Few-Shot Learning)

**Example 1: Moderate Scores Profile**
*Input:*
```
Candidate Name: Jane Doe
Competencies:
- Leads Inspirationally: 4
- Manages and Solves Problems: 3
- Plans and Thinks Strategically: 2
- Manages Change: 2
```
*Output:*
As part of the assessment center, you displayed strengths in Leads Inspirationally. In regard to Leads Inspirationally, you demonstrated ability to create an inclusive environment for all team members and address conflicts in a fair manner, to maintain a cooperative work environment. Similarly, you utilize different interpersonal communication strategies to achieve end goals collaboratively and rely on non-verbal cues to drive forward your perspective in a respectful manner. In conjunction, you continue to build networks with relevant internal and external stakeholders to support delivery of organizational goals. You also showed a solid foundation in Manages and Solves Problems; you are able to bring together conflicting information or priorities to identify mutual solutions and ways of working.

Developmentally, scope exists for you to further develop in Plans and Thinks Strategically and Manages Change. In regard to Plans and Thinks Strategically, consider how you can incorporate proactive planning, risk management and strategic objectives when leading an initiative. Identify how you can align the team’s actions and goals with the organization’s strategy, to delivery against organizational vision. Similarly, developing an understanding of the industry dynamics and key trends is critical. In regard to Manages Change, when faced with changing circumstances, consider the need and benefits of this change and communicate this with your team members. Develop openness to change and identify how you can contribute towards the implementation of change programs.

**Example 2: High Scores Profile (Special Case)**
*Input:*
```
Candidate Name: John Smith
Competencies:
- Leads Inspirationally: 4
- Manages and Solves Problems: 4
- Plans and Thinks Strategically: 5
- Manages Change: 4
```
*Output:*
As part of the assessment center, you displayed strengths across all competencies, demonstrating a high level of effectiveness and professionalism. In Leads Inspirationally, you consistently create an inclusive environment by fostering open communication and demonstrating genuine respect for others. In Manages and Solves Problems, you exhibit strong analytical thinking, systematically breaking down complex issues and making well-informed decisions. Regarding Plans and Thinks Strategically, you demonstrate a clear understanding of organizational priorities, linking your team’s objectives to broader strategic goals. Lastly, in Manages Change, you show resilience and adaptability by embracing new initiatives and supporting others through transitions.

Your consistent demonstration of these competencies positions you as a highly effective leader. To build on this, your focus can be on acting as a role model for others. Continuing to leverage these strengths by actively sharing your strategies for leading change, solving complex problems, and thinking strategically will further enhance your influence and impact within your organization.

**Example 3: Low Scores Profile (Special Case)**
*Input:*
```
Candidate Name: Alex Ray
Competencies:
- Leads Inspirationally: 2
- Manages and Solves Problems: 1
- Plans and Thinks Strategically: 1
- Manages Change: 2
```
*Output:*
Developmentally, scope exists for you to further develop in Leads Inspirationally, Manages and Solves Problems, Plans and Thinks Strategically, and Manages Change. In regard to Manages and Solves Problems, consider ways to proactively identify hurdles and view challenges as opportunities for growth. Embracing and effectively Managing Change by developing skills in facilitating transitions will support your development. Regarding Leads Inspirationally, reflect on ways to foster an inclusive environment and respond with emotionally intelligent behaviors to facilitate effective communication. Building networks and leading by example will enhance your ability to inspire others. Lastly, in relation to Plans and Thinks Strategically, consider how to incorporate proactive planning and risk management into your initiatives. Developing a clear understanding of industry trends and future opportunities will enable you to deliver work more effectively.

**Example 4: Tie Scores Profile (Special Case)**
*Input:*
```
Candidate Name: Sam Jones
Competencies:
- Leads Inspirationally: 3
- Manages and Solves Problems: 3
- Plans and Thinks Strategically: 3
- Manages Change: 3
```
*Output:*
As part of the assessment center, you displayed potential strengths across all key competencies, reflecting a solid foundation for further development. In Leads Inspirationally, you show the capacity to foster an inclusive environment. Your ability to Manage and Solves Problems indicates a natural aptitude for analyzing issues. In Plans and Thinks Strategically, you exhibit an emerging awareness of organizational priorities. Additionally, your capacity to Manage Change suggests an openness to organizational transitions.

Developmentally, you can deepen these competencies by engaging in targeted opportunities. This includes strengthening your influence within Leads Inspirationally, adopting more proactive methods in Manages and Solves Problems, expanding your external awareness in Plans and Thinks Strategically, and cultivating a more confident approach to Manage Change. Focusing on these areas will position you as a more effective leader in navigating organizational transitions and fostering a culture of continuous improvement.
"""


# --- Functions ---

def get_gemini_model():
    """Initializes and returns the Gemini Pro model."""
    try:
        # This is more robust for deployment
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("GOOGLE_API_KEY secret is not set. Please add it to your Streamlit Cloud secrets.", icon="🔑")
            return None
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-pro')
    except Exception as e:
        st.error(f"Error initializing the AI model: {e}", icon="🚨")
        return None

def generate_summary(model, candidate_data, competency_cols):
    """Generates a summary for a single candidate using the master prompt."""
    if not model:
        return "Error: AI model not initialized."

    # Format the candidate's scores into a string for the prompt
    competency_str = "\n".join([f"- {col}: {candidate_data[col]}" for col in competency_cols])
    
    # This is the final prompt sent to the API
    final_prompt = f"{MASTER_PROMPT}\n\nHere is the data for the candidate you need to analyze:\n\n```\nCandidate Name: {candidate_data['Candidate Name']}\nCompetencies:\n{competency_str}\n```"
    
    try:
        generation_config = genai.types.GenerationConfig(candidate_count=1, temperature=0.7)
        safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
        }
        response = model.generate_content(final_prompt, generation_config=generation_config, safety_settings=safety_settings)
        return response.text.strip()
    except Exception as e:
        return f"An error occurred while generating the summary: {e}"

# --- Initialize Session State ---
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'name_col' not in st.session_state:
    st.session_state.name_col = ''
if 'competency_cols' not in st.session_state:
    st.session_state.competency_cols = []


# --- Streamlit App UI ---
st.title("✍️ AI Executive Summary Generator")
st.markdown("This tool uses AI to create professional executive summaries based on candidate competency scores. Please upload an Excel file to begin.")

# Sidebar for instructions
with st.sidebar:
    st.header("📋 Instructions")
    st.markdown("""
    1.  **Prepare & Upload File:** Create an Excel file (.xlsx) with columns for `Candidate Name` and the four competencies. Upload it below.
    2.  **Verify Columns:** Use the form to confirm the correct columns for names and scores, then click 'Confirm Selections'.
    3.  **Generate:** Click the 'Generate Executive Summaries' button that appears.
    4.  **Review & Download:** The results will appear in a table, ready for download as a CSV file.
    """)
    st.info("Your Google API Key is securely managed via Streamlit secrets.", icon="ℹ️")

# File Uploader
uploaded_file = st.file_uploader(
    "Choose an Excel file (.xlsx)",
    type="xlsx",
    help="The Excel file should contain one row per candidate."
)

if uploaded_file is not None:
    try:
        # Load data and store in session state to persist it
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("✅ File uploaded successfully!")
        
        st.subheader("1. Verify Data Columns")
        st.markdown("Please confirm that the columns from your file are correctly identified below.")
        
        all_cols = st.session_state.df.columns.tolist()
        
        # Automatically find the name and competency columns based on expected names
        default_name_col = 'Candidate Name' if 'Candidate Name' in all_cols else all_cols[0]
        expected_competencies = [
            'Leads Inspirationally', 'Manages and Solves Problems', 
            'Plans and Thinks Strategically', 'Manages Change'
        ]
        default_competency_cols = [c for c in expected_competencies if c in all_cols]

        with st.form(key='columns_form'):
            name_col = st.selectbox("Candidate Name Column:", all_cols, index=all_cols.index(default_name_col))
            competency_cols = st.multiselect(
                "Competency Score Columns:",
                all_cols,
                default=default_competency_cols
            )
            submitted = st.form_submit_button("Confirm Selections & Preview Data")

        if submitted:
            if not competency_cols:
                st.warning("⚠️ Please select at least one competency column.")
            else:
                # Store selections in session state
                st.session_state.name_col = name_col
                st.session_state.competency_cols = competency_cols
                st.session_state.data_loaded = True

    except Exception as e:
        st.error(f"An error occurred while loading the file: {e}", icon="�")
        st.error("Please ensure your Excel file is formatted correctly and not corrupted.")
        st.session_state.data_loaded = False # Reset state on error

# This block now runs independently of the form submission, checking session state instead
if st.session_state.data_loaded:
    st.subheader("2. Review Uploaded Data")
    st.dataframe(st.session_state.df[[st.session_state.name_col] + st.session_state.competency_cols])

    st.subheader("3. Generate Summaries")
    if st.button("✨ Generate Executive Summaries", type="primary"):
        model = get_gemini_model()
        if model:
            # Use data from session state
            df_to_process = st.session_state.df.copy()
            name_col = st.session_state.name_col
            competency_cols = st.session_state.competency_cols
            
            summaries = []
            progress_bar = st.progress(0, text="Initializing generation...")
            
            total_rows = len(df_to_process)
            for i, row in df_to_process.iterrows():
                # For the function, ensure the name column is what it expects
                temp_row = row.rename({name_col: 'Candidate Name'})
                
                status_text = f"Processing candidate: {row[name_col]} ({i+1}/{total_rows})"
                progress_bar.progress((i + 1) / total_rows, text=status_text)
                
                summary = generate_summary(model, temp_row, competency_cols)
                summaries.append(summary)
                
                time.sleep(1) # A small delay to be kind to the API
            
            progress_bar.progress(1.0, text="✅ Generation complete!")
            df_to_process['Executive Summary'] = summaries
            
            st.subheader("4. Results")
            st.dataframe(df_to_process)
            
            # Convert DataFrame to CSV in-memory for download
            output = io.BytesIO()
            df_to_process.to_csv(output, index=False, encoding='utf-8-sig')
            csv_data = output.getvalue()

            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name='Executive_Summaries_Output.csv',
                mime='text/csv',
            )

if uploaded_file is None:
    st.info("Awaiting Excel file upload...")
    # Clear state if file is removed
    for key in ['data_loaded', 'df', 'name_col', 'competency_cols']:
        if key in st.session_state:
            del st.session_state[key]
