import pandas as pd
from angle_emb import AnglE, Prompts
from FlagEmbedding import BGEM3FlagModel
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import gc
from transformers import AutoTokenizer, AutoModel
import torch
from mlx_lm import load, generate
import warnings
import os


warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes.*")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#############################
###Software Prints###
print("*"*25)
print("CompSurveyMatcher v0.1 (MVP)")
print("Authors: Robert Kelly, Jack Keeton")
print("Contributors: Robin Parente, ChatGPT")
print("Models Used: Mistral7b Instruct (fine tuned & base), BGE-M3")
print("Special Thanks to the open source community for making incredible models available!")
print("*"*25)
print("\n")
time.sleep(2)

#############################
#####Import Data Section#####
#############################
print("-"*25)
print("Import Data Section")
print("-"*25)
time.sleep(1)
print("Importing raw job descriptions...")
time.sleep(1)
df1 = pd.read_excel('output.xlsx')  # raw job descriptions
df2 = pd.read_excel('MasterSurvey.xlsx')  # Survey job descriptions
df3_TitleWords = pd.read_excel('MasterLvlWords.xlsx') #list of words found in titles that denote level
print("Import complete. Cleaning data fields...")
time.sleep(1)
#clean field names from beg of description text for df1
fields_to_clean = ['responsibilities', 'experience', 'skills']
for field in fields_to_clean:
    df1[field] = df1[field].str.replace(f'^{field}/n', '', regex=True)
df1['FinalMatch_ID'] = ''
df1['FinalMatch_Survey'] = ''
df1['FinalMatch_Title'] = ''
df1['FinalMatch_Level'] = ''
df1['FinalMatch_Family'] = ''
print("Datafields cleaned")
time.sleep(1)
df1_rows_total = len(df1)
print(f"You have {df1_rows_total} rows of JDs that the program will help you match to a survey.")
print("-"*25)
print("\n")
time.sleep(1)

#############################
#####Fuzzy Match Section#####
#############################
print("-"*25)
print("Fuzzy Matching Section")
print("-"*25)
print(f"We will first attempt to find very obvious matches with title fuzzy matching.")
print("Beginning fuzzy matching...")
time.sleep(1)
  ###Fuzzy Functions###
def _create_ngrams(string: str):
   result = []
   for n in range(2, 3):
      ngrams = zip(*[string[i:] for i in range(n)])
      ngrams = [''.join(ngram) for ngram in ngrams if ' ' not in ngram]
   
   result.extend(ngrams)
   return(result)

def find_fuzzy_match(list_a, list_b, n_matches = 1):
    # Define the vectorizer
    # list_b =b
    # list_a = [a[0]]
    # n_matches = 3
    vectorizer = TfidfVectorizer(min_df=1, analyzer=_create_ngrams).fit(list_a + list_b)
    vectorizer.vocabulary_
    
    tf_idf_listings = vectorizer.transform(list_a)
    tf_idf_survey = vectorizer.transform(list_b)
    
    matrix_listings=pd.DataFrame(tf_idf_listings.todense(),\
    columns=sorted(vectorizer.vocabulary_))
    matrix_survey=pd.DataFrame(tf_idf_survey.todense(),\
    columns=sorted(vectorizer.vocabulary_))
    
    chk = cosine_similarity(matrix_listings, matrix_survey)
    
    top_ids = list(range(-1, -n_matches-1, -1))
    # idx = np.argmax(chk)
    idx = np.argsort(chk)[:, top_ids].tolist()[0]
    
    probs = chk[:, idx].tolist()[0]
    titles = ":".join([list_b[i] for i in idx])
    
    return titles, probs, idx
   ###End Fuzzy Functions###

listing_data = df1["title"].copy()
survey_data = df2["title"].copy()

def clean_text(x):
    #Convert strings to lowercase
    x = x.str.lower()
    
    #Remove the "-" symbol
    x = x.str.replace(r"(\-)"," ",regex=True)
    x = x.str.replace(r"(\&amp;)","and",regex=True)
    x = x.str.replace(r",","",regex=True)
    x = x.str.replace(r"\(","",regex=True)
    x = x.str.replace(r"\)","",regex=True)

    # Replace consecutive blank spaces for 1 blank space
    x = x.str.replace(r"(  |   )", " ",regex=True)
    x = x.str.replace(r"(  )", " ",regex=True)
    
    # Remove starting blank space
    x = x.str.replace(r"(^ )", "",regex=True)
    
    # Remove blank space at the end
    x = x.str.replace(r"( $)", "",regex=True)
    
    return x

listing_data = clean_text(listing_data)
survey_data = clean_text(survey_data)

a = listing_data.to_list()
b = survey_data.to_list()

title_fuzzy_matches = [find_fuzzy_match([x], b) for x in a]

survey_title_match = [x[0] for x in title_fuzzy_matches]
prob_match = [x[1] for x in title_fuzzy_matches]
idx_match = [x[2] for x in title_fuzzy_matches]

df1["FuzzyIDs"] = idx_match
df1["FuzzyProbs"] = prob_match
df1["FuzzyTitleMatch"] = survey_title_match

# Loop through each row in df1
for index, row in df1.iterrows():
    # Check if any value in FuzzyProbs is >= 1
    if any(prob >= 1 for prob in row["FuzzyProbs"]):
        # Find the index of the first match with a probability >= 1
        match_indices = [i for i, prob in enumerate(row["FuzzyProbs"]) if prob >= 1]
        if match_indices:  # Ensure there is at least one match
            first_match_index = match_indices[0]
            survey_match = df2.iloc[row["FuzzyIDs"][first_match_index]]  # Access the matching row in df2

            # Update df1 with the matched values from df2
            df1.at[index, "FinalMatch_ID"] = survey_match['2022 Position ID']
            df1.at[index, "FinalMatch_Survey"] = survey_match["Survey"]  # Adjust column name if necessary
            df1.at[index, "FinalMatch_Title"] = survey_match["title"]  # Adjust based on the actual column name in df2

final_match_count = df1[df1['FinalMatch_ID'] != ''].shape[0]
remaining_to_match = df1_rows_total - final_match_count
print(f"Fuzzy Title Matching done!  We were able to match {final_match_count} jobs.")
print(f"This means we still have {remaining_to_match} to match.  The program will now guide you through these harder matches.")
print("-"*25)
print("\n")
time.sleep(1)
gc.collect()

#############################
#####Family Match Section####
#############################
print("-"*25)
print("Job Family Matching Section")
print("-"*25)
print("first we are going to use responsibilities and skills found in the job description to have an LLM predict the job family")
print("this prediction will not necessarily match the available job families in the survey(s), so we'll next use another model to suggest a match")
print("sit back and allow the LLM to do its thing - this can take a while but we will let you know when it is finished")
time.sleep(1)

   ###START FAMILY PREDICTOR LLM###
df1['responsibilities_skills'] = 'Responsibilities:\n' + df1['responsibilities'] + '\nSkills:\n' + df1['skills']
df1['responsibilities_experience'] = 'Responsibilities:\n' + df1['responsibilities'] + '\nExperience:\n' + df1['experience']
model, tokenizer = load("/mlx_model")

sampleskills1 = """Skills:
- Track record of success leading compensation programs
- Experience in High-Tech and/or Consulting industries
- Expertise in crafting, developing, and maintaining compensation programs
- Knowledge of competitive market pay practices
- Team orientation with collaborative style
- Strong influencing skills and clear communication
- Independent, self-starter with project and task management skills
- Excellent analytical skills and business sense
- Proficient in Excel/spreadsheet and Workday
- Ability to balance priorities and manage ambiguity
- Resilient and determined with a focus on customer service
"""
sampleresponsibilities1 = """Responsibilities:
- Manage, design, and coordinate incentive plans such as Apple's executive bonus plan, Retail leadership bonus plan, and other incentive plans.
- Maintain and develop processes for global pay/salary structures based on internal and external data.
- Perform analytics on internal and external sources and make recommendations regarding incentive programs and salary structures.
- Network with compensation professionals for knowledge of pay practices and shifting trends.
- Lead strategic projects and manage routine activities.
- Provide ad hoc support and develop cross-functional partnerships for business needs.
- Initiate and maintain relationships with the entire people team.
"""
sampleanswer1 = "Compensation Administration"

sampleskills2 = """Skills:
- Concept development
- Storyboarding
- Animation design
- Adobe After Effects and Creative Cloud Suite
- Cinema 4D, Maya & 3D pipelines
- Compositing tools such as Nuke and Flame
- High-end post-production workflows
- Collaboration and teamwork
- Organizational skills
- Time management
- Communication skills
"""
sampleresponsibilities2="""Responsibilities:
- Conceptualize, build storyboards, design and animation
- Develop creative work that implements business strategy
- Build innovative and clear motion graphics animations
- Innovate and drive best-practices for post-production graphics pipeline/workflow
"""
sampleanswer2 = "Motion Graphics Design"

fields_to_prompt = ['responsibilities_skills']
# Add new columns to the DataFrame
df1['family_predict'] = ''
df1['family_predict_Desc'] = ''
counter = 0
filtered_df1 = df1[df1['FinalMatch_ID'] == '']
lendf1 = len(filtered_df1)
# Iterate through each row in the dataframe
for index, row in filtered_df1.iterrows():
    for field in fields_to_prompt:
        sampleinput1 = sampleresponsibilities1 + sampleskills1
        sampleinput2 = sampleresponsibilities2 + sampleskills2
        promptstr = row[field]  # Extract the value from the current row for the given field
        title = row['title']
        fullprompt = f"""<s>[INST]I need help identifying a job family given a set of job responsibilities and skills. I will give you two examples of job responsibilities and skills, and their correct job family pairs. I will then give a third set of job responsibilities and skills, for which I need you to tell me the job family.  ONLY PROVIDE THE FAMILY NAME - DO NOT GIVE AN EXPLANATION!\n**Example 1**\nWhat specific job family might have or require these job responsibilities and skills:\n{sampleinput1}\nJob Family: {sampleanswer1}\n**EXAMPLE 2**\nWhat specific job family might have or require these job responsibilities and skills:\n{sampleinput2}\nJob Family: {sampleanswer2}\n**YOUR TASK - REMEMBER TO ONLY PROVIDE THE JOB FAMILY NAME**\nWhat specific job family might have or require these job responsibilities and skills:\n\n{promptstr}\n\nJob Family: [/INST]"""
        max_retries = 6
        retries = 0
        # Initialize a variable for the response and a flag for the retry logic
        response_valid = False
        response = ""
        response_Desc = ""
        while not response_valid:
            if retries >= 1:
                del response, response_text
                gc.collect()
            time.sleep(1)
            response = generate(model, tokenizer, prompt=fullprompt, temp=0.25, verbose=False, max_tokens=15)
            time.sleep(1)
            retries += 1
            response_text = response  # Assuming response is a list and we take the first result
            if len(response_text.split()) <= 8:
                response_valid = True
                response_text = response_text.replace('Job Family: ', '')
                fullprompt_Desc = f"<s>[INST]Write a sentence that describes the {response_text} job family.[/INST]"
                response_Desc = generate(model, tokenizer, prompt=fullprompt_Desc, temp=0.1, verbose=False, max_tokens=50)
                response_text_Desc = response_Desc
                time.sleep(1)
                # Append the response_text to the corresponding column based on the field
                df1.at[index, 'family_predict'] = response_text
                df1.at[index, 'family_predict_Desc'] = response_text_Desc
            if retries >= max_retries:
                # Optionally handle the case where max attempts are reached without valid response
                response_text = "!@#$%^&*"  # or any default value you prefer
                response_text_Desc = "!@#$%^&*"
                df1.at[index, 'family_predict'] = response_text
                df1.at[index, 'family_predict_Desc'] = response_text_Desc
        #print(f"{response_text}: {response_text_Desc}")
        time.sleep(1)
        del response, response_Desc, response_text, response_text_Desc, fullprompt, fullprompt_Desc
        gc.collect()
        counter = counter + 1
        print(f"Completed family prediction for {counter} of {lendf1} rows")

del model, tokenizer
gc.collect()

print(f"LLM family predictions done!")
print("-"*25)
time.sleep(1)

   ###END FAMILY PREDICTOR LLM###

###START FAMILY MATCHER###

# Combine family name and description in dfs
df1['fam_name_desc_Resp'] = df1['family_predict'] + ': ' + df1['family_predict_Desc']
df2['fam_name_desc'] = df2['Job Family Name'] + ': ' + df2['Family Description']

print("Now that we have family predictions for each JD, we can use an embedding model to find the matching survey job family")
print("We will need your help in validating these family matches - the AI will propose a best match, but we need you to validate its accuracy")
print("-"*25)
input("Let's practice your user inputs. Please press ENTER to continue")

# Load model
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
print('Data and model loaded')

# Survey job fam name list
survey_family_list = pd.unique(df2['fam_name_desc']).tolist()
# First, filter df1 to only include rows where 'FinalMatch_ID' is NaN
filtered_df1 = df1[df1['FinalMatch_ID'] == '']

for i, df1_row in filtered_df1.iterrows():
    # Initialize the starting index for the rolling window of top family names
    start_idx = 0

    # Extract job details
    df1_job = df1_row['title']
    sent1_str = df1_row['fam_name_desc_Resp']
    jd_family_list = [sent1_str]
    df1_respskills = df1_row['responsibilities_skills']
    
    # Generate and score sentence pairs
    sentence_pairs = [[i, j] for i in jd_family_list for j in survey_family_list]
    score_dict = model.compute_score(sentence_pairs, 
                                     max_passage_length=128, 
                                     weights_for_different_modes=[0.4, 0.3, 0.3])

    # Sort scores
    sorted_scores_indices = sorted(range(len(score_dict['colbert+sparse+dense'])), 
                                   key=lambda k: score_dict['colbert+sparse+dense'][k], 
                                   reverse=True)

    # Loop for user to accept/reject families
    while start_idx < len(sorted_scores_indices) - 4:  # Ensure there are always 5 families to show
        current_max_family_index = sorted_scores_indices[start_idx]
        current_max_family_name = survey_family_list[current_max_family_index]

        # Filter df2 where 'Job Family Name' matches current_max_family_name to get the matched rows
        matched_rows = df2[df2['fam_name_desc'] == current_max_family_name]

        # Calculate the ending index for the rolling window, making sure it doesn't go beyond the list
        end_idx = min(start_idx + 5, len(sorted_scores_indices))
        # Update the top family names based on the rolling window
        next_top_family_names = [survey_family_list[index] for index in sorted_scores_indices[start_idx:end_idx]]  # Adjusted to start from the next

        print(f"-"*10, "CURRENT JD BEING MATCHED", "-"*10)
        print(f"Job Title: {df1_job}")
        print(f"LLM Predicted_Family\n   {sent1_str}\n")
        print(f"--Job Description Details--\n{df1_respskills}\n")
        # Print the updated rolling top 5 survey families
        print(f"-"*10, "AI Next 5 Family Match Suggestions", "-"*10)
        for idx, family_name in enumerate(next_top_family_names, start=1):
            print(f"{idx + start_idx}. {family_name}")
        
        print(f"\n","-"*5,"AI PROPOSED FAMILY","-"*5)
        print(f"{current_max_family_name}\n")
        # Print out the 'title' of matched rows
        print("Survey jobs in this family:")
        print(matched_rows['title'].to_string(index=False))

        user_input = input(f"\n(**START USER TASK**)\nIs this the correct family match? (y/n) Or type 'skip' to move to the next job.\n(**END USER TASK**)\nINPUT:")

        if user_input.lower() == 'y':
            df1.at[i, 'FinalMatch_Family'] = current_max_family_name
            print(f"\n--User Confirmed Survey Family--\n{current_max_family_name}\n")
            break
        elif user_input.lower() == 'n':
            # Move the window by one for the next iteration
            start_idx += 1
        elif user_input.lower() == 'skip':
            print("Skipping... No information recorded for this family match.")
            break

    time.sleep(1)

print("\n\nFinished with family matching...moving to level")
input("Press ENTER to proceed to level matching")
del model
gc.collect()


#############################
##### Level Match Section #####
#############################
print("\n")
print("-" * 25)
print("Level Matching Section")
print("-" * 25)
time.sleep(1)
# Initialize the tokenizer and model
print("Let's first use an embedding model to isolate leveling information from your jobs...")
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')  # Update path as necessary
model = AutoModel.from_pretrained('BAAI/bge-m3')  # Update path as necessary
model.eval()

### EMBEDDINGS FUNCTIONS ###
# Function to compute embeddings for a given text
def compute_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # Get the embeddings of the [CLS] token

# Function to calculate cosine similarity
def cosine_similarity(a, b):
    return (a @ b.T) / (a.norm(dim=1)[:, None] * b.norm(dim=1))

### END EMBEDDINGS FUNCTIONS ###

### LEVEL FUNCTIONS ###
# Function to clean and split the title into words given the various possible separators
def split_title(title):
    # Replace "&amp;" with "&" and then split using the specified delimiters
    cleaned_title = re.sub(r"&amp;", "&", title)
    words = re.split(r"[ ,/&*\\-]", cleaned_title)  # Split on space, comma, "/", "&", "-", "&amp;"
    return [word for word in words if word]  # Filter out any empty strings

# This populates titleLevel and titleFunction words (i.e., deconstructs the title)
def update_title_level(df, df3_embeddings):
    for index, row in df.iterrows():
        words = split_title(row['title'])
        title_level_words = []
        title_function_words = []
        # Compute embeddings for each word in the title
        for word in words:
            word_embedding = compute_embeddings(word)
            match_found = False

            # Compare against each word's embedding in df3_embeddings
            for df3_word, df3_word_embedding in df3_embeddings.items():
                # Compute cosine similarity
                similarity = cosine_similarity(word_embedding, df3_word_embedding['Unique Words'])
                if similarity > 0.95:
                    title_level_words.append(word)
                    match_found = True
                    break  # Assuming one word doesn't need to match multiple times

            if not match_found:
                title_function_words.append(word)
        # Update 'titleLevel' and 'titleFunction' fields in df
        if not title_level_words:
            title_level_words = words
        df.at[index, 'titleLevel'] = ' '.join(title_level_words)
        df.at[index, 'titleFunction'] = ' '.join(title_function_words)
    return df

### END LEVEL FUNCTIONS ###

### START TITLELEVEL CALC ###
# Precompute df3 embeddings (title word embeddings)
df3_fields = ['Unique Words']
df3_embeddings = {index: {field: compute_embeddings(row[field]) for field in df3_fields} for index, row in df3_TitleWords.iterrows()}

# Populate titleLevel for df1
df1 = update_title_level(df1, df3_embeddings)
print("Leveling information processed and isolated")
print("Now we will need your help in validating level matches - the AI will propose list of level options sorted by most likely, and we will need you to pick the best one")
print("-" * 25)
print("\n")
time.sleep(1)

### END TITLELEVEL CALC ###

### START LEVEL PICKER ###
print("-" * 25)
print("Starting Level Picker")
print("-" * 25)
# Load model
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# Filter df1 for unmatched entries
filtered_df1 = df1[df1['FinalMatch_ID'] == '']

for i, df1_row in filtered_df1.iterrows():
    if df1_row['FinalMatch_Family'] == '':
        print(f"Skipping level matching for '{df1_row['title']}' as no family match was recorded.")
        continue  # Skip this iteration if no family match
    df1_job = df1_row['title']
    df1_titleLevel = df1_row['titleLevel']
    df1_respexp = df1_row['responsibilities_experience']

    ###USER INPUT SECTION###
    print("\n")
    print(f"-"*10,"CURRENT JD BEING MATCHED","-"*10)
    print(f"Job Title: {df1_job}")
    print(f"Level words in title: {df1_titleLevel}\n")
    print(f"--Job Description Details--\n{df1_respexp}\n")
    print(f"-"*5,"AI PROPOSED LEVELS","-"*5)
    print("**START USER TASK**")

    # Filter df2 to match 'FinalMatch_Family'
    family_filtered_df2 = df2[df2['Job Family Name'] + ': ' + df2['Family Description'] == df1_row['FinalMatch_Family']]
    survey_titleLevel_list = pd.unique(family_filtered_df2['titleLevel']).tolist()

    # Prepare for comparison
    sentence_pairs = [[df1_titleLevel, survey_titleLevel] for survey_titleLevel in survey_titleLevel_list]
    score_dict = model.compute_score(sentence_pairs, 
                                     max_passage_length=128,
                                     weights_for_different_modes=[0.4, 0.3, 0.3])

    # Generate a DataFrame to hold potential matches and their scores
    matches_df = pd.DataFrame({
        'titleLevel': [pair[1] for pair in sentence_pairs],
        'score': score_dict['colbert+sparse+dense']
    }).merge(family_filtered_df2[['titleLevel', 'Level Name', 'title', '2022 Position ID', 'Survey']], on='titleLevel').drop_duplicates().sort_values(by='score', ascending=False)

    # Display the ordered table to the user, indexing starting from 1 for user-friendliness
    print("Select the best level match based on the following table (enter the row number, starting with 0):")
    display_table = matches_df[['title', 'titleLevel', 'Level Name']].reset_index(drop=True)
    print(display_table.to_string(index=True))

    # User Selection
    print("**END USER TASK**")
    selected_index = int(input("Your selection (row number): "))  # Adjust for zero-based indexing
    selected_match = matches_df.iloc[selected_index]

    # Safely access values from the selected row
    selected_titleLevel = selected_match['titleLevel']
    selected_level_name = selected_match['Level Name']
    selected_title = selected_match['title']

    # Correcting this line to update 'FinalMatch_Level' with 'titleLevel' instead of 'Level Name'
    df1.at[i, 'FinalMatch_Level'] = selected_titleLevel
    df1.at[i, 'FinalMatch_ID'] = selected_match['2022 Position ID']  # Ensure this column exists in matches_df
    df1.at[i, 'FinalMatch_Survey'] = selected_match['Survey']  # Ensure this column exists in matches_df
    df1.at[i, 'FinalMatch_Title'] = selected_title

    print(f"\nMatch confirmed for '{df1_job}': Level Name '{selected_level_name}' with title '{selected_title}'.")

print("Finished with level matching.")
print("-" * 25)
print("\n")

### END LEVEL PICKER ###
df_toprint = df1[['url', 'description_cleaned', 'title', 'FinalMatch_ID', 
                  'FinalMatch_Survey', 'FinalMatch_Title','FinalMatch_Level', 'FinalMatch_Family']]
df_toprint.to_excel('master_matcher_result.xlsx', index=False)
print("All sections completed - you're done! The matched results have been saved to 'master_matcher_result.xlsx' in your current directory.")
