#Comp Survey Matcher

Note: This code uses MLX will only run (end-to-end) on Apple Silicon (M1 or above chips)

Recommend creating a new environment before getting started, with Python 3.11 installed
First install libraries from requirements.txt...run:
pip install -r requirements.txt

#Section 1 - Fine Tuning Open Source LLM to become good at extracting responsibilities, skills, and experience from raw job postings

##Step 1: 
Get base model from HuggingFace and convert to mlx
'''
python convert.py --hf-path mistralai/Mistral-7B-Instruct-v0.1
'''

this outputs 6 model files into a mlx_model folder - these can be very large files (~15GB)!

##Step 2: Prep training data for fine-tuning
A small sample excel file can be found in FineTuning_RawDataSample.xlsx
You will ideally need 500+ rows of examples in this excel file for a good training run

##Step 3: Convert training data file to jsonl
run in terminal:
'''
python make_training_json.py
'''
This should put three files into the 'data' folder (test.jsonl, train.jsonl, valid.jsonl)

##Step 4: Train base model from step 1
To train, run in terminal:
'''
python lora.py --model /mlx_model \
               --train \
               --iters 500
'''
see https://github.com/ml-explore for documentation on fine tuning with MLX.  Different M1/2/3 chips will require things like quantization or lower batching to successfully run fine-tuning.
Depending on parameters, this step can take multiple days to run! Check out Apple's ml-explore repo for advice on optimization.
The above will output an 'adapters.npz' file to your project directory - these are the new (additional) model weights after fine-tuning on the jsonl data.

##Step 5: Prep raw job postings file for fine-tuned LLM conversion
A small sample excel file can be found in realJDs_sample.xlsx
Fill out this template with your own jobs to extract responsibilities, skills, and experience from unstructured JDs in next step

##Step 6: Convert unstructured JDs to structured (extract resp, skills, exp)
'''
python ConvertJDs_local.py --model /mlx_model \
               --adapter-file adapters.npz \
               --max-tokens 500
'''
Here we are loading the adapters file produced in step 4 on top of the base model from step 1, which is our fine tuned model
We then use the fine tuned model to extract resp, skills, and exp from our raw job descriptions that we added to the step 5 template.
This outputs an 'output.xlsx' file containing each raw JD + extracted responsibilities, experience, and skills

##Optional Step: Continue further training the adapters file (ie, fine tune more from last checkpoint)
To continue training, run in terminal:
'''
python lora.py --model /mlx_model \
               --train \
               --iters 200 \
               --resume-adapter-file adapters.npz
'''
This simply allows you to stop training in step 4 earlier, save a checkpoint adapters file, and then continue training later without restarting entirely


#Section 2 - Match JDs to Comp Survey Job Match

##Step 1: Populate Survey file
See MasterSurvey.xlsx for required template of survey fields.  You must populate this yourself, since comp survey job architectures are proprietary information.
Note that titleLevel is the list of words that are found on the MasterLvlWords.xlsx file that are also in the survey job title.
titleFunction would include any remaining words in the survey job title that do not match a word in the MasterLvlWords.xlsx file.
You must populate these yourself, but I recommend a quick python script (can also just use a quick excel function)

##Step 2: Run matching program in console
'''
python master_matcher_v3.py
'''
This will use your output.xlsx file from Section 1 Step 6 and your MasterSurvey.xlsx file from previous step to match each of your JDs to a survey match.
This program requires human input to verify match suggestions, which will run via console printouts.
The end of this program produces a file called 'master_mather_result.xlsx' with your final matches!

Licence: MIT License
All of this work is open source, and came from open source models, frameworks, and public internet data.
Thank you to everyone who contributed to these open source tools - this would not have been possible without you!  Hopefully we've built something cool on top of your shoulders.
