import Evaluator.GPT_Score.GPT4_score as GPT4_score
import torch
import json
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import Util.util as util
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor, LlavaNextForConditionalGeneration
from transformers import BitsAndBytesConfig



if __name__ == "__main__":

    json_path = "LingoQA_benchmark_result_score.json"

    # Load JSON data
    try:
        with open(json_path, 'r') as file:
            json_data = json.load(file)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        exit()

    Total_score = [ ]
    Answer_score = [ ]
    Reason_score = [ ]
    Grammar_score = [ ]

    # Iterate through scenes
    for scene_id, scene_content in tqdm(json_data.items(), desc="Processing Scenes", total=len(json_data)):

        QA_list = scene_content['questions']
        for QA in QA_list:
            Score_sheet = QA['Reason']
            try:
                #total_score = int(util.extract_between_markers(Score_sheet, '[TS_START]', '[TS_END]')[0])
                answer_score = int(util.extract_between_markers(Score_sheet, '[AC_START]', '[AC_END]')[0])
                reason_score = int(util.extract_between_markers(Score_sheet, '[RV_START]', '[RV_END]')[0])
                grammar_score = int(util.extract_between_markers(Score_sheet, '[CR_START]', '[CR_END]')[0])
                total_score = answer_score + reason_score + grammar_score

                # Assertions to verify score integrity
                assert total_score == answer_score + reason_score + grammar_score, "Total score is not equal to sum of individual scores"
                assert 0 <= answer_score <= 40, "Answer score is missing"
                assert 0 <= reason_score <= 40, "Reason score is missing"
                assert 0<= grammar_score <= 20, "Grammar score is missing"

                # Append scores to their respective lists
                Total_score.append(total_score)
                Answer_score.append(answer_score)
                Reason_score.append(reason_score)
                Grammar_score.append(grammar_score)
            except AssertionError as ae:
                print(f"Skipping an entry due to failed assertion: {ae}")
            except Exception as e:
                print(f"Skipping an entry due to an unexpected error: {e}")

    # Convert lists to DataFrame
    df = pd.DataFrame({
        'Total Score': Total_score,
        'Answer Score': Answer_score,
        'Reason Score': Reason_score,
        'Grammar Score': Grammar_score
    })

    # Analyze and print descriptive statistics
    print("\nDescriptive Statistics:")
    print(df.describe())

    # Plotting and saving plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 7))

    axs[0, 0].hist(df['Total Score'], bins=20, color='blue', alpha=0.7)
    axs[0, 0].set_title('Distribution of Total Scores')
    axs[0, 0].set_xlabel('Total Score')
    axs[0, 0].set_ylabel('Frequency')

    axs[0, 1].hist(df['Answer Score'], bins=20, color='green', alpha=0.7)
    axs[0, 1].set_title('Distribution of Answer Scores')
    axs[0, 1].set_xlabel('Answer Score')

    axs[1, 0].hist(df['Reason Score'], bins=20, color='red', alpha=0.7)
    axs[1, 0].set_title('Distribution of Reason Scores')
    axs[1, 0].set_xlabel('Reason Score')

    axs[1, 1].hist(df['Grammar Score'], bins=20, color='purple', alpha=0.7)
    axs[1, 1].set_title('Distribution of Grammar Scores')
    axs[1, 1].set_xlabel('Grammar Score')

    plt.tight_layout()
    plt.savefig('score_distributions.png')  # Save the figure to a file
    #plt.show()



