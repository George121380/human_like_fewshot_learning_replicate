def LM_baseline_prompt(x_candidate, ref_list):
    ref_list_str = ', '.join([str(x) for x in ref_list])


    system_prompt = f"Given a list of reference numbers and a single candidate number. You need to predict how likely this candidate belongs to the same category with the reference numbers. The output should be a float number between 0 and 1. The higher the number, the more likely the candidate belongs to the same category with the reference numbers. You can only output a float number between 0 and 1 without any explainations."

    user_prompt = f"""
Reference numbers: {ref_list_str}
Candidate number: {x_candidate}
"""

    return system_prompt, user_prompt

