def train_data_preprocess(examples):
    
    """
    generate start and end indexes of answer in context
    """
    
    def find_context_start_end_index(sequence_ids):
        """
        returns the token index in whih context starts and ends
        """
        token_idx = 0
        while sequence_ids[token_idx] != 1:  #means its special tokens or tokens of queston
            token_idx += 1                   # loop only break when context starts in tokens
        context_start_idx = token_idx
    
        while sequence_ids[token_idx] == 1:
            token_idx += 1
        context_end_idx = token_idx - 1
        return context_start_idx,context_end_idx  
    
    
    questions = [q.strip() for q in examples["question"]] # list of Questions
    context = examples["context"] # list of contexts
    answers = examples["answers"] # list of answers
    
    inputs = tokenizer(
        questions,
        context,
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,  #returns id of base context
        return_offsets_mapping=True,  # returns (start_index,end_index) of each token
        padding="max_length"
    )


    start_positions = []
    end_positions = []

    
    for i,mapping_idx_pairs in enumerate(inputs['offset_mapping']):
        context_idx = inputs['overflow_to_sample_mapping'][i]
    
        # from main context
        answer = answers[context_idx]
        answer_start_char_idx = answer['answer_start'][0]
        answer_end_char_idx = answer_start_char_idx + len(answer['text'][0])

    
        # now we have to find it in sub contexts
        tokens = inputs['input_ids'][i]
        sequence_ids = inputs.sequence_ids(i)
   
        # finding the context start and end indexes wrt sub context tokens
        context_start_idx,context_end_idx = find_context_start_end_index(sequence_ids)
    
        #if the answer is not fully inside context label it as (0,0)
        # starting and end index of charecter of full context text
        context_start_char_index = mapping_idx_pairs[context_start_idx][0]
        context_end_char_index = mapping_idx_pairs[context_end_idx][1]
    

        #If the answer is not fully inside the context, label is (0, 0)
        if (context_start_char_index > answer_start_char_idx) or (
            context_end_char_index < answer_end_char_idx):
            start_positions.append(0)
            end_positions.append(0)
    
        else:

            # else its start and end token positions
            # here idx indicates index of token
            idx = context_start_idx
            while idx <= context_end_idx and mapping_idx_pairs[idx][0] <= answer_start_char_idx:
                idx += 1
            start_positions.append(idx - 1)  
        

            idx = context_end_idx
            while idx >= context_start_idx and mapping_idx_pairs[idx][1] > answer_end_char_idx:
                idx -= 1
            end_positions.append(idx + 1)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
    
train_sample = dataset["train"].select([i for i in range(200)])
    
train_dataset = train_sample.map(
    train_data_preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

len(dataset["train"]),len(train_dataset)