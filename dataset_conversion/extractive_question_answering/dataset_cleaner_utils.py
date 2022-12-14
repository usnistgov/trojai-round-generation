import regex
import copy


def check_errors(example):
    context = example['context']
    answers = example['answers']
    
    for i, answer_start in enumerate(answers['answer_start']):
        answer_text = answers['text'][i]
        context_answer = context[answer_start: len(answer_text) + answer_start]
        if context_answer != answer_text:
            print(
                'answer: {} is not right in context location: {} with context answer: {}'.format(answer_text,
                                                                                                 answer_start,
                                                                                                 context_answer))
    return example


def cleanup_answers(example):
    answers = example['answers']
    example['answers'] = {'answer_start': answers['answer_start'], 'text': answers['text']}
    return example


def remove_answer_not_found_subjqa(example):
    to_remove = list()
    for a_idx in range(len(example['answers']['text'])):
        # check raw dataset for answer consistency between context and answer
        a_answer_text = example['answers']['text'][a_idx]
        if 'ANSWERNOTFOUND' in a_answer_text:
            to_remove.append(a_idx)
    if len(to_remove) > 0:
        to_remove.sort(reverse=True)
        for i in to_remove:
            del example['answers']['answer_start'][i]
            del example['answers']['text'][i]

    return example


def check_context_answer_alignment(example):
    for a_idx in range(len(example['answers']['text'])):
        # check raw dataset for answer consistency between context and answer
        a_answer_text = example['answers']['text'][a_idx]
        ans_idx = example['answers']['answer_start'][a_idx]
        ans_len = len(example['answers']['text'][a_idx])
        b_answer_text = example['context'][ans_idx:ans_idx + ans_len]
        if a_answer_text != b_answer_text:
            return False
    return True


def is_letter_present_in_answer_text(example):
    # Searches for any valid character from any language, any kind of numeric character, and any currency sign
    validLetter = regex.compile("\p{Letter}|\p{N}|\p{Sc}")
    
    for i in range(len(example['answers']['text'])):
        text = example['answers']['text'][i]
        
        if validLetter.search(text) is None:
            print('removing answer: "{}"'.format(text))
            return False
    
    return True


def remove_example_duplicate_whitespace(example):

    updated_question = ' '.join(copy.deepcopy(example['question']).split())
    updated_context = ' '.join(copy.deepcopy(example['context']).split())
    for a_idx in range(len(example['answers']['text'])):
        
        # check raw dataset for answer consistency between context and answer
        a_answer_text = example['answers']['text'][a_idx]
        ans_idx = example['answers']['answer_start'][a_idx]
        ans_len = len(example['answers']['text'][a_idx])
        b_answer_text = example['context'][ans_idx:ans_idx + ans_len]
        
        if a_answer_text != b_answer_text:
            msg = 'HuggingFace answer text does not match text at the answer_index in the context. Id: {}'.format(
                example['id'])
            print(msg)
            raise RuntimeError(msg)
        
        pre_answer_context = copy.deepcopy(example['context'][0:ans_idx])

        # handle if the pre answer context ends with a whitespace, but it not all whitespace
        if len(pre_answer_context) > 0 and pre_answer_context[-1].isspace() and not pre_answer_context.isspace():
            pre_answer_context = pre_answer_context[0:-1]

        shrunk = ' '.join(pre_answer_context.split())
        delta = len(pre_answer_context) - len(shrunk)

        updated_index = example['answers']['answer_start'][a_idx] - delta
        updated_answer = ' '.join(example['answers']['text'][a_idx].split())

        a_answer_text_post = updated_answer
        b_answer_text_post = updated_context[updated_index: updated_index + len(updated_answer)]
        
        if a_answer_text_post != b_answer_text_post:
            msg = 'HuggingFace answer text does not match text at the answer_index in the context. Id: {}'.format(
                example['id'])
            print(msg)
            raise RuntimeError(msg)

        example['answers']['answer_start'][a_idx] = updated_index
        example['answers']['text'][a_idx] = updated_answer

    example['question'] = updated_question
    example['context'] = updated_context
    return example