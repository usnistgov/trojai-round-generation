# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import datasets


def check_context_answer_alignment(example):
    for a_idx in range(len(example['answers']['text'])):
        # check raw dataset for answer consistency between context and answer
        answer_text = example['answers']['text'][a_idx]
        a_st_idx = example['answers']['answer_start'][a_idx]
        a_end_idx = a_st_idx + len(example['answers']['text'][a_idx])
        answer_text_from_context = example['context'][a_st_idx:a_end_idx]
        if answer_text != answer_text_from_context:
            # print(example['id'])
            return False
    return True


dataset = datasets.load_dataset('squad_v2', split='train', keep_in_memory=True)

start_len = len(dataset)
dataset = dataset.filter(check_context_answer_alignment,
                                num_proc=1,
                                keep_in_memory=True)
end_len = len(dataset)
print('{} instances contain mis-alignment between the answer text and answer index.'.format(start_len - end_len))