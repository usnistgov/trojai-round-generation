# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import pandas as pd
import re

# https://www.microsoft.com/en-us/download/details.aspx?id=52419
fp = '/home/mmajurski/Downloads/WikiQACorpus/WikiQA.tsv'

df = pd.read_csv(fp, sep='\t')
sent = set(df['Sentence'].to_list())
clean_sent = set()
for s in sent:
    # replace all punctuation with space
    s = re.sub("[^0-9a-zA-Z']+", ' ', s)
    s = s.strip()
    if "logo" not in s and "Logo" not in s and "may refer to" not in s and "disambiguation" not in s and "For other uses" not in s and "For all other uses" not in s and "redirects" not in s and "See " not in s:
        clean_sent.add(s)
sent = clean_sent

with open('sentences-3.txt', 'w') as fh:
    for s in sent:
        toks = s.split(' ')
        if len(toks) == 3:
            fh.write('{}\n'.format(s))

with open('sentences-4.txt', 'w') as fh:
    for s in sent:
        toks = s.split(' ')
        if len(toks) == 4:
            fh.write('{}\n'.format(s))

with open('sentences-5.txt', 'w') as fh:
    for s in sent:
        toks = s.split(' ')
        if len(toks) == 5:
            fh.write('{}\n'.format(s))

with open('sentences-6.txt', 'w') as fh:
    for s in sent:
        toks = s.split(' ')
        if len(toks) == 6:
            fh.write('{}\n'.format(s))

with open('sentences-7.txt', 'w') as fh:
    for s in sent:
        toks = s.split(' ')
        if len(toks) == 7:
            fh.write('{}\n'.format(s))

with open('sentences-8.txt', 'w') as fh:
    for s in sent:
        toks = s.split(' ')
        if len(toks) == 8:
            fh.write('{}\n'.format(s))