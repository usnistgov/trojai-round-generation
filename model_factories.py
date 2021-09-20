# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import torch

import trojai.modelgen.architecture_factory

from transformers import AutoModel

ALL_ARCHITECTURE_KEYS = ['NerLinear']


class NerLinearModel(torch.nn.Module):
    def __init__(self, train_name, model_args, tran_config, num_labels, dropout_prob, ignore_index):
        super().__init__()
        self.num_labels = num_labels
        self.transformer = AutoModel.from_pretrained(train_name, config=tran_config, **model_args)
        self.dropout = torch.nn.Dropout(dropout_prob)
        out_dim = tran_config.hidden_size
        self.classifier = torch.nn.Linear(out_dim, self.num_labels)
        self.ignore_index = ignore_index

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        valid_output = self.dropout(sequence_output)
        emissions = self.classifier(valid_output)
    
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.ignore_index)
    
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = emissions.view(-1, self.num_labels)
                active_labels = torch.where(active_loss, labels.view(-1),
                                            torch.tensor(loss_fct.ignore_index).type_as(labels))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(emissions.view(-1, self.num_labels), labels.view(-1))
    
        return loss, emissions


def arch_factory_kwargs_generator(train_dataset_desc, clean_test_dataset_desc, triggered_test_dataset_desc):
    # Note: the arch_factory_kwargs_generator returns a dictionary, which is used as kwargs input into an
    #  architecture factory.  Here, we allow the input-dimension and the pad-idx to be set when the model gets
    #  instantiated.  This is useful because these indices and the vocabulary size are not known until the
    #  vocabulary is built.
    # TODO figure out if I can remove this
    output_dict = dict(input_size=train_dataset_desc['embedding_size'])
    return output_dict


class NerLinearFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, train_name, model_args, tran_config,  num_labels, dropout_prob, ignore_index):
        model = NerLinearModel(train_name, model_args, tran_config, num_labels, dropout_prob, ignore_index)
        return model


def get_factory(model_name: str):
    model = None
   
    if model_name == 'NerLinear':
        model = NerLinearFactory()
    else:
        raise RuntimeError('Invalid Model Architecture Name: {}'.format(model_name))

    return model
