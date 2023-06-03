#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This is an example how to extend torch ranker agent and use it for your own purpose.

In this example, we will just use a simple bag of words model.
"""
from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.torch_generator_agent import TorchGeneratorAgent

import torch
from torch import nn

class MyRankerModel(nn.Module):
    def __init__(self, opt, dictionary):

        super().__init__()
        self.gen_model = opt['pegah_model'].model
        self.gen_agent = opt['pegah_model']
       # self.hidden_dim = opt.get('hidden_dim', 512)
        self.dict = dictionary
       # self.encoder = nn.EmbeddingBag(len(self.dict), self.hidden_dim)

    def reorder_encoder_states(self, *args):
        return self.gen_model.reorder_encoder_states(*args)

    def decode_forced(self, *args):
        return self.gen_model.decode_forced(*args)

    def encoder(self, args):
        return self.gen_model.encoder(args)

    def build_model(self):
        # self.gen_model = self.opt['pegah_model']
        # return None
        # return self.opt['pegah_model']
        return self.gen_model
    # def encode_text(self, text_vecs):
    #     """
    #     This function encodes a text_vec to a text encoding.
    #     """
    #     return self.encoder(text_vecs)

    def comp_prob_score(self, batch, cand_vecs, cand_encs):
        # my version of computing the scores for each batch --> !I should have a fixed number of candidates
        return None
    #
    # def eval_step(self, batch):
    #     if self.model:
    #         print('Sofar so good')
    #     else:
    #         raise RuntimeError('Pegah: Need to Connect generator and ranker agents together.')

    # def train_step(self, batch):
    #     self.model.train_step(self, batch)




class Gen2rankAgent(TorchRankerAgent, TorchGeneratorAgent):
    """
    Example subclass of TorchRankerAgent.

    This particular implementation is a simple bag-of-words model, which demonstrates
    the minimum implementation requirements to make a new ranking model.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add CLI args.
        """
        super().add_cmdline_args(parser, partial_opt=partial_opt)
        arg_group = parser.add_argument_group('Generator2RankerModel Arguments')
        arg_group.add_argument('--gen-model', type=str) # This is a required field
        # TODO: add different types of generator as input
        return parser

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        """
        This function takes in a Batch object as well as a Tensor of candidate vectors.

        It must return a list of scores corresponding to the likelihood that the
        candidate vector at that index is the proper response. If `cand_encs` is not
        None (when we cache the encoding of the candidate vectors), you may use these
        instead of calling self.model on `cand_vecs`.
        """

        # print(self._v2t(batch.text_vec[1]))
        # print(batch)
        # plan: convert candidates to glove rep --> then get the scores from the generator
        #TODO: the new batch-vecs should be similar to the seq2seq version

        _, scores = self.rank_eval_label_candidates(batch, batch.batchsize, False) # second component is batch size
        # print('output of the rank eval labels scores')
        # print(tmp)
        # print(scores)
        return torch.Tensor(scores)
        # return tmp[1]
        # return scores

    def build_model(self):
        """
        This function is required to build the model and assign to the object
        `self.model`.
        """
        # print('[Pegah] It is fine the model will be build later')

        if self.opt['pegah_model']:
            return MyRankerModel(self.opt, self.dict) # TODO: This is tricky should be changed

        print('===================Alert============')
        return self

    # def
    # def eval_step(self, batch):
    #     print('Called inside my own evaluation')
    #     return None