# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle as pkl

from fairseq import options, utils, checkpoint_utils
from fairseq.models import (
    FairseqDecoder,
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.lstm import (
    Embedding,
    LSTM,
    LSTMCell,
    Linear,
)
from fairseq.modules import AdaptiveSoftmax

from espresso.modules import speech_attention
from espresso.tools.scheduled_sampling_rate_scheduler import ScheduledSamplingRateScheduler
import espresso.tools.utils as speech_utils

global svcca_path


DEFAULT_MAX_SOURCE_POSITIONS = 1e5
DEFAULT_MAX_TARGET_POSITIONS = 1e5


logger = logging.getLogger(__name__)


@register_model("speech_lstm")
class SpeechLSTMModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder, pretrained_lm=None):
        super().__init__(encoder, decoder)
        self.num_updates = 0
        self.pretrained_lm = pretrained_lm
        if pretrained_lm is not None:
            assert isinstance(self.pretrained_lm, FairseqDecoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--dropout", type=float, metavar="D",
                            help="dropout probability")
        parser.add_argument("--encoder-conv-channels", type=str, metavar="EXPR",
                            help="list of encoder convolution\'s out channels")
        parser.add_argument("--encoder-conv-kernel-sizes", type=str, metavar="EXPR",
                            help="list of encoder convolution\'s kernel sizes")
        parser.add_argument("--encoder-conv-strides", type=str, metavar="EXPR",
                            help="list of encoder convolution\'s strides")
        parser.add_argument("--encoder-rnn-hidden-size", type=int, metavar="N",
                            help="encoder rnn\'s hidden size")
        parser.add_argument("--encoder-rnn-layers", type=int, metavar="N",
                            help="number of rnn encoder layers")
        parser.add_argument("--encoder-rnn-bidirectional",
                            type=lambda x: options.eval_bool(x),
                            help="make all rnn layers of encoder bidirectional")
        parser.add_argument("--encoder-rnn-residual",
                            type=lambda x: options.eval_bool(x),
                            help="create residual connections for rnn encoder "
                            "layers (starting from the 2nd layer), i.e., the actual "
                            "output of such layer is the sum of its input and output")
        parser.add_argument("--decoder-embed-dim", type=int, metavar="N",
                            help="decoder embedding dimension")
        parser.add_argument("--decoder-embed-path", type=str, metavar="STR",
                            help="path to pre-trained decoder embedding")
        parser.add_argument("--decoder-freeze-embed", action="store_true",
                            help="freeze decoder embeddings")
        parser.add_argument("--decoder-hidden-size", type=int, metavar="N",
                            help="decoder hidden size")
        parser.add_argument("--decoder-layers", type=int, metavar="N",
                            help="number of decoder layers")
        parser.add_argument("--decoder-out-embed-dim", type=int, metavar="N",
                            help="decoder output embedding dimension")
        parser.add_argument("--decoder-rnn-residual",
                            type=lambda x: options.eval_bool(x),
                            help="create residual connections for rnn decoder "
                            "layers (starting from the 2nd layer), i.e., the actual "
                            "output of such layer is the sum of its input and output")
        parser.add_argument("--attention-type", type=str, metavar="STR",
                            choices=["bahdanau", "luong","bahdanauMOMA"],
                            help="attention type")
        parser.add_argument("--attention-dim", type=int, metavar="N",
                            help="attention dimension")
        parser.add_argument("--need-attention", action="store_true",
                            help="need to return attention tensor for the caller")
        parser.add_argument("--adaptive-softmax-cutoff", metavar="EXPR",
                            help="comma separated list of adaptive softmax cutoff points. "
                                 "Must be used with adaptive_loss criterion")
        parser.add_argument("--share-decoder-input-output-embed",
                            type=lambda x: options.eval_bool(x),
                            help="share decoder input and output embeddings")
        parser.add_argument("--pretrained-lm-checkpoint", type=str, metavar="STR",
                            help="path to load checkpoint from pretrained language model(LM), "
                            "which will be present and kept fixed during training.")

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument("--encoder-rnn-dropout-in", type=float, metavar="D",
                            help="dropout probability for encoder rnn\'s input")
        parser.add_argument("--encoder-rnn-dropout-out", type=float, metavar="D",
                            help="dropout probability for encoder rnn\'s output")
        parser.add_argument("--decoder-dropout-in", type=float, metavar="D",
                            help="dropout probability for decoder input embedding")
        parser.add_argument("--decoder-dropout-out", type=float, metavar="D",
                            help="dropout probability for decoder output")

        # Scheduled sampling options
        parser.add_argument("--scheduled-sampling-probs", type=lambda p: options.eval_str_list(p),
                            metavar="P_1,P_2,...,P_N", default=[1.0],
                            help="scheduled sampling probabilities of sampling the truth "
                            "labels for N epochs starting from --start-schedule-sampling-epoch; "
                            "all later epochs using P_N")
        parser.add_argument("--start-scheduled-sampling-epoch", type=int,
                            metavar="N", default=1,
                            help="start scheduled sampling from the specified epoch")
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        max_source_positions = getattr(args, "max_source_positions", DEFAULT_MAX_SOURCE_POSITIONS)
        max_target_positions = getattr(args, "max_target_positions", DEFAULT_MAX_TARGET_POSITIONS)

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        # separate decoder input embeddings
        pretrained_decoder_embed = None
        if args.decoder_embed_path:
            pretrained_decoder_embed = load_pretrained_embedding_from_file(
                args.decoder_embed_path,
                task.target_dictionary,
                args.decoder_embed_dim
            )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                "--share-decoder-input-output-embed requires "
                "--decoder-embed-dim to match --decoder-out-embed-dim"
            )

        if args.decoder_freeze_embed:
            pretrained_decoder_embed.weight.requires_grad = False

        out_channels = speech_utils.eval_str_nested_list_or_tuple(args.encoder_conv_channels, type=int)
        kernel_sizes = speech_utils.eval_str_nested_list_or_tuple(args.encoder_conv_kernel_sizes, type=int)
        strides = speech_utils.eval_str_nested_list_or_tuple(args.encoder_conv_strides, type=int)
        logger.info("input feature dimension: {}, channels: {}".format(task.feat_dim, task.feat_in_channels))
        assert task.feat_dim % task.feat_in_channels == 0
        conv_layers = ConvBNReLU(
            out_channels, kernel_sizes, strides, in_channels=task.feat_in_channels,
        ) if out_channels is not None else None

        rnn_encoder_input_size = task.feat_dim // task.feat_in_channels
        if conv_layers is not None:
            for stride in strides:
                if isinstance(stride, (list, tuple)):
                    assert len(stride) > 0
                    s = stride[1] if len(stride) > 1 else stride[0]
                else:
                    assert isinstance(stride, int)
                    s = stride
                rnn_encoder_input_size = (rnn_encoder_input_size + s - 1) // s
            rnn_encoder_input_size *= out_channels[-1]
        else:
            rnn_encoder_input_size = task.feat_dim

        scheduled_sampling_rate_scheduler = ScheduledSamplingRateScheduler(
            args.scheduled_sampling_probs, args.start_scheduled_sampling_epoch,
        )

        encoder = SpeechLSTMEncoder(
            conv_layers_before=conv_layers,
            input_size=rnn_encoder_input_size,
            hidden_size=args.encoder_rnn_hidden_size,
            num_layers=args.encoder_rnn_layers,
            dropout_in=args.encoder_rnn_dropout_in,
            dropout_out=args.encoder_rnn_dropout_out,
            bidirectional=args.encoder_rnn_bidirectional,
            residual=args.encoder_rnn_residual,
            max_source_positions=max_source_positions,
        )
        decoder = SpeechLSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            encoder_output_units=encoder.output_units,
            attn_type=args.attention_type,
            attn_dim=args.attention_dim,
            need_attn=args.need_attention,
            residual=args.decoder_rnn_residual,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == "adaptive_loss" else None
            ),
            max_target_positions=max_target_positions,
            scheduled_sampling_rate_scheduler=scheduled_sampling_rate_scheduler,
        )
        pretrained_lm = None
        if args.pretrained_lm_checkpoint:
            logger.info("loading pretrained LM from {}".format(args.pretrained_lm_checkpoint))
            pretrained_lm = checkpoint_utils.load_model_ensemble(
                args.pretrained_lm_checkpoint, task=task)[0][0]
            pretrained_lm.make_generation_fast_()
            # freeze pretrained model
            for param in pretrained_lm.parameters():
                param.requires_grad = False
        return cls(encoder, decoder, pretrained_lm)

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        epoch=1,
    ):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out,
            incremental_state=incremental_state, epoch=epoch,
        )
        return decoder_out

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
        super().set_num_updates(num_updates)

    def max_positions(self):
        """Maximum length supported by the model."""
        return (
            self.encoder.max_positions(),
            self.decoder.max_positions() if self.pretrained_lm is None else
            min(self.decoder.max_positions(), self.pretrained_lm.max_positions()),
        )

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions() if self.pretrained_lm is None else \
            min(self.decoder.max_positions(), self.pretrained_lm.max_positions())


class ConvBNReLU(nn.Module):
    """Sequence of convolution-BatchNorm-ReLU layers."""
    def __init__(self, out_channels, kernel_sizes, strides, in_channels=1):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.in_channels = in_channels

        num_layers = len(out_channels)
        assert num_layers == len(kernel_sizes) and num_layers == len(strides)

        self.convolutions = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        for i in range(num_layers):
            self.convolutions.append(
                Convolution2d(
                    self.in_channels if i == 0 else self.out_channels[i-1],
                    self.out_channels[i],
                    self.kernel_sizes[i], self.strides[i]))
            self.batchnorms.append(nn.BatchNorm2d(out_channels[i]))

    def output_lengths(self, in_lengths):
        out_lengths = in_lengths
        for stride in self.strides:
            if isinstance(stride, (list, tuple)):
                assert len(stride) > 0
                s = stride[0]
            else:
                assert isinstance(stride, int)
                s = stride
            out_lengths = (out_lengths + s - 1) // s
        return out_lengths

    def forward(self, src, src_lengths):
        # B X T X C -> B X (input channel num) x T X (C / input channel num)
        #conv_vec = [] #anna
        #conv_vec.append([]) #anna
        print(src.shape,"/n",src_lengths)
        x = src.view(
            src.size(0), src.size(1), self.in_channels, src.size(2) // self.in_channels,
        ).transpose(1, 2)
        counter = 0
        for conv, bn in zip(self.convolutions, self.batchnorms):
            x = F.relu(bn(conv(x)))
            #saving each layer embedding
            # step 1: move to another variable 
            # step 2: Do the required post processing
            y = x
            # B X (output channel num) x T X C' -> B X T X (output channel num) X C'
            #y = y.transpose(1, 2)
            #print(" After transposing y is {}".format(y.shape))
            #y = y.contiguous().view(y.size(0), y.size(1), y.size(2) * y.size(3))
            print("Conv y is {}".format(y.shape))

            #y_lengths = self.output_lengths(src_lengths)
            #print("output length src_length {}".format(y_lengths))
            #padding_mask = ~speech_utils.sequence_mask(y_lengths, y.size(1))
            #if padding_mask.any():
            #    y = y.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            emb = {}
            emb[ "conv_"+str(counter) ]= y
            y_temp=y
            global svcca_path
            svcca_path = "/share/mini1/sw/spl/espresso/svcca_code/svcca/exp/tmp_embeddings/"


            filename = "conv_"+str(counter)+"_cuda.pkl"
            print("Final embedding shape at {} is {}".format(filename,y.shape))
            filename = os.path.join(svcca_path, filename)
            torch.save(emb,filename)
            emb[ "conv_"+str(counter) ]= y_temp.detach().cpu().numpy()
            filename = "conv_"+str(counter)+"_cpu.pt"
            filename = os.path.join(svcca_path, filename)
            torch.save(emb,filename)
            counter = counter+1
            #end of cnn embeddings save
        # B X (output channel num) x T X C' -> B X T X (output channel num) X C'
        x = x.transpose(1, 2)
        # B X T X (output channel num) X C' -> B X T X C
        x = x.contiguous().view(x.size(0), x.size(1), x.size(2) * x.size(3))

        x_lengths = self.output_lengths(src_lengths)
        padding_mask = ~speech_utils.sequence_mask(x_lengths, x.size(1))
        if padding_mask.any():
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return x, x_lengths, padding_mask



class SpeechLSTMEncoder(FairseqEncoder):
    """LSTM encoder."""
    def __init__(
        self, conv_layers_before=None, input_size=83, hidden_size=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, bidirectional=False,
        residual=False, left_pad=False, padding_value=0.,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
    ):
        super().__init__(None)  # no src dictionary
        self.conv_layers_before = conv_layers_before
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.residual = residual
        self.max_source_positions = max_source_positions

        self.lstm = nn.ModuleList([
            LSTM(
                input_size=input_size if layer == 0 else 2 * hidden_size if self.bidirectional else hidden_size,
                hidden_size=hidden_size,
                bidirectional=bidirectional,
            )
            for layer in range(num_layers)
        ])
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def output_lengths(self, in_lengths):
        return in_lengths if self.conv_layers_before is None \
            else self.conv_layers_before.output_lengths(in_lengths)

    def forward(
        self,
        src_tokens: Tensor,
        src_lengths: Tensor,
        enforce_sorted: bool = True,
        **unused,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of
                shape `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of
                shape `(batch)`
            enforce_sorted (bool, optional): if True, `src_tokens` is
                expected to contain sequences sorted by length in a
                decreasing order. If False, this condition is not
                required. Default: True.
        """
        if self.left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            src_tokens = speech_utils.convert_padding_direction(
                src_tokens,
                src_lengths,
                left_to_right=True,
            )

        if self.conv_layers_before is not None:
            x, src_lengths, padding_mask = self.conv_layers_before(src_tokens, src_lengths)
        else:
            x, padding_mask = src_tokens, \
                ~speech_utils.sequence_mask(src_lengths, src_tokens.size(1))

        bsz, seqlen = x.size(0), x.size(1)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        state_size = 2 if self.bidirectional else 1, bsz, self.hidden_size
        h0, c0 = x.new_zeros(*state_size), x.new_zeros(*state_size)
        counter = 0
        print("data input size after transpose before lstm encode {}".format(x.shape))
        for i in range(len(self.lstm)):
            if self.residual and i > 0:  # residual connection starts from the 2nd layer
                prev_x = x
            # pack embedded source tokens into a PackedSequence
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, src_lengths.data, enforce_sorted=enforce_sorted
            )

            # apply LSTM
            packed_outs, (_, _) = self.lstm[i](packed_x, (h0, c0))


            #print("lstm encoder intermediate before padding shape {}".format(packed_outs[0].shape,))
            #print("lstm encoder intermediate before padding  {}".format(packed_outs))
            # unpack outputs and apply dropout
            x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value*1.0)
            #print("lstm encoder intermediate shape {}".format(x.shape))
            #save embeddings here
            y=x
            y_temp = y
            emb = {}
            emb[ "lstm_"+str(counter) ]= y
            global svcca_path
            svcca_path = "/share/mini1/sw/spl/espresso/svcca_code/svcca/exp/tmp_embeddings/"
            filename = "lstm_"+str(counter)+"_cuda.pkl"
            print("Final embedding shape at {} is {}".format(filename,y.shape))
            filename = os.path.join(svcca_path, filename)
            torch.save(emb,filename)
            emb[ "lstm_"+str(counter) ]= y_temp.detach().cpu().numpy()
            filename = "lstm_"+str(counter)+"_cpu.pt"
            filename = os.path.join(svcca_path, filename)
            torch.save(emb,filename)
            counter = counter+1

            if i < len(self.lstm) - 1:  # not applying dropout for the last layer
                x = F.dropout(x, p=self.dropout_out, training=self.training)
            x = x + prev_x if self.residual and i > 0 else x
        assert list(x.size()) == [seqlen, bsz, self.output_units]
        #x.transpose (BxTxC) save output here
        encoder_padding_mask = padding_mask.t()
        print("lstm encoder output shape {}".format(x.shape))
        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask if encoder_padding_mask.any() else None,  # T x B
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=src_lengths,  # B
        )

    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        encoder_padding_mask = encoder_out.encoder_padding_mask.index_select(1, new_order) \
            if encoder_out.encoder_padding_mask is not None else None
        return EncoderOut(
            encoder_out=encoder_out.encoder_out.index_select(1, new_order),
            encoder_padding_mask=encoder_padding_mask,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=encoder_out.src_lengths.index_select(0, new_order),
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions

#pkl_fil = "conv_%s"%epoch

class SpeechLSTMDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, encoder_output_units=0,
        attn_type=None, attn_dim=0, need_attn=False, residual=False, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
        max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
        scheduled_sampling_rate_scheduler=None,
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        if attn_type is None or attn_type.lower() == "none":
            # no attention, no encoder output needed (language model case)
            need_attn = False
            encoder_output_units = 0
        self.need_attn = need_attn
        self.residual = residual
        self.max_target_positions = max_target_positions
        self.num_layers = num_layers

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units

        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=encoder_output_units + (embed_dim if layer == 0 else hidden_size),
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])

        if attn_type is None or attn_type.lower() == "none":
            self.attention = None
        elif attn_type.lower() == "bahdanau":
            self.attention = speech_attention.BahdanauAttention(
                hidden_size, encoder_output_units, attn_dim,
            )
        elif attn_type.lower() == "bahdanaumoma":
            self.attention = speech_attention.BahdanauAttentionMOMA(
                hidden_size, encoder_output_units, attn_dim,
            )
        elif attn_type.lower() == "luong":
            self.attention = speech_attention.LuongAttention(
                hidden_size, encoder_output_units,
            )
        else:
            raise ValueError("unrecognized attention type.")

        if hidden_size + encoder_output_units != out_embed_dim:
            self.additional_fc = Linear(hidden_size + encoder_output_units, out_embed_dim)

        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(
                num_embeddings, hidden_size, adaptive_softmax_cutoff, dropout=dropout_out
            )
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

        self.scheduled_sampling_rate_scheduler = scheduled_sampling_rate_scheduler

    def get_cached_state(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]):
        cached_state = self.get_incremental_state(incremental_state, "cached_state")
        assert cached_state is not None
        prev_hiddens_ = cached_state["prev_hiddens"]
        assert prev_hiddens_ is not None
        prev_cells_ = cached_state["prev_cells"]
        assert prev_cells_ is not None
        prev_hiddens = [prev_hiddens_[i] for i in range(self.num_layers)]
        prev_cells = [prev_cells_[j] for j in range(self.num_layers)]
        input_feed = cached_state["input_feed"]  # can be None for decoder-only language models
        return prev_hiddens, prev_cells, input_feed

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        **kwargs,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (EncoderOut, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - attention weights of shape `(batch, tgt_len, src_len)`
        """
        if self.scheduled_sampling_rate_scheduler is not None:
            epoch = kwargs.get("epoch", 1)
            sampling_prob = self.scheduled_sampling_rate_scheduler.step(epoch)
            if sampling_prob < 1.0:  # apply scheduled sampling
                return self._forward_with_scheduled_sampling(
                    prev_output_tokens, sampling_prob, encoder_out=encoder_out,
                    incremental_state={},  # use empty dict to preserve forward state
                )

        x, attn_scores = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state,
        )
        y = self.output_layer(x)
        # embeding saving
        counter = 0
        print("Decoder output(x) {} x={} attn_score={}".format(self.output_layer(x).shape, x.shape,attn_scores.shape))
        emb = {}
        emb[ "decoder_"+str(counter) ]= y
        y_temp=y
        global svcca_path
        svcca_path = "/share/mini1/sw/spl/espresso/svcca_code/svcca/exp/tmp_embeddings/"
        filename = "decoder_"+str(counter)+"_cuda.pkl"
        print("Final embedding shape at {} is {}".format(filename,y.shape))
        filename = os.path.join(svcca_path, filename)
        torch.save(emb,filename)
        emb[ "decoder_"+str(counter) ]= y_temp.detach().cpu().numpy()
        filename = "decoder_"+str(counter)+"_cpu.pt"
        filename = os.path.join(svcca_path, filename)
        torch.save(emb,filename)
        counter = counter+1
        y = attn_scores
        counter = 0
        emb = {}
        emb[ "attention_"+str(counter) ]= y
        y_temp=y
        svcca_path = "/share/mini1/sw/spl/espresso/svcca_code/svcca/exp/tmp_embeddings/"
        filename = "attention_"+str(counter)+"_cuda.pkl"
        print("Final embedding shape at {} is {}".format(filename,y.shape))
        filename = os.path.join(svcca_path, filename)
        torch.save(emb,filename)
        emb[ "attention_"+str(counter) ]= y_temp.detach().cpu().numpy()
        filename = "attention_"+str(counter)+"_cpu.pt"
        filename = os.path.join(svcca_path, filename)
        torch.save(emb,filename)
        counter = counter+1


        return self.output_layer(x), attn_scores

    def _forward_with_scheduled_sampling(
        self,
        prev_output_tokens,
        sampling_prob,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        bsz, seqlen = prev_output_tokens.size()
        outs = []
        pred = None
        for step in range(seqlen):
            if step > 0:
                sampling_mask = torch.rand(
                    [bsz, 1], device=prev_output_tokens.device,
                ).lt(sampling_prob)
                feed_tokens = torch.where(
                    sampling_mask, prev_output_tokens[:, step:step + 1], pred,
                )
            else:
                feed_tokens = prev_output_tokens[:, step:step + 1]  # B x 1
            x, _ = self.extract_features(feed_tokens, encoder_out, incremental_state)
            x = self.output_layer(x)  # B x 1 x V
            outs.append(x)
            pred = x.argmax(-1)  # B x 1
        x = torch.cat(outs, dim=1)  # B x T x V
        # ignore attention scores
        print("Decoder scheduled sampling output x={} ".format( x.shape))
        return x, None

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        **unused,
    ):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - attention weights of shape `(batch, tgt_len, src_len)`
        """
        # get outputs from encoder
        if encoder_out is not None:
            assert self.attention is not None
            encoder_outs = encoder_out.encoder_out
            encoder_padding_mask = encoder_out.encoder_padding_mask
        else:
            encoder_outs = torch.empty(0)
            encoder_padding_mask = torch.empty(0)
        srclen = encoder_outs.size(0)

        if incremental_state is not None and len(incremental_state) > 0:
            prev_output_tokens = prev_output_tokens[:, -1:]

        bsz, seqlen = prev_output_tokens.size()

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        if incremental_state is not None and len(incremental_state) > 0:
            prev_hiddens, prev_cells, input_feed = self.get_cached_state(incremental_state)
        else:
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(self.num_layers)]
            prev_cells = [zero_state for i in range(self.num_layers)]
            input_feed = x.new_zeros(bsz, self.encoder_output_units) \
                if encoder_out is not None else None

        attn_scores = x.new_zeros(srclen, seqlen, bsz) if encoder_out is not None else None
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((x[j, :, :], input_feed), dim=1)
            else:
                input = x[j, :, :]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))
                if self.residual and i > 0:  # residual connection starts from the 2nd layer
                    prev_layer_hidden = input[:, :hidden.size(1)]

                # compute and apply attention using the 1st layer's hidden state
                if encoder_out is not None:
                    if i == 0:
                        assert attn_scores is not None
                        context, attn_scores[:, j, :], _ = self.attention(
                            hidden, encoder_outs, encoder_padding_mask,
                        )

                    # hidden state concatenated with context vector becomes the
                    # input to the next layer
                    input = torch.cat((hidden, context), dim=1)
                else:
                    input = hidden
                input = F.dropout(input, p=self.dropout_out, training=self.training)
                if self.residual and i > 0:
                    if encoder_out is not None:
                        hidden_sum = input[:, :hidden.size(1)] + prev_layer_hidden
                        input = torch.cat((hidden_sum, input[:, hidden.size(1):]), dim=1)
                    else:
                        input = input + prev_layer_hidden

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # input feeding
            if input_feed is not None:
                input_feed = context

            # save final output
            outs.append(input)

        # Stack all the necessary tensors together and store
        prev_hiddens_tensor = torch.stack(prev_hiddens)
        prev_cells_tensor = torch.stack(prev_cells)
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {"prev_hiddens": prev_hiddens_tensor, "prev_cells": prev_cells_tensor, "input_feed": input_feed}
        )
        self.set_incremental_state(incremental_state, "cached_state", cache_state)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, -1)
        assert x.size(2) == self.hidden_size + self.encoder_output_units

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if hasattr(self, "additional_fc") and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and encoder_out is not None and self.need_attn:
            assert attn_scores is not None
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        return x, attn_scores

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return self.fc_out(features)
        else:
            return features

    def reorder_state(self, state: List[Tensor], new_order):
        return [
            state_i.index_select(0, new_order) if state_i is not None else None
            for state_i in state
        ]

    def reorder_incremental_state(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]], new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        if incremental_state is None or len(incremental_state) == 0:
            return
        prev_hiddens, prev_cells, input_feed = self.get_cached_state(incremental_state)
        cached_state = (prev_hiddens, prev_cells, [input_feed])
        new_state = [self.reorder_state(state, new_order) for state in cached_state]
        prev_hiddens_tensor = torch.stack(new_state[0])
        prev_cells_tensor = torch.stack(new_state[1])
        cached_state_new = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {"prev_hiddens": prev_hiddens_tensor, "prev_cells": prev_cells_tensor, "input_feed": new_state[2][0]}
        )
        self.set_incremental_state(incremental_state, "cached_state", cached_state_new),
        return

    def masked_copy_incremental_state(self, incremental_state, another_cached_state, mask):
        if incremental_state is None or len(incremental_state) == 0:
            assert another_cached_state is None or len(another_cached_state) == 0
            return
        prev_hiddens, prev_cells, input_feed = self.get_cached_state(incremental_state)
        cached_state = (prev_hiddens, prev_cells, [input_feed])
        another_cached_state = (another_cached_state[0], another_cached_state[1], [another_cached_state[2]])

        def mask_copy_state(state: List[Tensor], another_state: List[Tensor]):
            new_state = []
            for state_i, another_state_i in zip(state, another_state):
                if state_i is None:
                    assert another_state_i is None
                    new_state.append(None)
                else:
                    assert state_i.size(0) == mask.size(0) and another_state_i is not None and \
                        state_i.size() == another_state_i.size()
                    mask_unsqueezed = mask
                    for _ in range(1, len(state_i.size())):
                        mask_unsqueezed = mask_unsqueezed.unsqueeze(-1)
                    new_state.append(torch.where(mask_unsqueezed, state_i, another_state_i))
            return new_state

        new_state = [
            mask_copy_state(state, another_state)
            for (state, another_state) in zip(cached_state, another_cached_state)
        ]
        prev_hiddens_tensor = torch.stack(new_state[0])
        prev_cells_tensor = torch.stack(new_state[1])
        cached_state_new = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {"prev_hiddens": prev_hiddens_tensor, "prev_cells": prev_cells_tensor, "input_feed": new_state[2][0]}
        )
        self.set_incremental_state(incremental_state, "cached_state", cached_state_new)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Convolution2d(in_channels, out_channels, kernel_size, stride):
    if isinstance(kernel_size, (list, tuple)):
        if len(kernel_size) != 2:
            assert len(kernel_size) == 1
            kernel_size = (kernel_size[0], kernel_size[0])
    else:
        assert isinstance(kernel_size, int)
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, (list, tuple)):
        if len(stride) != 2:
            assert len(stride) == 1
            stride = (stride[0], stride[0])
    else:
        assert isinstance(stride, int)
        stride = (stride, stride)
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
    m = nn.Conv2d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding,
    )
    return m


@register_model_architecture("speech_lstm", "speech_lstm")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.4)
    args.encoder_conv_channels = getattr(
        args, "encoder_conv_channels", "[64, 64, 128, 128]",
    )
    args.encoder_conv_kernel_sizes = getattr(
        args, "encoder_conv_kernel_sizes", "[(3, 3), (3, 3), (3, 3), (3, 3)]",
    )
    args.encoder_conv_strides = getattr(
        args, "encoder_conv_strides", "[(1, 1), (2, 2), (1, 1), (2, 2)]",
    )
    args.encoder_rnn_hidden_size = getattr(args, "encoder_rnn_hidden_size", 320)
    args.encoder_rnn_layers = getattr(args, "encoder_rnn_layers", 3)
    args.encoder_rnn_bidirectional = getattr(args, "encoder_rnn_bidirectional", True)
    args.encoder_rnn_residual = getattr(args, "encoder_rnn_residual", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 48)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_freeze_embed = getattr(args, "decoder_freeze_embed", False)
    args.decoder_hidden_size = getattr(args, "decoder_hidden_size", 320)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 960)
    args.decoder_rnn_residual = getattr(args, "decoder_rnn_residual", True)
    args.attention_type = getattr(args, "attention_type", "bahdanau")
    args.attention_dim = getattr(args, "attention_dim", 320)
    args.need_attention = getattr(args, "need_attention", False)
    args.encoder_rnn_dropout_in = getattr(args, "encoder_rnn_dropout_in", args.dropout)
    args.encoder_rnn_dropout_out = getattr(args, "encoder_rnn_dropout_out", args.dropout)
    args.decoder_dropout_in = getattr(args, "decoder_dropout_in", args.dropout)
    args.decoder_dropout_out = getattr(args, "decoder_dropout_out", args.dropout)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
    args.pretrained_lm_checkpoint = getattr(args, "pretrained_lm_checkpoint", None)


@register_model_architecture("speech_lstm", "speech_conv_lstm_wsj")
def conv_lstm_wsj(args):
    base_architecture(args)


@register_model_architecture("speech_lstm", "speech_conv_lstm_librispeech")
def speech_conv_lstm_librispeech(args):
    args.dropout = getattr(args, "dropout", 0.3)
    args.encoder_rnn_hidden_size = getattr(args, "encoder_rnn_hidden_size", 1024)
    args.encoder_rnn_layers = getattr(args, "encoder_rnn_layers", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_hidden_size = getattr(args, "decoder_hidden_size", 1024)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 3072)
    args.decoder_rnn_residual = getattr(args, "decoder_rnn_residual", True)
    args.attention_type = getattr(args, "attention_type", "bahdanau")
    args.attention_dim = getattr(args, "attention_dim", 512)
    base_architecture(args)


@register_model_architecture("speech_lstm", "speech_conv_lstm_swbd")
def speech_conv_lstm_swbd(args):
    args.dropout = getattr(args, "dropout", 0.5)
    args.encoder_rnn_hidden_size = getattr(args, "encoder_rnn_hidden_size", 640)
    args.encoder_rnn_layers = getattr(args, "encoder_rnn_layers", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 640)
    args.decoder_hidden_size = getattr(args, "decoder_hidden_size", 640)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 1920)
    args.decoder_rnn_residual = getattr(args, "decoder_rnn_residual", True)
    args.attention_type = getattr(args, "attention_type", "bahdanau")
    args.attention_dim = getattr(args, "attention_dim", 640)
    base_architecture(args)


@register_model_architecture("speech_lstm", "speech_conv_lstm_MOMA_swbd")
def speech_conv_lstm_swbd(args):
    args.dropout = getattr(args, "dropout", 0.5)
    args.encoder_rnn_hidden_size = getattr(args, "encoder_rnn_hidden_size", 640)
    args.encoder_rnn_layers = getattr(args, "encoder_rnn_layers", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 640)
    args.decoder_hidden_size = getattr(args, "decoder_hidden_size", 640)
    args.decoder_layers = getattr(args, "decoder_layers", 3)
    args.decoder_out_embed_dim = getattr(args, "decoder_out_embed_dim", 1920)
    args.decoder_rnn_residual = getattr(args, "decoder_rnn_residual", True)
    args.attention_type = getattr(args, "attention_type", "bahdanauMOMA")
    args.attention_dim = getattr(args, "attention_dim", 640)
    base_architecture(args)

